using Newtonsoft.Json;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using UnityEngine;
using UnityEngine.AI;
using UnityEngine.Networking;
using Debug = UnityEngine.Debug;

[System.Serializable]
public class Message { public string role; public string content; }
[System.Serializable]
public class Choice { public Message message; }
[System.Serializable]
public class OpenAIResponse { public List<Choice> choices; }

public class ChatGPTManager : MonoBehaviour
{
    [Header("Policy")]
    public bool ignoreColors = true;

    [Header("Vision Link")]
    [SerializeField] private List<YoloObjectDetector> detectors = new();   // cars
    [SerializeField] private List<PropsDetector> propDetectors = new();    // props

    [Header("Guide Bot")]
    [SerializeField] private BotNavigator botNavigator;
    [SerializeField] private PathDrawer pathDrawer;
    [SerializeField] private float navSampleRadius = 3f;
    [SerializeField] private float approachOffset = 1.0f;
    [SerializeField] private string lastGuidedLabel = null;
    [SerializeField] private bool navDebugLogs = true;

    [Header("Vision De-dup")]
    [SerializeField] private float gridSizeMeters = 1.5f;
    [SerializeField] private bool preferObjectDescriptorId = true;

    [SerializeField] private float ttlSeconds = 1.5f;
    [SerializeField] private bool autoFindPropDetectors = true;

    private string visionContext = "nothing";
    private string openAI_APIKey;
    private readonly string openAI_Endpoint = "https://api.openai.com/v1/chat/completions";

    private static readonly Regex TriggerRegex = new(
        @"(?:\b(?:guide|take|lead|walk|go)\s+me\s+to\s+(?<target>.+)$)|(?:\b(?:go|walk)\s+to\s+(?<target>.+)$)|(?:\bnearest\s+(?<target>.+)$)",
        RegexOptions.IgnoreCase | RegexOptions.Compiled);

    [Serializable] private struct Seen { public DetectionInfo det; public float lastSeen; }
    private readonly Dictionary<string, Seen> _ttl = new();

    private static readonly string[] ColorWords = { "red", "gray", "grey", "blue", "green", "yellow", "white", "black", "orange", "purple", "pink", "brown", "cyan" };

    private static readonly Dictionary<string, string[]> LabelSynonyms = new()
    {
        { "bicycle", new[] { "bike", "cycle", "bicycles", "bikes" } },
        { "couch", new[] { "sofa", "settee" } },
        { "tv", new[] { "television", "screen", "monitor" } },
        { "dining table", new[] { "table", "desk" } },
        { "cell phone", new[] { "phone", "mobile" } },
        { "microwave", new[] { "oven" } },
        { "bottle", new[] { "bottles" } },
        { "bench", new[] { "benches" } },
        { "chair", new[] { "chairs", "seat" } },
        { "clock", new[] { "wall clock" } }
    };

    private enum NavTriggerMode { UserOnly, ModelAllowed, Disabled }
    [SerializeField] private NavTriggerMode navTrigger = NavTriggerMode.UserOnly;

    private bool HasNavigationIntent(string text)
        => !string.IsNullOrEmpty(text) && (text.Contains("guide me to", StringComparison.OrdinalIgnoreCase)
            || text.Contains("take me to", StringComparison.OrdinalIgnoreCase)
            || text.Contains("lead me to", StringComparison.OrdinalIgnoreCase));

    private bool HasDisambiguationClues(string text)
    {
        if (string.IsNullOrEmpty(text)) return false;
        var t = text.ToLowerInvariant();
        if (!ignoreColors && !string.IsNullOrEmpty(ExtractColor(t))) return true;
        if (Regex.IsMatch(t, @"\b(nearest|closest|farthest|furthest)\b")) return true;
        if (Regex.IsMatch(t, @"\b(left|right|center|middle|front|back|behind|ahead|near|far)\b")) return true;
        if (Regex.IsMatch(t, @"\b\d+(\.\d+)?\b")) return true;
        if (Regex.IsMatch(t, @"\b(first|second|third)\b")) return true;
        return false;
    }

    private void Awake() => openAI_APIKey = APIKeyLoader.LoadAPIKey();

    private void OnEnable()
    {
        if (autoFindPropDetectors && (propDetectors == null || propDetectors.Count == 0))
            propDetectors = FindObjectsByType<PropsDetector>(FindObjectsSortMode.None).ToList();

        if (detectors != null) foreach (var d in detectors) if (d != null) d.OnDetections += HandleDetections;
        if (propDetectors != null) foreach (var p in propDetectors) if (p != null) p.OnDetections += HandleDetections;
    }

    private void OnDisable()
    {
        if (detectors != null) foreach (var d in detectors) if (d != null) d.OnDetections -= HandleDetections;
        if (propDetectors != null) foreach (var p in propDetectors) if (p != null) p.OnDetections -= HandleDetections;
    }

    private string Canon(string s)
    {
        if (string.IsNullOrEmpty(s)) return "";
        return new string(s.ToLowerInvariant().Where(char.IsLetterOrDigit).ToArray());
    }

    private string[] CanonWords(string s)
        => Regex.Split(s ?? "", @"[^a-zA-Z0-9]+").Where(w => w.Length > 0).Select(w => w.ToLowerInvariant()).ToArray();

    private string MakeStableKey(DetectionInfo d)
    {
        if (preferObjectDescriptorId && TryGetDescriptorNear(d.worldPos, out var od, out var go) && go != null)
            return $"{d.label}|obj:{go.GetInstanceID()}";

        float g = Mathf.Max(0.25f, gridSizeMeters);
        int gx = Mathf.FloorToInt(d.worldPos.x / g);
        int gz = Mathf.FloorToInt(d.worldPos.z / g);
        return $"{d.label}|gx:{gx}|gz:{gz}";
    }

    private List<DetectionInfo> DedupDetections(IEnumerable<DetectionInfo> src)
    {
        var srcList = src as List<DetectionInfo> ?? (src?.ToList() ?? new List<DetectionInfo>());
        if (srcList.Count == 0) return new List<DetectionInfo>();

        var withSurface = srcList
            .Where(d => d.distance > 0 && !string.IsNullOrEmpty(d.surface) && d.surface != "nohit")
            .GroupBy(d => (d.label.Trim(), d.surface.Trim()))
            .Select(g => g.OrderBy(d => d.distance).First())
            .ToList();

        const float mergeMeters = 1.5f;
        var noSurface = srcList
            .Where(d => d.distance <= 0 || string.IsNullOrEmpty(d.surface) || d.surface == "nohit")
            .OrderBy(d => d.label)
            .ToList();

        var mergedNoSurface = new List<DetectionInfo>();
        foreach (var d in noSurface)
        {
            bool merged = false;
            for (int i = 0; i < mergedNoSurface.Count; i++)
            {
                var m = mergedNoSurface[i];
                if (!string.Equals(m.label, d.label, StringComparison.OrdinalIgnoreCase)) continue;
                if (Vector3.Distance(m.worldPos, d.worldPos) < mergeMeters)
                {
                    if (d.distance > 0 && (m.distance <= 0 || d.distance < m.distance))
                        mergedNoSurface[i] = d;
                    merged = true; break;
                }
            }
            if (!merged) mergedNoSurface.Add(d);
        }

        var result = new List<DetectionInfo>(withSurface);
        foreach (var d in mergedNoSurface)
        {
            bool covered = result.Any(r =>
                string.Equals(r.label, d.label, StringComparison.OrdinalIgnoreCase) &&
                (r.surface != "nohit" || r.distance > 0) &&
                Vector3.Distance(r.worldPos, d.worldPos) < 2.0f);
            if (!covered) result.Add(d);
        }
        return result;
    }

    private void HandleDetections(List<DetectionInfo> list)
    {
        float now = Time.time;

        if (list != null)
        {
            foreach (var d in list)
            {
                string key = MakeStableKey(d);
                if (!_ttl.TryGetValue(key, out var seen) || d.confidence > seen.det.confidence)
                    _ttl[key] = new Seen { det = d, lastSeen = now };
                else
                    _ttl[key] = new Seen { det = seen.det, lastSeen = now };
            }
        }

        var toRemove = new List<string>();
        foreach (var kv in _ttl)
            if (now - kv.Value.lastSeen > ttlSeconds)
                toRemove.Add(kv.Key);
        for (int i = 0; i < toRemove.Count; i++)
            _ttl.Remove(toRemove[i]);

        if (_ttl.Count == 0) { visionContext = "nothing"; return; }

        var dedupList = DedupDetections(_ttl.Values.Select(v => v.det));
        var groupedParts = dedupList.GroupBy(d => d.label).OrderByDescending(g => g.Count()).Select(g => $"{g.Count()} {g.Key}");
        visionContext = string.Join(", ", groupedParts);
    }

    private IEnumerable<DetectionInfo> GetAllDetections()
    {
        var all = new List<DetectionInfo>();
        if (detectors != null) foreach (var d in detectors) if (d != null) all.AddRange(d.GetLatestDetections());
        if (propDetectors != null) foreach (var p in propDetectors) if (p != null) all.AddRange(p.GetLatestDetections());
        return all;
    }

    private bool TryGetDescriptorNear(Vector3 pos, out ObjectDescriptor od, out GameObject go)
    {
        const float radius = 0.8f;
        var cols = Physics.OverlapSphere(pos, radius);
        foreach (var c in cols)
        {
            var d = c.GetComponentInParent<ObjectDescriptor>();
            if (d != null) { od = d; go = d.gameObject; return true; }
        }
        od = null; go = null; return false;
    }

    private string BuildSceneFacts(int maxItems = 20)
    {
        var all = DedupDetections(GetAllDetections());
        var ordered = all.GroupBy(d => d.label)
            .SelectMany(g => g.OrderBy(d => d.distance > 0 ? d.distance : 9999f).Take(3))
            .OrderBy(d => d.distance > 0 ? d.distance : 9999f)
            .Take(maxItems).ToList();

        if (ordered.Count == 0) return "nothing";
        var counts = ordered.GroupBy(d => d.label).Select(g => $"{g.Key}:{g.Count()}").ToList();
        return string.Join(", ", counts);
    }

    public IEnumerator SendMessageToOpenAI(string userMessage, Action<string> callback)
    {
        string sceneFacts = BuildSceneFacts(8);

        var requestData = new
        {
            model = "gpt-3.5-turbo",
            messages = new[] {
                new { role = "system", content =
                    "You are a helpful VR assistant inside a virtual gallery. " +
                    $"Detections summary: {visionContext}. " +
                    (string.IsNullOrEmpty(sceneFacts) ? "" : $"Scene facts: {sceneFacts}. ") +
                    "If the user asks about an object, describe it briefly before taking action." },
                new { role = "user", content = userMessage }
            },
            temperature = 0.7f
        };

        string jsonData = JsonConvert.SerializeObject(requestData);
        using UnityWebRequest request = new UnityWebRequest(openAI_Endpoint, "POST");
        request.uploadHandler = new UploadHandlerRaw(System.Text.Encoding.UTF8.GetBytes(jsonData));
        request.downloadHandler = new DownloadHandlerBuffer();
        request.SetRequestHeader("Content-Type", "application/json");
        request.SetRequestHeader("Authorization", $"Bearer {openAI_APIKey}");

        yield return request.SendWebRequest();

        if (request.result != UnityWebRequest.Result.Success)
        {
            Debug.LogError("OpenAI error: " + request.error);
            callback("Sorry, something went wrong.");
        }
        else
        {
            var response = JsonConvert.DeserializeObject<OpenAIResponse>(request.downloadHandler.text);
            string reply = response.choices[0].message.content.Trim();
            Debug.Log("GPT says: " + reply);

            if (navTrigger == NavTriggerMode.UserOnly && HasNavigationIntent(userMessage))
                TryHandleNavigation(userMessage);

            callback(reply);
        }
    }

    private string ExtractTargetPhrase(string msgLower)
    {
        var m = TriggerRegex.Match(msgLower);
        if (!m.Success) return null;
        var phrase = m.Groups["target"].Value.Trim();
        phrase = Regex.Replace(phrase, @"[^a-zA-Z0-9\s\-]", "");
        return phrase.Trim(' ', '.', '!', '?');
    }

    private bool IsPronounTarget(string phrase) =>
        !string.IsNullOrEmpty(phrase) && Regex.IsMatch(phrase, @"^(it|that|this)$", RegexOptions.IgnoreCase);

    private string ExtractColor(string msgLower)
    {
        foreach (var c in ColorWords)
            if (Regex.IsMatch(msgLower, $@"\b{Regex.Escape(c)}\b"))
                return c == "grey" ? "gray" : c;
        return null;
    }

    private string ResolveTargetLabelFromMessage(string msgLower, IEnumerable<string> visibleLabels)
    {
        string bestLabel = null; int bestLength = -1;
        foreach (var raw in visibleLabels)
        {
            if (string.IsNullOrWhiteSpace(raw)) continue;
            string label = raw.ToLowerInvariant();

            if (WordBoundaryContains(msgLower, label) && label.Length > bestLength) { bestLabel = raw; bestLength = label.Length; }

            if (LabelSynonyms.TryGetValue(label, out var syns))
                foreach (var s in syns)
                    if (WordBoundaryContains(msgLower, s.ToLowerInvariant()) && s.Length > bestLength)
                    { bestLabel = raw; bestLength = s.Length; }

            if (bestLength < 0)
                foreach (var token in msgLower.Split(' '))
                {
                    var t = token.Trim(); if (t.Length < 3) continue;
                    if (label.Contains(t) && WordBoundaryContains(msgLower, t) && t.Length > bestLength)
                    { bestLabel = raw; bestLength = t.Length; }
                }
        }
        return bestLabel;
    }

    private bool WordBoundaryContains(string textLower, string phraseLower)
        => Regex.IsMatch(textLower, $@"\b{Regex.Escape(phraseLower)}\b");

    private Vector3 ToNavmesh(Vector3 desired, float sampleRadius)
    {
        if (NavMesh.SamplePosition(desired, out var hit, sampleRadius, NavMesh.AllAreas))
            return hit.position;

        if (Physics.Raycast(desired + Vector3.up * 5f, Vector3.down, out var downHit, 20f))
        {
            if (NavMesh.SamplePosition(downHit.point, out var hit2, 2f, NavMesh.AllAreas))
                return hit2.position;
            return downHit.point;
        }
        return desired;
    }

    private bool TryGroundBelow(Vector3 src, out Vector3 ground)
    {
        if (Physics.Raycast(src + Vector3.up * 5f, Vector3.down, out var h, 50f, ~0, QueryTriggerInteraction.Ignore))
        { ground = h.point; return true; }
        ground = src; return false;
    }

    private Vector3 FindReachableAround(Vector3 objWorld, float inner = 0.3f, float outer = 5f, int ringSamples = 16)
    {
        TryGroundBelow(objWorld, out var basePt);

        if (NavMesh.SamplePosition(basePt, out var hit0, navSampleRadius, NavMesh.AllAreas))
        {
            var path0 = new NavMeshPath();
            var from0 = botNavigator ? botNavigator.transform.position :
                      (Camera.main ? Camera.main.transform.position : Vector3.zero);

            if (NavMesh.CalculatePath(from0, hit0.position, NavMesh.AllAreas, path0) &&
                path0.status == NavMeshPathStatus.PathComplete)
                return hit0.position;
        }

        for (float r = inner; r <= outer; r += 0.75f)
        {
            for (int i = 0; i < ringSamples; i++)
            {
                float ang = (i / (float)ringSamples) * Mathf.PI * 2f;
                var p = basePt + new Vector3(Mathf.Cos(ang), 0f, Mathf.Sin(ang)) * r;

                if (NavMesh.SamplePosition(p, out var hit, navSampleRadius, NavMesh.AllAreas))
                {
                    var path = new NavMeshPath();
                    var from = botNavigator ? botNavigator.transform.position :
                              (Camera.main ? Camera.main.transform.position : Vector3.zero);

                    if (NavMesh.CalculatePath(from, hit.position, NavMesh.AllAreas, path) &&
                        path.status == NavMeshPathStatus.PathComplete)
                        return hit.position;
                }
            }
        }
        return ToNavmesh(basePt, Mathf.Max(navSampleRadius, outer));
    }

    private Vector3 ComputeFrontApproachPoint(GameObject targetGO, Vector3 from, float standOff)
    {
        var rend = targetGO.GetComponentInChildren<Renderer>();
        Vector3 center = targetGO.transform.position;
        float radius = 0.75f;

        if (rend)
        {
            center = rend.bounds.center;
            radius = Mathf.Max(0.5f, rend.bounds.extents.magnitude * 0.35f);
        }

        Vector3 fromHoriz = new Vector3(from.x, center.y, from.z);
        Vector3 dir = center - fromHoriz; dir.y = 0f;
        if (dir.sqrMagnitude < 1e-4f) dir = targetGO.transform.forward; else dir.Normalize();

        Vector3 desired = center - dir * Mathf.Max(standOff, radius);
        return FindReachableAround(desired, 0.3f, 5f, 20);
    }

    private bool MatchesSurface(DetectionInfo d, string targetLabel)
    {
        var surf = (d.surface ?? "").ToLowerInvariant();
        if (string.IsNullOrEmpty(surf) || surf == "nohit") return true;

        var canon = targetLabel.ToLowerInvariant();
        var keys = new List<string> { canon };
        if (LabelSynonyms.TryGetValue(canon, out var syns))
            keys.AddRange(syns.Select(s => s.ToLowerInvariant()));

        return keys.Any(k => surf.Contains(k));
    }

    private static bool HasValidHit(DetectionInfo d)
        => d.distance > 0f && !string.Equals(d.surface, "nohit", StringComparison.OrdinalIgnoreCase);

    // NEW: robust scene search when there is no valid hit
    private GameObject FindBestTargetGO(string targetLabel)
    {
        string canonTarget = Canon(targetLabel);
        var words = CanonWords(targetLabel);

        GameObject best = null; int bestScore = 0;

        // 1) Prefer objects with colliders on Detectables layer
        var cols = FindObjectsByType<Collider>(FindObjectsSortMode.None);
        foreach (var col in cols)
        {
            var go = col.transform.root.gameObject;
            if (!go.activeInHierarchy) continue;

            // prefer Detectables layer if you use it
            int layerBonus = (LayerMask.LayerToName(go.layer).Equals("Detectables", StringComparison.OrdinalIgnoreCase)) ? 10 : 0;

            string cname = Canon(go.name);
            int score = 0;

            if (cname.Contains(canonTarget))
                score = canonTarget.Length * 10;
            else
                score = words.Sum(w => cname.Contains(w) ? w.Length : 0);

            if (score + layerBonus > bestScore)
            {
                bestScore = score + layerBonus;
                best = go;
            }
        }

        // 2) As a secondary pass, check ObjectDescriptors (if any)
        if (best == null)
        {
            var descriptors = FindObjectsByType<ObjectDescriptor>(FindObjectsSortMode.None);
            foreach (var od in descriptors)
            {
                var go = od.gameObject; if (!go.activeInHierarchy) continue;
                string cname = Canon((od.objectName ?? "") + " " + od.name);
                int score = 0;

                if (cname.Contains(canonTarget))
                    score = canonTarget.Length * 10;
                else
                    score = words.Sum(w => cname.Contains(w) ? w.Length : 0);

                if (score > bestScore)
                {
                    bestScore = score;
                    best = go;
                }
            }
        }

        if (navDebugLogs && best != null)
            Debug.Log($"[Nav] Fallback matched scene object: {best.name} (score={bestScore})");

        return best;
    }

    private void TryHandleNavigation(string userMessage)
    {
        string msg = userMessage.ToLowerInvariant();
        if (!msg.Contains("guide me to") && !msg.Contains("take me to") && !msg.Contains("lead me to")) return;

        var allDetections = DedupDetections(GetAllDetections()).ToList();
        if (allDetections.Count == 0) { if (navDebugLogs) Debug.LogWarning("[Nav] No live detections yet."); return; }

        string phrase = ExtractTargetPhrase(msg);
        string targetColor = ignoreColors ? null : ExtractColor(msg);
        if (navDebugLogs) Debug.Log($"[Nav] Phrase='{phrase ?? "(null)"}', Color='{targetColor ?? "(none)"}'");

        string targetLabel = IsPronounTarget(phrase) ? lastGuidedLabel : null;
        if (string.IsNullOrEmpty(targetLabel))
        {
            var visible = allDetections.Select(d => d.label).Distinct(StringComparer.InvariantCultureIgnoreCase);
            targetLabel = ResolveTargetLabelFromMessage(string.IsNullOrEmpty(phrase) ? msg : phrase, visible);
        }
        if (string.IsNullOrEmpty(targetLabel))
        {
            var available = string.Join(", ", allDetections.Select(d => d.label).Distinct().OrderBy(s => s));
            Debug.LogWarning($"[Nav] Could not parse a known object label.\nMessage: \"{userMessage}\"\nVisible: {available}");
            return;
        }

        var filtered = allDetections.Where(d => d.label.Equals(targetLabel, StringComparison.InvariantCultureIgnoreCase));
        if (!ignoreColors && !string.IsNullOrEmpty(targetColor))
            filtered = filtered.Where(d => d.colour.Equals(targetColor, StringComparison.InvariantCultureIgnoreCase));

        var candidates = filtered.ToList();

        float r2 = 1.5f * 1.5f;
        var merged = new List<DetectionInfo>();
        foreach (var d in candidates.OrderByDescending(c => c.confidence))
            if (!merged.Any(m => m.label.Equals(d.label, StringComparison.InvariantCultureIgnoreCase) &&
                                 (m.worldPos - d.worldPos).sqrMagnitude <= r2))
                merged.Add(d);
        candidates = merged;

        if (candidates.Count == 0) { Debug.LogWarning($"[Nav] No detections for '{targetLabel}'."); return; }
        if (candidates.Count > 1 && !HasDisambiguationClues(userMessage))
        { if (navDebugLogs) Debug.Log($"[Nav] Ambiguous '{targetLabel}' with {candidates.Count} matches; waiting for qualifier."); return; }

        var surfaceMatched = candidates.Where(d => MatchesSurface(d, targetLabel)).ToList();
        if (surfaceMatched.Count > 0) candidates = surfaceMatched;

        var withHit = candidates.Where(HasValidHit).ToList();
        Vector3 navTarget;
        Vector3 from = botNavigator ? botNavigator.transform.position :
                        (Camera.main ? Camera.main.transform.position : Vector3.zero);

        if (withHit.Count > 0)
        {
            // Use best valid hit
            DetectionInfo best = default; float bestScore = float.MaxValue;
            foreach (var det in withHit)
            {
                TryGroundBelow(det.worldPos, out var objGround);
                float travel = Vector3.Distance(new Vector3(from.x, 0f, from.z), new Vector3(objGround.x, 0f, objGround.z));
                float score = travel * (TryGetDescriptorNear(det.worldPos, out _, out _) ? 0.75f : 1f);
                if (score < bestScore) { bestScore = score; best = det; }
            }

            if (TryGetDescriptorNear(best.worldPos, out _, out var go) && go != null)
                navTarget = ComputeFrontApproachPoint(go, from, approachOffset);
            else
            {
                TryGroundBelow(best.worldPos, out var objGround2);
                var dir = objGround2 - from; dir.y = 0f; if (dir.sqrMagnitude > 0.001f) dir.Normalize(); else dir = Vector3.forward;
                navTarget = FindReachableAround(objGround2 - dir * Mathf.Max(0f, approachOffset), 0.3f, 5f, 16);
            }
        }
        else
        {
            // NEW: No valid hits – snap to scene object by name and approach front
            var go = FindBestTargetGO(targetLabel);
            if (go == null) { Debug.LogWarning($"[Nav] No valid hits and no scene object matched for '{targetLabel}'."); return; }
            navTarget = ComputeFrontApproachPoint(go, from, approachOffset);
        }

        if (navDebugLogs)
        {
            var sample = string.Join(", ", candidates.Take(5)
                .Select(d => $"{d.label}@{d.worldPos:F1} surf={d.surface} dist={d.distance:0.00}"));
            Debug.Log($"[Nav] target='{targetLabel}', chosenNav={navTarget:F2}, candidates={candidates.Count}, top=[{sample}]");
        }

        if (pathDrawer != null && pathDrawer.startPoint == null && botNavigator != null)
            pathDrawer.startPoint = botNavigator.transform;

        botNavigator?.MoveToTarget(navTarget);
        pathDrawer?.DrawPathTo(navTarget);
        lastGuidedLabel = targetLabel;

        Debug.Log($"[Nav] Navigating to {targetLabel} -> {navTarget:F2}");
    }

    public bool HasContextReady() => !string.IsNullOrEmpty(visionContext) && visionContext != "nothing";
}