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
public class Message
{
    public string role;
    public string content;
}

[System.Serializable]
public class Choice
{
    public Message message;
}

[System.Serializable]
public class OpenAIResponse
{
    public List<Choice> choices;
}

public class ChatGPTManager : MonoBehaviour
{
    [Header("Vision Link")]
    [SerializeField] private List<YoloObjectDetector> detectors = new();   // cars (fine-tuned)
    [SerializeField] private List<PropsDetector> propDetectors = new();  // props (COCO-80)

    [Header("Guide Bot")]
    [SerializeField] private BotNavigator botNavigator;
    [SerializeField] private PathDrawer pathDrawer;
    [SerializeField] private float navSampleRadius = 3f;   // search radius for nearest NavMesh
    [SerializeField] private float approachOffset = 1.0f; // stand-off distance from the object
    [SerializeField] private string lastGuidedLabel = null; // for "guide me to it"
    [SerializeField] private bool navDebugLogs = true;

    private static string Normalize(string s)
    {
        if (string.IsNullOrEmpty(s)) return string.Empty;
        return System.Text.RegularExpressions.Regex.Replace(s.ToLowerInvariant(), @"[^a-z0-9]+", "");
    }

    private static bool ContainsQualifier(string p)
    {
        if (string.IsNullOrEmpty(p)) return false;
        // recognize distance/side/“closest” words and simple colors
        return System.Text.RegularExpressions.Regex.IsMatch(
            p, @"\b(closest|nearest|farther|farthest|furthest|leftmost|rightmost|left|right|center|middle|red|blue|green|gray|grey|black|white|yellow|orange|purple|cyan)\b",
            System.Text.RegularExpressions.RegexOptions.IgnoreCase);
    }

    // === De-dup config ===
    [Header("Vision De-dup")]
    [SerializeField] private float gridSizeMeters = 1.5f; // quantization size when no descriptor
    [SerializeField] private bool preferObjectDescriptorId = true; // use ObjectDescriptor if present

    // Build a stable key for the same physical object across cameras/frames
    private string MakeStableKey(DetectionInfo d)
    {
        // Prefer a nearby ObjectDescriptor (stable across frames)
        if (preferObjectDescriptorId && TryGetDescriptorNear(d.worldPos, out var od, out var go) && go != null)
            return $"{d.label}|obj:{go.GetInstanceID()}";

        // Fallback: coarse world grid (x/z only)
        float g = Mathf.Max(0.25f, gridSizeMeters);
        int gx = Mathf.FloorToInt(d.worldPos.x / g);
        int gz = Mathf.FloorToInt(d.worldPos.z / g);
        return $"{d.label}|gx:{gx}|gz:{gz}";
    }

    // Collapse many raw detections to one per physical object (best confidence wins)
    private List<DetectionInfo> DedupDetections(IEnumerable<DetectionInfo> dets)
    {
        var bestByKey = new Dictionary<string, DetectionInfo>();
        foreach (var d in dets)
        {
            var key = MakeStableKey(d);
            if (!bestByKey.TryGetValue(key, out var cur) || d.confidence > cur.confidence)
                bestByKey[key] = d;
        }
        return bestByKey.Values.ToList();
    }



    //private readonly Dictionary<string, (DetectionInfo det, float lastSeen)> _ttl = new();
    [SerializeField] private float ttlSeconds = 1.5f;

    [Header("Autowire")]
    [SerializeField] private bool autoFindPropDetectors = true;

    private string visionContext = "nothing";
    private string openAI_APIKey;
    private readonly string openAI_Endpoint = "https://api.openai.com/v1/chat/completions";

    // ----- parsing helpers -----
    private static readonly Regex TriggerRegex = new(
        @"(?:\b(?:guide|take|lead|walk|go)\s+me\s+to\s+(?<target>.+)$)|(?:\b(?:go|walk)\s+to\s+(?<target>.+)$)|(?:\bnearest\s+(?<target>.+)$)",
        RegexOptions.IgnoreCase | RegexOptions.Compiled);

    [Serializable]
    private struct Seen
    {
        public DetectionInfo det;
        public float lastSeen;
    }

    // TTL cache: keeps most recent sightings for a short time window
    private readonly Dictionary<string, Seen> _ttl = new Dictionary<string, Seen>();

    // If this field already exists in your file, keep your value and do NOT duplicate it.
    // [SerializeField] private float ttlSeconds = 1.5f;  // <-- you already have this


    private static readonly string[] ColorWords =
    {
        "red","gray","grey","blue","green","yellow","white","black",
        "orange","purple","pink","brown","cyan"
    };

    private static readonly Dictionary<string, string[]> LabelSynonyms = new()
    {
        { "bicycle",      new[] { "bike", "cycle", "bicycles", "bikes" } },
        { "couch",        new[] { "sofa", "settee" } },
        { "tv",           new[] { "television", "screen", "monitor" } },
        { "dining table", new[] { "table", "desk" } },
        { "cell phone",   new[] { "phone", "mobile" } },
        { "microwave",    new[] { "oven" } },
        { "bottle",       new[] { "bottles" } },
        { "bench",        new[] { "benches" } },
        { "chair",        new[] { "chairs", "seat" } },
        { "clock",        new[] { "wall clock" } }
    };
    // ---------------------------

    // --- Navigation trigger policy ---
    private enum NavTriggerMode { UserOnly, ModelAllowed, Disabled }

    // Default: only the user's text can trigger movement
    [SerializeField] private NavTriggerMode navTrigger = NavTriggerMode.UserOnly;

    // Simple check for nav verbs ("guide/take/lead me to ...")
    private bool HasNavigationIntent(string text)
    {
        if (string.IsNullOrEmpty(text)) return false;
        text = text.ToLowerInvariant();
        return text.Contains("guide me to") || text.Contains("take me to") || text.Contains("lead me to");
    }

    // Do we have any qualifier that helps disambiguate many same-type objects?
    // (color, nearest/closest/farthest, side, or a number that could be a distance)
    private bool HasDisambiguationClues(string text)
    {
        if (string.IsNullOrEmpty(text)) return false;

        var t = text.ToLowerInvariant();

        // color?
        if (!string.IsNullOrEmpty(ExtractColor(t))) return true;

        // nearest / closest / farthest
        if (Regex.IsMatch(t, @"\b(nearest|closest|farthest|furthest)\b")) return true;

        // direction / side
        if (Regex.IsMatch(t, @"\b(left|right|center|middle|front|back|behind|ahead|near|far)\b")) return true;

        // a number (e.g., "20.6" meters) anywhere in phrase
        if (Regex.IsMatch(t, @"\b\d+(\.\d+)?\b")) return true;

        // quick ordinal check: first/second/third
        if (Regex.IsMatch(t, @"\b(first|second|third)\b")) return true;

        return false;
    }


    private void Awake()
    {
        openAI_APIKey = APIKeyLoader.LoadAPIKey();
    }

    private void OnEnable()
    {
        if (autoFindPropDetectors && (propDetectors == null || propDetectors.Count == 0))
            propDetectors = FindObjectsByType<PropsDetector>(FindObjectsSortMode.None).ToList();

        if (detectors != null)
            foreach (var d in detectors) if (d != null) d.OnDetections += HandleDetections;

        if (propDetectors != null)
            foreach (var p in propDetectors) if (p != null) p.OnDetections += HandleDetections;
    }

    private void OnDisable()
    {
        if (detectors != null)
            foreach (var d in detectors) if (d != null) d.OnDetections -= HandleDetections;

        if (propDetectors != null)
            foreach (var p in propDetectors) if (p != null) p.OnDetections -= HandleDetections;
    }

    private void HandleDetections(List<DetectionInfo> list)
    {
        float now = Time.time;

        // 1) Update TTL with all detections we just received
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

        // 2) Purge any entries older than ttlSeconds
        var toRemove = new List<string>();
        foreach (var kv in _ttl)
            if (now - kv.Value.lastSeen > ttlSeconds)
                toRemove.Add(kv.Key);
        for (int i = 0; i < toRemove.Count; i++)
            _ttl.Remove(toRemove[i]);

        // 3) Build stable vision context from the remaining TTL items
        if (_ttl.Count == 0)
        {
            visionContext = "nothing";
            return;
        }

        var dedupList = DedupDetections(_ttl.Values.Select(v => v.det));
        var groupedParts = dedupList
            .GroupBy(d => $"{d.colour} {d.label}")
            .OrderByDescending(g => g.Count())
            .Select(g => $"{g.Count()} {g.Key}");

        visionContext = string.Join(", ", groupedParts);
    }




    // ======== Scene facts (ObjectDescriptor-aware) ========

    private IEnumerable<DetectionInfo> GetAllDetections()
    {
        var all = new List<DetectionInfo>();

        if (detectors != null)
            foreach (var d in detectors) if (d != null) all.AddRange(d.GetLatestDetections());

        if (propDetectors != null)
            foreach (var p in propDetectors) if (p != null) all.AddRange(p.GetLatestDetections());

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

    private string BuildSceneFacts(int maxItems = 8)
    {
        var dets = DedupDetections(GetAllDetections())
            .Where(d => d.distance > 0)
            .GroupBy(d => d.label)
            .SelectMany(g => g.OrderBy(d => d.distance).Take(3))
            .OrderBy(d => d.distance)
            .Take(maxItems)
            .ToList();

        if (dets.Count == 0) return "";

        var lines = new List<string>();
        foreach (var d in dets)
        {
            string dist = $"{Mathf.Round(d.distance * 10f) / 10f:0.0}m";
            if (TryGetDescriptorNear(d.worldPos, out var od, out _))
            {
                var title = string.IsNullOrWhiteSpace(od.objectName) ? d.label : od.objectName;
                var artist = string.IsNullOrWhiteSpace(od.artistName) ? "" : $" by {od.artistName}";
                lines.Add($"{d.colour} {d.label}: \"{title}\"{artist} (~{dist})");
            }
            else
            {
                lines.Add($"{d.colour} {d.label} (~{dist})");
            }
        }
        return string.Join(" | ", lines);
    }


    // ======================================================

    public IEnumerator SendMessageToOpenAI(string userMessage, Action<string> callback)
    {
        string sceneFacts = BuildSceneFacts(8);

        var requestData = new
        {
            model = "gpt-3.5-turbo",
            messages = new[] {
                new {
                    role = "system",
                    content =
                        "You are a helpful VR assistant inside a virtual gallery. " +
                        "Use the live computer-vision context and the scene facts below to communicate precisely. " +
                        $"Detections summary: {visionContext}. " +
                        (string.IsNullOrEmpty(sceneFacts) ? "" : $"Scene facts: {sceneFacts}. ") +
                        "If the user asks about an object, describe it briefly with the information you have about the said object before taking action."
                },
                new { role = "user", content = userMessage }
            },
            temperature = 0.7f
        };

        string jsonData = JsonConvert.SerializeObject(requestData);
        byte[] jsonToSend = System.Text.Encoding.UTF8.GetBytes(jsonData);

        using UnityWebRequest request = new UnityWebRequest(openAI_Endpoint, "POST");
        request.uploadHandler = new UploadHandlerRaw(jsonToSend);
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
            string responseJson = request.downloadHandler.text;
            var response = JsonConvert.DeserializeObject<OpenAIResponse>(responseJson);
            string reply = response.choices[0].message.content.Trim();

            Debug.Log("GPT says: " + reply);

            // Only trigger navigation off the *user's* message (not the model reply),
            // and only when nav is enabled by policy.
            if (navTrigger == NavTriggerMode.UserOnly && HasNavigationIntent(userMessage))
            {
                TryHandleNavigation(userMessage);
            }


            callback(reply);
        }
    }

    // ----------------- Navigation helpers -----------------

    private string ExtractTargetPhrase(string msgLower)
    {
        var m = TriggerRegex.Match(msgLower);
        if (!m.Success) return null;
        var phrase = m.Groups["target"].Value.Trim();
        // strip odd punctuation like trailing "/"
        phrase = Regex.Replace(phrase, @"[^a-zA-Z0-9\s\-]", "");
        return phrase.Trim(' ', '.', '!', '?');
    }

    private bool IsPronounTarget(string phrase) =>
        !string.IsNullOrEmpty(phrase) && Regex.IsMatch(phrase, @"^(it|that|this)$", RegexOptions.IgnoreCase);

    private string ExtractColor(string msgLower)
    {
        foreach (var c in ColorWords)
        {
            if (Regex.IsMatch(msgLower, $@"\b{Regex.Escape(c)}\b"))
                return c == "grey" ? "gray" : c;
        }
        return null;
    }

    private string ResolveTargetLabelFromMessage(string msgLower, IEnumerable<string> visibleLabels)
    {
        string bestLabel = null;
        int bestLength = -1;

        foreach (var raw in visibleLabels)
        {
            if (string.IsNullOrWhiteSpace(raw)) continue;
            string label = raw.ToLowerInvariant();

            if (WordBoundaryContains(msgLower, label) && label.Length > bestLength)
            {
                bestLabel = raw; bestLength = label.Length;
            }

            if (LabelSynonyms.TryGetValue(label, out var syns))
            {
                foreach (var s in syns)
                {
                    var syn = s.ToLowerInvariant();
                    if (WordBoundaryContains(msgLower, syn) && syn.Length > bestLength)
                    {
                        bestLabel = raw; bestLength = syn.Length;
                    }
                }
            }

            if (bestLength < 0)
            {
                foreach (var token in msgLower.Split(' '))
                {
                    var t = token.Trim();
                    if (t.Length < 3) continue;
                    if (label.Contains(t) && WordBoundaryContains(msgLower, t) && t.Length > bestLength)
                    {
                        bestLabel = raw; bestLength = t.Length;
                    }
                }
            }
        }
        return bestLabel;
    }

    private bool WordBoundaryContains(string textLower, string phraseLower)
    {
        var pattern = $@"\b{Regex.Escape(phraseLower)}\b";
        return Regex.IsMatch(textLower, pattern);
    }

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

    // Search a ring of points around the target to find a *reachable* NavMesh position.
    private Vector3 FindReachableAround(Vector3 objWorld, float inner = 0.3f, float outer = 5f, int ringSamples = 16)
    {
        TryGroundBelow(objWorld, out var basePt);

        // Try center first
        if (NavMesh.SamplePosition(basePt, out var hit0, navSampleRadius, NavMesh.AllAreas))
        {
            var path0 = new NavMeshPath();
            var from0 = botNavigator ? botNavigator.transform.position :
                      (Camera.main ? Camera.main.transform.position : Vector3.zero);

            if (NavMesh.CalculatePath(from0, hit0.position, NavMesh.AllAreas, path0) &&
                path0.status == NavMeshPathStatus.PathComplete)
                return hit0.position;
        }

        // Expand outwards
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

        // Last resort
        return ToNavmesh(basePt, Mathf.Max(navSampleRadius, outer));
    }

    // If collider we hit is named/tagged like the thing we want, prefer it.
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

    private void TryHandleNavigation(string userMessage)
    {
        string msg = userMessage.ToLowerInvariant();
        if (!msg.Contains("guide me to") && !msg.Contains("take me to") && !msg.Contains("lead me to"))
            return;

        var allDetections = DedupDetections(GetAllDetections()).ToList();
        if (allDetections.Count == 0)
        {
            if (navDebugLogs) Debug.LogWarning("[Nav] No live detections yet.");
            return;
        }

        string phrase = ExtractTargetPhrase(msg);
        string targetColor = ExtractColor(msg);
        if (navDebugLogs) Debug.Log($"[Nav] Phrase='{phrase ?? "(null)"}', Color='{targetColor ?? "(none)"}'");

        string targetLabel = null;
        if (IsPronounTarget(phrase))
        {
            targetLabel = lastGuidedLabel;
            if (navDebugLogs) Debug.Log($"[Nav] Pronoun; using last label '{targetLabel ?? "(null)"}'");
        }

        if (string.IsNullOrEmpty(targetLabel))
        {
            var visible = allDetections.Select(d => d.label)
                                       .Distinct(StringComparer.InvariantCultureIgnoreCase);
            targetLabel = ResolveTargetLabelFromMessage(string.IsNullOrEmpty(phrase) ? msg : phrase, visible);
        }

        if (string.IsNullOrEmpty(targetLabel))
        {
            var available = string.Join(", ",
                allDetections.Select(d => d.label).Distinct().OrderBy(s => s));
            Debug.LogWarning($"[Nav] Could not parse a known object label.\nMessage: \"{userMessage}\"\nVisible: {available}");
            return;
        }

        var filtered = allDetections.Where(d =>
            d.label.Equals(targetLabel, StringComparison.InvariantCultureIgnoreCase));

        if (!string.IsNullOrEmpty(targetColor))
            filtered = filtered.Where(d =>
                d.colour.Equals(targetColor, StringComparison.InvariantCultureIgnoreCase));

        var candidates = filtered.ToList();

        // Merge near-identical candidates (same label, within 1.5m)
        float r2 = 1.5f * 1.5f;
        var merged = new List<DetectionInfo>();
        foreach (var d in candidates.OrderByDescending(c => c.confidence))
        {
            if (!merged.Any(m => m.label.Equals(d.label, StringComparison.InvariantCultureIgnoreCase) &&
                                 (m.worldPos - d.worldPos).sqrMagnitude <= r2))
                merged.Add(d);
        }
        candidates = merged;

        if (candidates.Count == 0)
        {
            Debug.LogWarning($"[Nav] No detections for '{targetLabel}' color='{(targetColor ?? "any")}'.");
            return;
        }
        // If there are multiple candidates of the same label and the user didn't give
        // any qualifier, DO NOT MOVE. Let the assistant ask and wait for the next message.
        if (candidates.Count > 1 && !HasDisambiguationClues(userMessage))
        {
            if (navDebugLogs)
                Debug.Log($"[Nav] Ambiguous request for '{targetLabel}' with {candidates.Count} matches " +
                          "and no qualifiers (color/nearest/left/right/distance). Waiting for clarification.");
            return;
        }


        // Prefer surface-matching hits
        var surfaceMatched = candidates.Where(d => MatchesSurface(d, targetLabel)).ToList();
        if (surfaceMatched.Count > 0) candidates = surfaceMatched;

        // Score: prefer with descriptor, then nearest
        DetectionInfo best = default;
        float bestScore = float.MaxValue;

        foreach (var det in candidates)
        {
            float score = det.distance > 0 ? det.distance : 9999f;
            if (TryGetDescriptorNear(det.worldPos, out _, out _)) score *= 0.75f;
            if (score < bestScore) { bestScore = score; best = det; }
        }

        if (string.IsNullOrEmpty(best.label))
        {
            Debug.LogWarning("[Nav] Candidates existed but none selected.");
            return;
        }

        // Compute a standoff point in front of the object and find a reachable spot near it
        Vector3 from = botNavigator ? botNavigator.transform.position :
                        (Camera.main ? Camera.main.transform.position : Vector3.zero);

        TryGroundBelow(best.worldPos, out var objGround);
        Vector3 dir = objGround - from; dir.y = 0f;
        if (dir.sqrMagnitude > 0.001f) dir.Normalize(); else dir = Vector3.forward;

        Vector3 desired = objGround - dir * Mathf.Max(0f, approachOffset);
        Vector3 navTarget = FindReachableAround(desired, 0.3f, 5f, 16);

        if (navDebugLogs)
        {
            var sample = string.Join(", ",
                candidates.Take(5).Select(d => $"{d.label}@{d.worldPos:F1}({d.distance:0.0}m)"));
            Debug.Log($"[Nav] target='{targetLabel}', candidates={candidates.Count}, top5=[{sample}]");
        }

        botNavigator?.MoveToTarget(navTarget);
        pathDrawer?.DrawPathTo(navTarget);
        lastGuidedLabel = targetLabel;

        Debug.Log($"[Nav] Navigating to {(string.IsNullOrEmpty(targetColor) ? "" : targetColor + " ")}{targetLabel} " +
                  $"(obj:{objGround:F2} -> nav:{navTarget:F2})");
    }

    // ------------------------------------------------------

    public bool HasContextReady() =>
        !string.IsNullOrEmpty(visionContext) && visionContext != "nothing";

    // (still used by older code paths somewhere else)
    private bool IsColor(string word)
    {
        string[] colors = { "red", "gray", "blue", "green", "yellow", "white", "black", "orange", "purple", "pink", "brown" };
        return colors.Contains(word.ToLower());
    }
}
