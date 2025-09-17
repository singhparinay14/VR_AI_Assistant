using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.Networking;
using Newtonsoft.Json;

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
    [SerializeField] private List<PropsDetector> propDetectors = new(); // props (COCO-80)
    [SerializeField] private BotNavigator botNavigator;
    [SerializeField] private PathDrawer pathDrawer;

    [SerializeField] private bool autoFindPropDetectors = true; // auto-wire fallback

    private string visionContext = "nothing";
    private string openAI_APIKey;

    private readonly string openAI_Endpoint = "https://api.openai.com/v1/chat/completions";

    private void Awake()
    {
        openAI_APIKey = APIKeyLoader.LoadAPIKey();
    }

    private void OnEnable()
    {
        // Auto-find props if not manually assigned
        if (autoFindPropDetectors && (propDetectors == null || propDetectors.Count == 0))
            propDetectors = FindObjectsByType<PropsDetector>(FindObjectsSortMode.None).ToList();

        if (detectors != null)
        {
            foreach (var detector in detectors)
                if (detector != null) detector.OnDetections += HandleDetections;
        }

        if (propDetectors != null)
        {
            foreach (var p in propDetectors)
                if (p != null) p.OnDetections += HandleDetections;
        }
    }

    private void OnDisable()
    {
        if (detectors != null)
        {
            foreach (var detector in detectors)
                if (detector != null) detector.OnDetections -= HandleDetections;
        }

        if (propDetectors != null)
        {
            foreach (var p in propDetectors)
                if (p != null) p.OnDetections -= HandleDetections;
        }
    }

    private void HandleDetections(List<DetectionInfo> list)
    {
        if (list == null || list.Count == 0)
        {
            visionContext = "nothing";
            return;
        }

        var grouped = list
            .GroupBy(d => $"{d.colour} {d.label}")
            .Select(g => $"{g.Count()} {g.Key}");

        visionContext = string.Join(", ", grouped);
    }

    // ---------- NEW: richer scene facts from ObjectDescriptor ----------
    private IEnumerable<DetectionInfo> GetAllDetections()
    {
        var all = new List<DetectionInfo>();
        if (detectors != null)
            foreach (var d in detectors)
                if (d != null) all.AddRange(d.GetLatestDetections());

        if (propDetectors != null)
            foreach (var p in propDetectors)
                if (p != null) all.AddRange(p.GetLatestDetections());

        return all;
    }

    private bool TryGetDescriptorNear(Vector3 pos, out ObjectDescriptor od, out GameObject go)
    {
        // Small radius to catch the collider with the descriptor
        const float radius = 0.8f;
        var cols = Physics.OverlapSphere(pos, radius);
        foreach (var c in cols)
        {
            var d = c.GetComponentInParent<ObjectDescriptor>();
            if (d != null)
            {
                od = d; go = d.gameObject;
                return true;
            }
        }
        od = null; go = null;
        return false;
    }

    private string BuildSceneFacts(int maxItems = 8)
    {
        var dets = GetAllDetections()
            .OrderBy(d => d.distance < 0 ? float.MaxValue : d.distance) // nearest first
            .Take(40) // cap work
            .ToList();

        if (dets.Count == 0) return "";

        var lines = new List<string>();
        int added = 0;

        foreach (var d in dets)
        {
            if (added >= maxItems) break;

            if (TryGetDescriptorNear(d.worldPos, out var od, out var go))
            {
                // Use descriptor details when available
                var title = string.IsNullOrWhiteSpace(od.objectName) ? d.label : od.objectName;
                var artist = string.IsNullOrWhiteSpace(od.artistName) ? "" : $" by {od.artistName}";
                var desc = string.IsNullOrWhiteSpace(od.description) ? "" : $". {od.description}";
                lines.Add($"{d.colour} {d.label}: \"{title}\"{artist}{desc}");
            }
            else
            {
                // Fallback to detection info only
                var distTxt = d.distance > 0 ? $" (~{d.distance:0.0}m)" : "";
                lines.Add($"{d.colour} {d.label}{distTxt}");
            }
            added++;
        }

        return string.Join(" | ", lines);
    }
    // -------------------------------------------------------------------

    public IEnumerator SendMessageToOpenAI(string userMessage, System.Action<string> callback)
    {
        // Build richer, structured context
        string sceneFacts = BuildSceneFacts(8);

        var requestData = new
        {
            model = "gpt-3.5-turbo",
            messages = new[] {
                new {
                    role = "system",
                    content =
                        "You are a helpful VR assistant inside a virtual gallery. " +
                        "Use the live computer-vision context and the scene facts below to answer precisely. " +
                        $"Detections summary: {visionContext}. " +
                        (string.IsNullOrEmpty(sceneFacts) ? "" : $"Scene facts: {sceneFacts}. ") +
                        "If the user asks about an object, describe it succinctly (title/artist/color/nearby info) before taking action."
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
            UnityEngine.Debug.LogError("OpenAI error: " + request.error);
            callback("Sorry, something went wrong.");
        }
        else
        {
            string responseJson = request.downloadHandler.text;
            var response = JsonConvert.DeserializeObject<OpenAIResponse>(responseJson);
            string reply = response.choices[0].message.content.Trim();

            UnityEngine.Debug.Log("GPT says: " + reply);

            // Reuse your guidance behaviour (works for cars and props)
            TryHandleNavigation(userMessage);

            callback(reply);
        }
    }

    private void TryHandleNavigation(string userMessage)
    {
        string msg = userMessage.ToLower();

        // trigger words like before
        if (!msg.Contains("guide me to") && !msg.Contains("take me to") && !msg.Contains("lead me to"))
            return;

        UnityEngine.Debug.Log("[TryHandleNavigation] Attempting to guide to target object");

        string[] words = msg.Split(' ');
        string targetLabel = "";
        string targetColor = "";

        foreach (string word in words)
        {
            if (IsColor(word))
            {
                targetColor = word.ToLower();
            }
            else
            {
                // cars (fine-tuned)
                if (detectors != null)
                    foreach (var d in detectors)
                        if (d != null && d.HasLabel(word))
                            targetLabel = word.ToLower();

                // props (COCO-80)
                if (propDetectors != null)
                    foreach (var p in propDetectors)
                        if (p != null && p.HasLabel(word))
                            targetLabel = word.ToLower();
            }
        }

        if (string.IsNullOrEmpty(targetLabel))
        {
            UnityEngine.Debug.LogWarning("Navigation: No known object label in message.");
            return;
        }

        // Gather all detections (cars + props)
        var allDetections = GetAllDetections().ToList();

        // Filter by label and optional color
        var filtered = allDetections.Where(d => d.label.ToLower() == targetLabel);
        if (!string.IsNullOrEmpty(targetColor))
            filtered = filtered.Where(d => d.colour.ToLower() == targetColor);

        // If descriptions exist, prefer the one with an ObjectDescriptor
        DetectionInfo best = default;
        float bestScore = float.MaxValue;

        foreach (var det in filtered)
        {
            float score = det.distance > 0 ? det.distance : 9999f;
            if (TryGetDescriptorNear(det.worldPos, out _, out _))
                score *= 0.75f; // prefer described pieces slightly

            if (score < bestScore)
            {
                bestScore = score;
                best = det;
            }
        }

        if (!string.IsNullOrEmpty(best.label))
        {
            botNavigator?.MoveToTarget(best.worldPos);
            pathDrawer?.DrawPathTo(best.worldPos);
            UnityEngine.Debug.Log($"Navigating to {(string.IsNullOrEmpty(targetColor) ? "" : targetColor + " ")}{targetLabel} at {best.worldPos}");
        }
        else
        {
            UnityEngine.Debug.LogWarning($"No matching {(string.IsNullOrEmpty(targetColor) ? "" : targetColor + " ")}{targetLabel} found.");
        }
    }

    public bool HasContextReady() =>
        !string.IsNullOrEmpty(visionContext) && visionContext != "nothing";

    private bool IsColor(string word)
    {
        string[] colors = { "red", "gray", "blue", "green", "yellow", "white", "black", "orange", "purple", "pink", "brown" };
        return colors.Contains(word.ToLower());
    }
}
