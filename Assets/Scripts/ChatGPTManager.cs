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
    [SerializeField] private List<YoloObjectDetector> detectors;
    [SerializeField] private BotNavigator botNavigator;
    [SerializeField] private PathDrawer pathDrawer;

    private string visionContext = "nothing";
    private string openAI_APIKey;

    private readonly string openAI_Endpoint = "https://api.openai.com/v1/chat/completions";

    private void Awake()
    {
        openAI_APIKey = APIKeyLoader.LoadAPIKey();
    }

    private void OnEnable()
    {
        foreach (var detector in detectors)
        {
            if (detector != null)
                detector.OnDetections += HandleDetections;
        }
    }

    private void OnDisable()
    {
        foreach (var detector in detectors)
        {
            if (detector != null)
                detector.OnDetections -= HandleDetections;
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

    public IEnumerator SendMessageToOpenAI(string userMessage, System.Action<string> callback)
    {
        var requestData = new
        {
            model = "gpt-3.5-turbo",
            messages = new[] {
                new {
                    role = "system",
                    content =
                        "You are a helpful VR assistant. " +
                        "You receive live computer-vision detections. " +
                        $"Right now you see: {visionContext}. " +
                        "Use this when answering questions about " +
                        "what is in front of you or its properties."
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
            string aiResponse = response.choices[0].message.content.Trim();

            UnityEngine.Debug.Log("GPT says: " + aiResponse);
            TryHandleNavigation(userMessage);

            callback(aiResponse);
        }
    }

    private void TryHandleNavigation(string userMessage)
    {
        string msg = userMessage.ToLower();

        if (!msg.Contains("guide me to") && !msg.Contains("take me to"))
            return;

        UnityEngine.Debug.Log("[TryHandleNavigation] Attempting to guide to red car");

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
                foreach (var d in detectors)
                {
                    if (d.IsCocoLabel(word))
                    {
                        targetLabel = word.ToLower();
                    }
                }
            }
        }

        if (string.IsNullOrEmpty(targetLabel))
        {
            UnityEngine.Debug.LogWarning("Navigation: No known object label in message.");
            return;
        }

        // Aggregate detections from all detectors
        var allDetections = new List<DetectionInfo>();
        foreach (var d in detectors)
        {
            allDetections.AddRange(d.GetLatestDetections());
        }

        // Optional: log all detected objects
        foreach (var det in allDetections)
        {
            UnityEngine.Debug.Log($"Detected: {det.label}, Color: {det.colour}, Pos: {det.worldPos}, Distance: {det.distance}");
        }

        // Filter based on label and optional color
        var filtered = allDetections
            .Where(d => d.label.ToLower() == targetLabel);

        if (!string.IsNullOrEmpty(targetColor))
        {
            filtered = filtered.Where(d => d.colour.ToLower() == targetColor);
        }

        var bestMatch = filtered
            .OrderBy(d => d.distance)
            .FirstOrDefault();

        if (!string.IsNullOrEmpty(bestMatch.label))
        {
            botNavigator?.MoveToTarget(bestMatch.worldPos);
            pathDrawer?.DrawPathTo(bestMatch.worldPos);
            UnityEngine.Debug.Log($"Navigating to {targetColor} {targetLabel} at {bestMatch.worldPos}");
        }
        else
        {
            UnityEngine.Debug.LogWarning($"No matching {targetColor} {targetLabel} found.");
        }
    }

    public bool HasContextReady()
    {
        return !string.IsNullOrEmpty(visionContext) && visionContext != "nothing";
    }

    private bool IsColor(string word)
    {
        string[] colors = { "red", "gray", "blue", "green", "yellow", "white", "black", "orange", "purple", "pink" };
        return colors.Contains(word.ToLower());
    }
}
