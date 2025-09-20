using System.Collections;
using System.IO;
using UnityEngine;
using UnityEngine.Networking;
using Debug = UnityEngine.Debug;

public class WhisperTranscriber : MonoBehaviour
{
    private string openAI_APIKey;

    [Header("OpenAI Project Scoping (optional)")]
    [Tooltip("Only set if you must scope the request to a specific Project. Otherwise leave blank and turn off 'Include Project Header'.")]
    [SerializeField] private string openAI_ProjectId = ""; // e.g., "proj_abc123"
    [Tooltip("Include the 'OpenAI-Project' header. Turn OFF if your key is already project-scoped or you don't need it.")]
    [SerializeField] private bool includeProjectHeader = false;

    [Header("Model")]
    [Tooltip("Preferred STT model. If access fails, optionally falls back to 'whisper-1'.")]
    [SerializeField] private string transcriptionModel = "gpt-4o-mini-transcribe";
    [Tooltip("If the chosen model isn't accessible, retry with 'whisper-1' (and without project header).")]
    [SerializeField] private bool fallbackToWhisper1 = true;

    private const string Endpoint = "https://api.openai.com/v1/audio/transcriptions";

    private void Awake()
    {
        openAI_APIKey = APIKeyLoader.LoadAPIKey();
        if (string.IsNullOrEmpty(openAI_APIKey))
        {
            Debug.LogError("OpenAI API key is missing. Aborting Whisper request.");
        }

        if (includeProjectHeader && string.IsNullOrEmpty(openAI_ProjectId))
        {
            Debug.LogWarning("WhisperTranscriber: 'Include Project Header' is enabled but Project ID is empty. The header will not be added.");
        }
    }

    public IEnumerator TranscribeAudio(string filePath, System.Action<string> callback)
    {
        if (string.IsNullOrEmpty(openAI_APIKey))
        {
            callback?.Invoke("API key missing.");
            yield break;
        }

        if (string.IsNullOrEmpty(filePath) || !File.Exists(filePath))
        {
            Debug.LogError("WhisperTranscriber: Audio file path is invalid or does not exist.");
            callback?.Invoke("Sorry, I couldn't hear you.");
            yield break;
        }

        byte[] audioData = File.ReadAllBytes(filePath);

        // Try preferred model first
        yield return StartCoroutine(TranscribeWithModel(audioData, transcriptionModel, includeProjectHeader, (success, text) =>
        {
            if (success)
            {
                callback?.Invoke(text);
            }
            else
            {
                // If access fails and fallback is allowed, retry with whisper-1 and no project header
                if (fallbackToWhisper1 && !string.Equals(transcriptionModel, "whisper-1"))
                {
                    Debug.LogWarning("Retrying STT with 'whisper-1' and without 'OpenAI-Project' header...");
                    StartCoroutine(TranscribeWithModel(audioData, "whisper-1", false, (ok, txt) =>
                    {
                        callback?.Invoke(ok ? txt : "Sorry, I couldn't hear you.");
                    }));
                }
                else
                {
                    callback?.Invoke("Sorry, I couldn't hear you.");
                }
            }
        }));
    }

    private IEnumerator TranscribeWithModel(byte[] audioData, string model, bool addProjectHeader, System.Action<bool, string> done)
    {
        WWWForm form = new WWWForm();
        form.AddBinaryData("file", audioData, "audio.wav", "audio/wav");
        form.AddField("model", model);

        UnityWebRequest www = UnityWebRequest.Post(Endpoint, form);
        www.SetRequestHeader("Authorization", $"Bearer {openAI_APIKey}");

        if (addProjectHeader && !string.IsNullOrEmpty(openAI_ProjectId))
            www.SetRequestHeader("OpenAI-Project", openAI_ProjectId);

        yield return www.SendWebRequest();

        if (www.result != UnityWebRequest.Result.Success)
        {
            string body = www.downloadHandler != null ? www.downloadHandler.text : "(no body)";
            Debug.LogError($"Whisper STT Error ({www.responseCode}): {www.error}\n{body}");

            // Detect model access/model-not-found conditions to trigger fallback
            bool projectNoAccess = www.responseCode == 403 && body.Contains("does not have access to model");
            bool modelNotFound = (www.responseCode == 404 || www.responseCode == 400) && body.Contains("model_not_found");

            if (projectNoAccess || modelNotFound)
            {
                done?.Invoke(false, null);
                yield break;
            }

            done?.Invoke(false, null);
            yield break;
        }

        var json = www.downloadHandler.text;
        var result = JsonUtility.FromJson<TranscriptionResult>(json);
        done?.Invoke(true, result != null ? result.text : string.Empty);
    }

    [System.Serializable]
    public class TranscriptionResult
    {
        public string text;
    }
}