using System.Collections;
using System.IO;
using UnityEngine;
using UnityEngine.Networking;
using Debug = UnityEngine.Debug;

public class WhisperTranscriber : MonoBehaviour
{
    private string openAI_APIKey;

    [Header("Model")]
    [Tooltip("Use the simple, stable STT model you’ve been using.")]
    [SerializeField] private string transcriptionModel = "whisper-1";

    private const string Endpoint = "https://api.openai.com/v1/audio/transcriptions";

    private void Awake()
    {
        openAI_APIKey = APIKeyLoader.LoadAPIKey();
        if (string.IsNullOrEmpty(openAI_APIKey))
            Debug.LogError("OpenAI API key is missing. Aborting Whisper request.");
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

        WWWForm form = new WWWForm();
        form.AddBinaryData("file", audioData, "audio.wav", "audio/wav");
        form.AddField("model", string.IsNullOrWhiteSpace(transcriptionModel) ? "whisper-1" : transcriptionModel);

        using UnityWebRequest www = UnityWebRequest.Post(Endpoint, form);
        www.SetRequestHeader("Authorization", $"Bearer {openAI_APIKey}");

        yield return www.SendWebRequest();

        if (www.result != UnityWebRequest.Result.Success)
        {
            string body = www.downloadHandler != null ? www.downloadHandler.text : "(no body)";
            Debug.LogError($"Whisper STT Error ({www.responseCode}): {www.error}\n{body}");
            callback?.Invoke("Sorry, I couldn't hear you.");
            yield break;
        }

        var json = www.downloadHandler.text;
        var result = JsonUtility.FromJson<TranscriptionResult>(json);
        callback?.Invoke(result != null ? result.text : string.Empty);
    }

    [System.Serializable]
    public class TranscriptionResult
    {
        public string text;
    }
}