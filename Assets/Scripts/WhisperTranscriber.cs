using System.Collections;
using System.IO;
using UnityEngine;
using UnityEngine.Networking;

public class WhisperTranscriber : MonoBehaviour
{
    private string openAI_APIKey;

    private void Awake()
    {
        openAI_APIKey = APIKeyLoader.LoadAPIKey();
    }

    public IEnumerator TranscribeAudio(string filePath, System.Action<string> callback)
    {
        byte[] audioData = File.ReadAllBytes(filePath);

        WWWForm form = new WWWForm();
        form.AddBinaryData("file", audioData, "audio.wav", "audio/wav");
        form.AddField("model", "whisper-1");

        UnityWebRequest www = UnityWebRequest.Post("https://api.openai.com/v1/audio/transcriptions", form);
        www.SetRequestHeader("Authorization", $"Bearer {openAI_APIKey}");

        yield return www.SendWebRequest();

        if (www.result != UnityWebRequest.Result.Success)
        {
            Debug.LogError("Whisper STT Error: " + www.error);
            callback("Sorry, I couldn't hear you.");
        }
        else
        {
            var json = www.downloadHandler.text;
            var result = JsonUtility.FromJson<TranscriptionResult>(json);
            callback(result.text);
        }
    }

    [System.Serializable]
    public class TranscriptionResult
    {
        public string text;
    }
}
