using System.Collections;
using System.IO;
using UnityEngine;
using UnityEngine.Networking;
using Newtonsoft.Json;

[RequireComponent(typeof(AudioSource))]
public class OpenAITTS : MonoBehaviour
{
    private string openAI_APIKey;
    public AudioSource audioSource;

    private void Awake()
    {
        openAI_APIKey = APIKeyLoader.LoadAPIKey();
    }

    [System.Serializable]
    public class TTSRequest
    {
        public string model = "tts-1";
        public string input;
        public string voice = "nova";
    }

    public IEnumerator SpeakText(string text)
    {
        TTSRequest requestData = new TTSRequest { input = text };
        string json = JsonConvert.SerializeObject(requestData);

        UnityWebRequest request = new UnityWebRequest("https://api.openai.com/v1/audio/speech", "POST");
        byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(json);
        request.uploadHandler = new UploadHandlerRaw(bodyRaw);
        request.downloadHandler = new DownloadHandlerBuffer();

        request.SetRequestHeader("Content-Type", "application/json");
        request.SetRequestHeader("Authorization", $"Bearer {openAI_APIKey}");

        yield return request.SendWebRequest();

        if (request.result != UnityWebRequest.Result.Success)
        {
            Debug.LogError("TTS Error: " + request.error);
            yield break;
        }

        string filePath = Path.Combine(Application.persistentDataPath, "response.mp3");
        File.WriteAllBytes(filePath, request.downloadHandler.data);

        StartCoroutine(PlayAudio(filePath));
    }

    private IEnumerator PlayAudio(string path)
    {
        using UnityWebRequest www = UnityWebRequestMultimedia.GetAudioClip("file://" + path, AudioType.MPEG);
        yield return www.SendWebRequest();

        if (www.result != UnityWebRequest.Result.Success)
        {
            Debug.LogError("Audio playback error: " + www.error);
            yield break;
        }

        AudioClip clip = DownloadHandlerAudioClip.GetContent(www);
        audioSource.clip = clip;
        audioSource.Play();
    }
}
