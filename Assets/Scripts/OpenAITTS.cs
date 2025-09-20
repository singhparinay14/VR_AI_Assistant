using Newtonsoft.Json;
using System.Collections;
using System.Diagnostics;
using System.IO;
using UnityEngine;
using UnityEngine.Networking;
using static System.Net.Mime.MediaTypeNames;
using Debug = UnityEngine.Debug;
using Application = UnityEngine.Application;

[RequireComponent(typeof(AudioSource))]
public class OpenAITTS : MonoBehaviour
{
    private string openAI_APIKey;
    public AudioSource audioSource;
    private bool isSpeaking = false; // prevent overlapping calls

    [Header("Model/Voice/Format")]
    [Tooltip("Stable simple TTS model")]
    public string model = "tts-1";
    [Tooltip("Common voices: alloy, nova")]
    public string voice = "nova";
    [Tooltip("mp3 or wav. WAV + fallback decoder is most robust across platforms.")]
    public string format = "wav";

    [Header("Debug")]
    public bool logHeaders = true;
    public bool logFirstBytes = true;

    private void Awake()
    {
        openAI_APIKey = APIKeyLoader.LoadAPIKey();
        if (!audioSource) audioSource = GetComponent<AudioSource>();
        if (!audioSource) audioSource = gameObject.AddComponent<AudioSource>();
        audioSource.playOnAwake = false;
        audioSource.loop = false;
        audioSource.spatialBlend = 0f; // 2D narration
        audioSource.volume = 1f;
    }

    [System.Serializable]
    public class TTSRequest
    {
        public string model = "tts-1";
        public string input;
        public string voice = "nova";
        // Ask the API to return the format we plan to decode
        public string format = "wav";           // accepted: wav | mp3 | ogg | flac
        public string response_format = "wav";  // compatibility
    }

    public IEnumerator SpeakText(string text)
    {
        if (isSpeaking)
        {
            Debug.LogWarning("TTS is already playing. Wait before making another request.");
            yield break;
        }
        if (string.IsNullOrWhiteSpace(openAI_APIKey))
        {
            Debug.LogError("OpenAITTS: Missing API key.");
            yield break;
        }
        if (string.IsNullOrWhiteSpace(text))
            yield break;

        isSpeaking = true;

        var req = new TTSRequest
        {
            model = string.IsNullOrWhiteSpace(model) ? "tts-1" : model,
            input = text,
            voice = string.IsNullOrWhiteSpace(voice) ? "nova" : voice,
            format = string.IsNullOrWhiteSpace(format) ? "wav" : format.ToLowerInvariant(),
            response_format = string.IsNullOrWhiteSpace(format) ? "wav" : format.ToLowerInvariant()
        };
        string json = JsonConvert.SerializeObject(req);

        using var http = new UnityWebRequest("https://api.openai.com/v1/audio/speech", "POST");
        http.uploadHandler = new UploadHandlerRaw(System.Text.Encoding.UTF8.GetBytes(json));
        http.downloadHandler = new DownloadHandlerBuffer();
        http.SetRequestHeader("Content-Type", "application/json");
        http.SetRequestHeader("Authorization", "Bearer " + openAI_APIKey);

        yield return http.SendWebRequest();

        if (http.responseCode == 429)
        {
            Debug.LogError("TTS Error: Rate limit (429). Retrying in 2s...");
            yield return new WaitForSeconds(2f);
            isSpeaking = false;
            yield return StartCoroutine(SpeakText(text));
            yield break;
        }

        if (http.result != UnityWebRequest.Result.Success)
        {
            string body = http.downloadHandler != null ? http.downloadHandler.text : "(no body)";
            Debug.LogError($"TTS Error: {http.error} (HTTP {http.responseCode})\n{body}");
            isSpeaking = false;
            yield break;
        }

        string contentType = http.GetResponseHeader("Content-Type") ?? "";
        byte[] data = http.downloadHandler.data ?? System.Array.Empty<byte>();

        if (logHeaders) Debug.Log($"[TTS] Content-Type: {contentType} | Bytes: {data.Length}");

        // If server returned JSON, log and bail (don’t try to decode JSON as audio)
        if (contentType.ToLowerInvariant().Contains("application/json"))
        {
            string body = http.downloadHandler.text;
            Debug.LogError("[TTS] Response is JSON (not audio). Body:\n" + body);
            isSpeaking = false;
            yield break;
        }

        // Quick signature for sanity (optional)
        if (logFirstBytes && data.Length >= 4)
        {
            var sig = System.Text.Encoding.ASCII.GetString(data, 0, Mathf.Min(16, data.Length)).Replace('\0', ' ');
            Debug.Log($"[TTS] First bytes: '{sig}'");
        }

        // Decide extension/type based on Content-Type; default to our requested format
        string ext = (req.format == "mp3") ? "mp3" : "wav";
        AudioType audioType = (req.format == "mp3") ? AudioType.MPEG : AudioType.WAV;

        var ct = contentType.ToLowerInvariant();
        if (ct.Contains("audio/mpeg") || ct.Contains("audio/mp3")) { ext = "mp3"; audioType = AudioType.MPEG; }
        else if (ct.Contains("audio/ogg")) { ext = "ogg"; audioType = AudioType.OGGVORBIS; }
        else if (ct.Contains("audio/wav") || ct.Contains("audio/x-wav") || ct.Contains("audio/wave")) { ext = "wav"; audioType = AudioType.WAV; }

        // Save audio to disk (optional; we can also keep bytes in memory)
        string filePath = Path.Combine(Application.persistentDataPath, $"tts_response.{ext}");
        try
        {
            File.WriteAllBytes(filePath, data);
            Debug.Log($"[TTS] Saved: {filePath} ({data.Length} bytes)");
        }
        catch (System.Exception ex)
        {
            Debug.LogError("Failed to write TTS audio file: " + ex.Message);
            isSpeaking = false;
            yield break;
        }

        // Try normal loader first
        bool played = false;
        yield return StartCoroutine(TryPlayWithUnityWebRequest(filePath, audioType, onPlayed: () => played = true));

        // Fallback: if WAV and normal load failed, decode WAV manually (PCM 16/24 or float32)
        if (!played && ext == "wav")
        {
            if (WavDecoder.TryDecode(File.ReadAllBytes(filePath), out var samples, out int channels, out int sampleRate))
            {
                var clip = AudioClip.Create("TTS_WAV", samples.Length / Mathf.Max(1, channels), channels, sampleRate, false);
                clip.SetData(samples, 0);
                audioSource.clip = clip;
                audioSource.Play();
                yield return new WaitForSeconds(clip.length + 0.2f);
                played = true;
            }
            else
            {
                Debug.LogError("[TTS] WAV fallback decode failed.");
            }
        }

        if (!played)
        {
            Debug.LogError("[TTS] Could not play audio (all paths failed).");
        }

        isSpeaking = false;
    }

    private IEnumerator TryPlayWithUnityWebRequest(string path, AudioType type, System.Action onPlayed)
    {
        var uri = new System.Uri(path).AbsoluteUri; // robust file:// URI

        using UnityWebRequest www = UnityWebRequestMultimedia.GetAudioClip(uri, type);
        yield return www.SendWebRequest();

        if (www.result != UnityWebRequest.Result.Success)
        {
            Debug.LogError("Audio playback error: " + www.error + " (" + uri + ")");
            // Last-resort: try WAV if we initially tried MP3
            if (type == AudioType.MPEG)
            {
                using var wavTry = UnityWebRequestMultimedia.GetAudioClip(uri, AudioType.WAV);
                yield return wavTry.SendWebRequest();
                if (wavTry.result == UnityWebRequest.Result.Success)
                {
                    var clip2 = DownloadHandlerAudioClip.GetContent(wavTry);
                    if (clip2)
                    {
                        audioSource.clip = clip2;
                        audioSource.Play();
                        yield return new WaitForSeconds(clip2.length + 0.3f);
                        onPlayed?.Invoke();
                        yield break;
                    }
                }
            }
            yield break;
        }

        AudioClip clip = DownloadHandlerAudioClip.GetContent(www);
        if (!clip)
        {
            Debug.LogError("Audio playback error: decoded clip is null (" + uri + ")");
            yield break;
        }

        audioSource.clip = clip;
        audioSource.Play();
        yield return new WaitForSeconds(clip.length + 0.3f);
        onPlayed?.Invoke();
    }
}