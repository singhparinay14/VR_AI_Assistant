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
    [Tooltip("TTS model. If you have access, gpt-4o-mini-tts is also supported.")]
    public string model = "tts-1";
    [Tooltip("Common voices: alloy, nova")]
    public string voice = "nova";
    [Tooltip("mp3 or wav. MP3 is most reliable with Unity/FM O D on Windows. WAV is fine with the manual decoder below.")]
    public string responseFormat = "mp3"; // default to mp3 for reliability

    [Header("Debug")]
    public bool logHeaders = true;
    public bool logFirstBytes = true;

    void Awake()
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
        public string model;
        public string input;
        public string voice;
        // API expects response_format to select container/codec
        public string response_format;  // wav | mp3 | ogg | flac
    }

    public IEnumerator SpeakText(string text)
    {
        if (isSpeaking)
        {
            Debug.LogWarning("TTS already speaking.");
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
        try
        {
            var req = new TTSRequest
            {
                model = string.IsNullOrWhiteSpace(model) ? "tts-1" : model,
                input = text,
                voice = string.IsNullOrWhiteSpace(voice) ? "nova" : voice,
                response_format = string.IsNullOrWhiteSpace(responseFormat) ? "mp3" : responseFormat.ToLowerInvariant()
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
                // Clear flag before retry to avoid "already speaking"
                isSpeaking = false;
                yield return StartCoroutine(SpeakText(text));
                yield break;
            }

            if (http.result != UnityWebRequest.Result.Success)
            {
                string body = http.downloadHandler != null ? http.downloadHandler.text : "(no body)";
                Debug.LogError($"TTS Error: {http.error} (HTTP {http.responseCode})\n{body}");
                yield break;
            }

            string contentType = http.GetResponseHeader("Content-Type") ?? "";
            byte[] data = http.downloadHandler.data ?? System.Array.Empty<byte>();

            if (logHeaders) Debug.Log($"[TTS] Content-Type: {contentType} | Bytes: {data.Length}");
            if (contentType.ToLowerInvariant().Contains("application/json"))
            {
                Debug.LogError("[TTS] Response is JSON (not audio). Body:\n" + http.downloadHandler.text);
                yield break;
            }
            if (logFirstBytes && data.Length >= 4)
            {
                var sig = System.Text.Encoding.ASCII.GetString(data, 0, Mathf.Min(16, data.Length)).Replace('\0', ' ');
                Debug.Log($"[TTS] First bytes: '{sig}'");
            }

            // Decide type from header, default to requested response_format
            string fmt = req.response_format;
            var ct = contentType.ToLowerInvariant();
            if (ct.Contains("audio/mpeg") || ct.Contains("audio/mp3")) fmt = "mp3";
            else if (ct.Contains("audio/ogg")) fmt = "ogg";
            else if (ct.Contains("audio/wav") || ct.Contains("audio/x-wav") || ct.Contains("audio/wave")) fmt = "wav";

            bool played = false;

            // If WAV requested or returned, first try robust manual decode. Otherwise write to disk and use Unity loader.
            if (fmt == "wav")
            {
                if (WavDecoder.TryDecode(data, out var samples, out int channels, out int sampleRate))
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
                    Debug.LogWarning("[TTS] WAV manual decode failed; will try Unity loader.");
                }
            }

            if (!played)
            {
                string ext = (fmt == "mp3") ? "mp3" : (fmt == "ogg" ? "ogg" : "wav");
                string filePath = Path.Combine(Application.persistentDataPath, $"tts_response.{ext}");
                try
                {
                    File.WriteAllBytes(filePath, data);
                    Debug.Log($"[TTS] Saved: {filePath} ({data.Length} bytes)");
                }
                catch (System.Exception ex)
                {
                    Debug.LogError("Failed to write TTS audio file: " + ex.Message);
                    yield break;
                }

                yield return StartCoroutine(TryPlayWithUnityWebRequest(filePath, ext, () => played = true));

                // Final fallback: if WAV and Unity loader failed, try manual decode again from file bytes
                if (!played && ext == "wav")
                {
                    if (WavDecoder.TryDecode(File.ReadAllBytes(filePath), out var samples2, out int ch2, out int sr2))
                    {
                        var clip2 = AudioClip.Create("TTS_WAV", samples2.Length / Mathf.Max(1, ch2), ch2, sr2, false);
                        clip2.SetData(samples2, 0);
                        audioSource.clip = clip2;
                        audioSource.Play();
                        yield return new WaitForSeconds(clip2.length + 0.2f);
                        played = true;
                    }
                    else
                    {
                        Debug.LogError("[TTS] WAV fallback decode failed.");
                    }
                }
            }

            if (!played)
                Debug.LogError("[TTS] Could not play audio (all paths failed).");
        }
        finally
        {
            isSpeaking = false;
        }
    }

    private IEnumerator TryPlayWithUnityWebRequest(string path, string ext, System.Action onPlayed)
    {
        // Robust file URI for Windows
        string uri = "file:///" + path.Replace("\\", "/");

        AudioType audioType = AudioType.UNKNOWN;
        switch (ext)
        {
            case "mp3": audioType = AudioType.MPEG; break;
            case "ogg": audioType = AudioType.OGGVORBIS; break;
            case "wav": audioType = AudioType.WAV; break;
        }

        using UnityWebRequest www = UnityWebRequestMultimedia.GetAudioClip(uri, audioType);
        yield return www.SendWebRequest();

        if (www.result != UnityWebRequest.Result.Success)
        {
            Debug.LogError("Audio playback error: " + www.error + " (" + uri + ")");
            // If we guessed MPEG but it's actually WAV, try WAV as a last resort
            if (audioType == AudioType.MPEG)
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
                    }
                }
            }
            yield break;
        }

        var clip = DownloadHandlerAudioClip.GetContent(www);
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

    // Optional convenience for StudyFlowManager reflection path
    public void Speak(string text)
    {
        StartCoroutine(SpeakText(text));
    }
}