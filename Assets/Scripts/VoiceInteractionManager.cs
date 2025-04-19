using System.Collections;
using UnityEngine;
using UnityEngine.UI;

public class VoiceInteractionManager : MonoBehaviour
{
    [Header("References")]
    public VoiceRecorder voiceRecorder;
    public WhisperTranscriber whisperTranscriber;
    public ChatGPTManager chatGPTManager;
    public OpenAITTS openAITTS;
    public BotAI botAI;

    [Header("UI (Optional)")]
    public Button recordButton;
    public Text buttonText;

    private bool isRecording = false;

    void Start()
    {
        if (recordButton != null)
        {
            recordButton.onClick.AddListener(OnRecordButtonPressed);
        }
    }

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.R))
        {
            if (!isRecording)
                StartVoiceRecording();
            else
                StopAndProcessRecording();
        }
    }

    public void OnRecordButtonPressed()
    {
        if (!isRecording)
        {
            StartVoiceRecording();
        }
        else
        {
            StopAndProcessRecording();
        }
    }

    public void StartVoiceRecording()
    {
        isRecording = true;
        voiceRecorder.StartRecording();

        if (buttonText != null)
        {
            buttonText.text = "Stop";
        }

        Debug.Log("Voice recording started");
    }

    public void StopAndProcessRecording()
    {
        isRecording = false;
        voiceRecorder.StopRecording();

        if (buttonText != null)
        {
            buttonText.text = "Speak";
        }

        Debug.Log("Voice recording stopped. Processing...");

        StartCoroutine(ProcessVoiceInput());
    }

    private IEnumerator ProcessVoiceInput()
    {
        // Step 1: Transcribe with Whisper
        yield return whisperTranscriber.TranscribeAudio(voiceRecorder.recordedFilePath, (string transcription) =>
        {
            Debug.Log("Transcription: " + transcription);
            botAI.DisplayUserMessage(transcription); // optional

            // Step 2: Send to GPT
            StartCoroutine(chatGPTManager.SendMessageToOpenAI(transcription, (string gptReply) =>
            {
                Debug.Log("GPT says: " + gptReply);
                botAI.DisplayAIMessage(gptReply); // update chat box

                // Step 3: Send to TTS and speak
                StartCoroutine(openAITTS.SpeakText(gptReply));
            }));
        });
    }
}
