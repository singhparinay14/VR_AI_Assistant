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
    public KeyCode micToggleKey = KeyCode.Keypad0;


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
        if (Input.GetKeyDown(micToggleKey))
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
        yield return whisperTranscriber.TranscribeAudio(voiceRecorder.recordedFilePath, (string transcription) =>
        {
            Debug.Log("Transcription: " + transcription);
            botAI.DisplayUserMessage(transcription);

            if (!string.IsNullOrWhiteSpace(transcription))
            {
                StartCoroutine(DelayedSend(transcription));
            }
        });
    }

    private IEnumerator DelayedSend(string message)
    {
        yield return new WaitUntil(() => chatGPTManager.HasContextReady());
        yield return new WaitForSeconds(0.2f);

        yield return chatGPTManager.SendMessageToOpenAI(message, (string gptReply) =>
        {
            Debug.Log("GPT says: " + gptReply);
            botAI.DisplayAIMessage(gptReply);
            StartCoroutine(openAITTS.SpeakText(gptReply));
        });
    }
}
