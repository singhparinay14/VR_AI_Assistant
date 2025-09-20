using System.Collections;
using System.Diagnostics;
using UnityEngine;
using UnityEngine.InputSystem; // NEW
using UnityEngine.UI;
using Debug = UnityEngine.Debug;

public class VoiceInteractionManager : MonoBehaviour
{
    [Header("References")]
    public VoiceRecorder voiceRecorder;
    public WhisperTranscriber whisperTranscriber;
    public ChatGPTManager chatGPTManager;
    public OpenAITTS openAITTS;
    public BotAI botAI;

    [Header("Keyboard Fallback")]
    public KeyCode micToggleKey = KeyCode.Keypad0;

    [Header("XR Input")]
    public InputActionProperty micToggleAction; // NEW (assign in Inspector)

    [Header("UI (Optional)")]
    public Button recordButton;
    public Text buttonText;

    private bool isRecording = false;

    void OnEnable()
    {
        var action = micToggleAction.action;          // ✅ works for inline OR asset reference
        if (action != null)
        {
            action.started += OnMicStarted;          // A button down
            action.canceled += OnMicCanceled;         // A button up
            action.Enable();
        }
        else
        {
            Debug.LogWarning("VoiceInteractionManager: Mic Toggle action is not assigned.");
        }
    }

    void OnDisable()
    {
        var action = micToggleAction.action;
        if (action != null)
        {
            action.started -= OnMicStarted;
            action.canceled -= OnMicCanceled;
            action.Disable();
        }
    }

    private void OnMicStarted(InputAction.CallbackContext ctx)
    {
        Debug.Log("[PTT] A down");
        if (!isRecording) StartVoiceRecording();
    }

    private void OnMicCanceled(InputAction.CallbackContext ctx)
    {
        Debug.Log("[PTT] A up");
        if (isRecording) StopAndProcessRecording();
    }

    void Start()
    {
        if (recordButton != null)
            recordButton.onClick.AddListener(OnRecordButtonPressed);
    }

    void Update()
    {
        // Keyboard fallback
        if (Input.GetKeyDown(micToggleKey))
        {
            ToggleRecording();
        }
    }

    private void OnMicTogglePerformed(InputAction.CallbackContext ctx) // NEW
    {
        // Debounced by the Input System "Press" interaction.
        ToggleRecording();
    }

    private void ToggleRecording() // NEW
    {
        if (!isRecording) StartVoiceRecording();
        else StopAndProcessRecording();
    }

    public void OnRecordButtonPressed()
    {
        ToggleRecording();
    }

    public void StartVoiceRecording()
    {
        isRecording = true;
        voiceRecorder.StartRecording();

        if (buttonText != null) buttonText.text = "Stop";
        Debug.Log("Voice recording started");
    }

    public void StopAndProcessRecording()
    {
        isRecording = false;
        voiceRecorder.StopRecording();

        if (buttonText != null) buttonText.text = "Speak";
        Debug.Log("Voice recording stopped. Processing...");

        StartCoroutine(ProcessVoiceInput());
    }

    private IEnumerator ProcessVoiceInput()
    {
        yield return whisperTranscriber.TranscribeAudio(voiceRecorder.recordedFilePath, (string transcription) =>
        {
            Debug.Log("Transcription: " + transcription);

            // Only proceed if we got real text (non-empty and not our error line)
            if (!string.IsNullOrWhiteSpace(transcription) && !transcription.StartsWith("Sorry"))
            {
                botAI.DisplayUserMessage(transcription);
                StartCoroutine(DelayedSend(transcription));
            }
            else
            {
                botAI.DisplayAIMessage("I couldn't hear that. Please try again.");
            }
        });
    }

    private IEnumerator DelayedSend(string message)
    {
        // waits until your vision context is ready before sending to GPT
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
