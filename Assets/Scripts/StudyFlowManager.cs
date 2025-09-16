using System.Diagnostics;
using TMPro;
using UnityEngine;
using UnityEngine.SceneManagement;

public class StudyFlowManager : MonoBehaviour
{
    [Header("Intro UI")]
    [TextArea(3, 10)]
    public string introMessage = "Welcome! In this study, you’ll briefly explore the open world. Then, please proceed to one of two galleries when instructed by the moderator.";
    public Canvas introCanvas;                // World-space or Screen-space
    public TextMeshProUGUI introText;         // Bind to a TMP text showing intro
    public GameObject introContinueButton;    // Optional: a Continue button on the intro UI

    [Header("Routing")]
    [Tooltip("If true, participant walks through doors. The moderator can unlock/select doors. If false, moderator routes directly (no doors needed).")]
    public bool useDoors = true;
    public TeleportDoor doorGalleryAI;        // Assign a door that teleports to Gallery_AI
    public TeleportDoor doorGalleryNonAI;     // Assign a door that teleports to Gallery_NonAI

    [Tooltip("Used for direct routing (or fallback if doors are missing). Names must match Build Settings.")]
    public string sceneGalleryAI = "Gallery_AI";
    public string sceneGalleryNonAI = "Gallery_NonAI";

    [Header("Optional: Audio Intro via OpenAITTS")]
    public bool speakIntro = false;
    public MonoBehaviour openAiTtsComponent;  // Assign your OpenAITTS component if you want spoken intro (expects a public Speak(string) method)

    void Start()
    {
        // Show intro
        if (introCanvas != null) introCanvas.enabled = true;
        if (introText != null) introText.text = introMessage;

        // Hide doors until intro is complete
        LockAllDoors();

        // Speak intro if configured
        if (speakIntro && openAiTtsComponent != null)
        {
            // Assumes your OpenAITTS has a public method Speak(string text)
            var method = openAiTtsComponent.GetType().GetMethod("Speak");
            if (method != null)
            {
                method.Invoke(openAiTtsComponent, new object[] { introMessage });
            }
            else
            {
                UnityEngine.Debug.LogWarning("StudyFlowManager: openAiTtsComponent doesn't have a public Speak(string) method.");
            }
        }
    }

    // Called from the intro UI "Continue" button OR from ModeratorPanel
    public void OnIntroComplete()
    {
        if (introCanvas != null) introCanvas.enabled = false;

        if (useDoors)
        {
            // Let the participant stand by; moderator will instruct which door to take (or unlock both)
            UnlockBothDoors(); // Or keep them locked until moderator chooses; adjust to your protocol
        }
    }

    // Moderator chooses AI condition
    public void SendToGalleryAI()
    {
        if (useDoors)
        {
            // Emphasize AI door and hide the other (participant walks to the door)
            if (doorGalleryAI != null) doorGalleryAI.gameObject.SetActive(true);
            if (doorGalleryNonAI != null) doorGalleryNonAI.gameObject.SetActive(false);
        }
        else
        {
            LoadSceneSafe(sceneGalleryAI);
        }
    }

    // Moderator chooses Non-AI condition
    public void SendToGalleryNonAI()
    {
        if (useDoors)
        {
            if (doorGalleryNonAI != null) doorGalleryNonAI.gameObject.SetActive(true);
            if (doorGalleryAI != null) doorGalleryAI.gameObject.SetActive(false);
        }
        else
        {
            LoadSceneSafe(sceneGalleryNonAI);
        }
    }

    // Optional: Moderator can unlock both doors (participant will be told which one to enter)
    public void UnlockBothDoors()
    {
        if (doorGalleryAI != null) doorGalleryAI.gameObject.SetActive(true);
        if (doorGalleryNonAI != null) doorGalleryNonAI.gameObject.SetActive(true);
    }

    // Optional: Moderator can lock all doors (e.g., during instructions)
    public void LockAllDoors()
    {
        if (doorGalleryAI != null) doorGalleryAI.gameObject.SetActive(false);
        if (doorGalleryNonAI != null) doorGalleryNonAI.gameObject.SetActive(false);
    }

    void LoadSceneSafe(string sceneName)
    {
        if (string.IsNullOrWhiteSpace(sceneName))
        {
            UnityEngine.Debug.LogError("StudyFlowManager: Target scene name is empty.");
            return;
        }

        // Ensure the scene is added under File > Build Settings
        SceneManager.LoadScene(sceneName, LoadSceneMode.Single);
    }
}