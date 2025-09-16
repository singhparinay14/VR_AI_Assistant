using UnityEngine;

public class ModeratorPanel : MonoBehaviour
{
    [Header("UI Panel")]
    public Canvas panel;                 // A small canvas with moderator buttons
    public KeyCode toggleKey = KeyCode.M;

    [Header("Flow")]
    public StudyFlowManager flow;

    void Start()
    {
        if (panel != null) panel.enabled = false;
    }

    void Update()
    {
        // Desktop/editor convenience toggle
        if (Input.GetKeyDown(toggleKey))
        {
            TogglePanel();
        }
    }

    public void TogglePanel()
    {
        if (panel != null) panel.enabled = !panel.enabled;
    }

    // Button hooks
    public void OnIntroComplete()
    {
        if (flow != null) flow.OnIntroComplete();
        if (panel != null) panel.enabled = false;
    }

    public void OnSendToAI()
    {
        if (flow != null) flow.SendToGalleryAI();
    }

    public void OnSendToNonAI()
    {
        if (flow != null) flow.SendToGalleryNonAI();
    }

    public void OnUnlockDoors()
    {
        if (flow != null) flow.UnlockBothDoors();
    }

    public void OnLockDoors()
    {
        if (flow != null) flow.LockAllDoors();
    }
}