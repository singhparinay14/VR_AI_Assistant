using UnityEngine;
using UnityEngine.InputSystem;
using System.Collections.Generic;
using Debug = UnityEngine.Debug;

public class RayModeSwitcher : MonoBehaviour
{
    public enum Strategy { ExclusiveHold, ExclusiveToggle, BothVisible }

    [Header("Mode")]
    public Strategy strategy = Strategy.ExclusiveHold;
    public bool defaultToTeleport = true; // When nothing is held in ExclusiveHold

    [Header("Ray Sets (assign in Inspector)")]
    public GameObject[] uiRays;        // e.g., RightHand_UI_Ray, LeftHand_UI_Ray
    public GameObject[] teleportRays;  // e.g., RightHand_Teleport_Ray, LeftHand_Teleport_Ray

    [Header("Input (enable any that should trigger the mode)")]
    public InputActionReference[] uiAimActions;       // e.g., XRI RightHand/PrimaryButton, XRI LeftHand/PrimaryButton
    public InputActionReference[] teleportAimActions; // e.g., XRI RightHand/Teleport Mode Activate, XRI LeftHand/Teleport Mode Activate

    int uiHoldCount = 0;
    int tpHoldCount = 0;
    bool uiMode = false; // For ExclusiveToggle

    // Keep delegates so we can unsubscribe if needed
    readonly Dictionary<InputAction, System.Action<InputAction.CallbackContext>> startedSubs = new();
    readonly Dictionary<InputAction, System.Action<InputAction.CallbackContext>> canceledSubs = new();

    void OnEnable()
    {
        SubAll(uiAimActions, isUI: true);
        SubAll(teleportAimActions, isUI: false);

        if (strategy == Strategy.ExclusiveToggle)
            uiMode = !defaultToTeleport;

        ApplyState(initial: true);
    }

    void OnDisable()
    {
        UnsubAll(uiAimActions);
        UnsubAll(teleportAimActions);
    }

    void SubAll(InputActionReference[] actions, bool isUI)
    {
        if (actions == null) return;
        foreach (var aref in actions)
        {
            if (aref == null || aref.action == null) continue;
            var a = aref.action;
            a.Enable();

            System.Action<InputAction.CallbackContext> started = _ =>
            {
                if (isUI)
                {
                    uiHoldCount++;
                    if (strategy == Strategy.ExclusiveToggle) { uiMode = true; }
                }
                else
                {
                    tpHoldCount++;
                    if (strategy == Strategy.ExclusiveToggle) { uiMode = false; }
                }
                if (strategy == Strategy.ExclusiveHold) ApplyState();
                else if (strategy == Strategy.ExclusiveToggle) ApplyState();
            };

            System.Action<InputAction.CallbackContext> canceled = _ =>
            {
                if (isUI) uiHoldCount = Mathf.Max(0, uiHoldCount - 1);
                else tpHoldCount = Mathf.Max(0, tpHoldCount - 1);
                if (strategy == Strategy.ExclusiveHold) ApplyState();
            };

            a.started += started;
            a.canceled += canceled;

            startedSubs[a] = started;
            canceledSubs[a] = canceled;
        }
    }

    void UnsubAll(InputActionReference[] actions)
    {
        if (actions == null) return;
        foreach (var aref in actions)
        {
            if (aref == null || aref.action == null) continue;
            var a = aref.action;
            if (startedSubs.TryGetValue(a, out var s)) a.started -= s;
            if (canceledSubs.TryGetValue(a, out var c)) a.canceled -= c;
            a.Disable();
        }
    }

    void ApplyState(bool initial = false)
    {
        if (strategy == Strategy.BothVisible)
        {
            SetActive(uiRays, true);
            SetActive(teleportRays, true);
            return;
        }

        bool showUI;

        if (strategy == Strategy.ExclusiveHold)
        {
            // If only one type is held, show that one. If both or none, use default.
            if (uiHoldCount > 0 && tpHoldCount == 0) showUI = true;
            else if (tpHoldCount > 0 && uiHoldCount == 0) showUI = false;
            else showUI = !defaultToTeleport;
        }
        else // ExclusiveToggle
        {
            showUI = uiMode;
        }

        SetActive(uiRays, showUI);
        SetActive(teleportRays, !showUI);
    }

    void SetActive(GameObject[] set, bool value)
    {
        if (set == null) return;
        foreach (var go in set) if (go) go.SetActive(value);
    }
}