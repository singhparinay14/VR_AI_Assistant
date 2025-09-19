using System.Diagnostics;
using Unity.XR.CoreUtils;
using UnityEngine;
using Debug = UnityEngine.Debug;

[DefaultExecutionOrder(1000)] // run after XROrigin updates
[RequireComponent(typeof(XROrigin))]
public class FloorModeFixedHeight : MonoBehaviour
{
    [Header("Desired eye height (meters)")]
    [Range(1.2f, 2.1f)] public float targetEyeHeight = 1.7f;

    [Header("Safety clamp for extra offset applied to Camera Floor Offset Object")]
    public Vector2 offsetClamp = new Vector2(-0.5f, 0.8f); // min/max meters

    [Header("Optional smoothing")]
    public bool smooth = true;
    [Range(0.01f, 0.5f)] public float smoothLerp = 0.15f;

    XROrigin xrOrigin;
    Transform floorOffset; // xrOrigin.CameraFloorOffsetObject

    void Awake()
    {
        xrOrigin = GetComponent<XROrigin>();
        floorOffset = xrOrigin.CameraFloorOffsetObject ?
                      xrOrigin.CameraFloorOffsetObject.transform : null;

        if (!floorOffset)
        {
            Debug.LogError("FloorModeFixedHeight: XROrigin.CameraFloorOffsetObject is null.");
            enabled = false;
        }
        // Ensure we’re actually in Floor mode
        xrOrigin.RequestedTrackingOriginMode = XROrigin.TrackingOriginMode.Floor;
    }

    void LateUpdate()
    {
        if (!xrOrigin || !floorOffset) return;

        // Current eye height within the origin space (already includes floor offset)
        float currentH = xrOrigin.CameraInOriginSpaceHeight;

        // If tracking is lost, bail out to avoid big jumps
        if (float.IsNaN(currentH) || float.IsInfinity(currentH)) return;

        float delta = targetEyeHeight - currentH;
        if (Mathf.Abs(delta) < 0.001f) return;

        Vector3 lp = floorOffset.localPosition;
        float desiredY = Mathf.Clamp(lp.y + delta, offsetClamp.x, offsetClamp.y);

        if (smooth)
            lp.y = Mathf.Lerp(lp.y, desiredY, smoothLerp);
        else
            lp.y = desiredY;

        floorOffset.localPosition = lp;
    }
}