using UnityEngine;
using UnityEngine.InputSystem;


public class TeleportOnStickRelease : MonoBehaviour
{
    [Header("References")]
    public InputActionReference teleportModeActivate;   // XRI RightHand Locomotion/Teleport Mode (1D Axis on primary2DAxis/y with Press)
    public UnityEngine.XR.Interaction.Toolkit.Interactors.XRRayInteractor teleportRay;                 // RightHand_Teleport_Ray
    public UnityEngine.XR.Interaction.Toolkit.Locomotion.Teleportation.TeleportationProvider teleportationProvider; // On XR Origin
    public Transform forwardSource;                     // Main Camera

    void OnEnable()
    {
        if (teleportModeActivate?.action == null) return;
        teleportModeActivate.action.performed += OnActivatePerformed;
        teleportModeActivate.action.canceled += OnActivateCanceled;
        teleportModeActivate.action.Enable();
    }

    void OnDisable()
    {
        if (teleportModeActivate?.action == null) return;
        teleportModeActivate.action.performed -= OnActivatePerformed;
        teleportModeActivate.action.canceled -= OnActivateCanceled;
        teleportModeActivate.action.Disable();
    }

    void OnActivatePerformed(InputAction.CallbackContext _)
    {
        if (teleportRay) teleportRay.gameObject.SetActive(true);
    }

    void OnActivateCanceled(InputAction.CallbackContext _)
    {
        if (teleportRay == null || teleportationProvider == null) return;

        if (teleportRay.TryGetCurrent3DRaycastHit(out var hit))
        {
            var anchor = hit.transform.GetComponentInParent<UnityEngine.XR.Interaction.Toolkit.Locomotion.Teleportation.TeleportationAnchor>();
            Vector3 destPos;
            Quaternion destRot;

            if (anchor != null && anchor.teleportAnchorTransform != null)
            {
                destPos = anchor.teleportAnchorTransform.position;
                var fwd = Vector3.ProjectOnPlane(anchor.teleportAnchorTransform.forward, Vector3.up).normalized;
                destRot = Quaternion.LookRotation(fwd, Vector3.up);
            }
            else
            {
                destPos = hit.point;
                var fwd = forwardSource ? forwardSource.forward : Vector3.forward;
                fwd = Vector3.ProjectOnPlane(fwd, Vector3.up).normalized;
                destRot = Quaternion.LookRotation(fwd, Vector3.up);
            }

            var request = new UnityEngine.XR.Interaction.Toolkit.Locomotion.Teleportation.TeleportRequest
            {
                destinationPosition = destPos,
                destinationRotation = destRot,
                matchOrientation = UnityEngine.XR.Interaction.Toolkit.Locomotion.Teleportation.MatchOrientation.TargetUpAndForward
            };
            teleportationProvider.QueueTeleportRequest(request);
        }

        teleportRay.gameObject.SetActive(false);
    }
}