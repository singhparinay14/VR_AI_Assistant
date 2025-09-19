using UnityEngine;
using UnityEngine.UI;

public class CarAudioDescription : MonoBehaviour
{
    [Header("References")]
    public Transform playerHead;              // Assign XR Origin > Camera Offset > Main Camera
    public Canvas uiCanvas;                   // World-space canvas with a single Button
    public Button playPauseButton;            // The Button on the canvas
    public AudioSource audioSource;           // On this car (spatialized if you like)
    public AudioClip descriptionClip;         // Per-car clip

    [Header("Behavior")]
    [Range(0.5f, 10f)] public float showRadius = 3.0f;
    public bool billboardToPlayer = true;

    // Keep track of the last (globally) playing description so we stop it before playing a new one
    private static CarAudioDescription _currentlyPlaying;

    void Reset()
    {
        // Helpful defaults if you add the component in Editor
        audioSource = GetComponent<AudioSource>();
        if (audioSource == null) audioSource = gameObject.AddComponent<AudioSource>();
        audioSource.playOnAwake = false;
        audioSource.loop = false;
        audioSource.spatialBlend = 1f; // 3D spatialized sound (0=2D, 1=3D)
    }

    void Awake()
    {
        if (uiCanvas != null) uiCanvas.enabled = false;
        if (playPauseButton != null) playPauseButton.onClick.AddListener(OnPlayPauseClicked);
        if (audioSource != null && descriptionClip != null) audioSource.clip = descriptionClip;
    }

    void Update()
    {
        if (playerHead == null || uiCanvas == null) return;

        float dist = Vector3.Distance(playerHead.position, transform.position);
        bool shouldShow = dist <= showRadius;

        if (uiCanvas.enabled != shouldShow)
            uiCanvas.enabled = shouldShow;

        if (billboardToPlayer && uiCanvas.enabled)
        {
            // Face the player smoothly
            Vector3 toPlayer = playerHead.position - uiCanvas.transform.position;
            toPlayer.y = 0f;
            if (toPlayer.sqrMagnitude > 0.001f)
                uiCanvas.transform.rotation = Quaternion.Slerp(
                    uiCanvas.transform.rotation,
                    Quaternion.LookRotation(toPlayer),
                    Time.deltaTime * 10f
                );
        }

        // If player walks away while playing, optionally fade/stop (simple stop here)
        if (!shouldShow && audioSource != null && audioSource.isPlaying)
        {
            audioSource.Stop();
            if (_currentlyPlaying == this) _currentlyPlaying = null;
            UpdateButtonLabel();
        }
    }

    private void OnPlayPauseClicked()
    {
        if (audioSource == null || descriptionClip == null) return;

        // Stop whatever else is playing
        if (_currentlyPlaying != null && _currentlyPlaying != this)
        {
            if (_currentlyPlaying.audioSource != null)
                _currentlyPlaying.audioSource.Stop();
            _currentlyPlaying.UpdateButtonLabel();
        }

        if (audioSource.isPlaying)
        {
            audioSource.Pause();
            _currentlyPlaying = null;
        }
        else
        {
            audioSource.clip = descriptionClip;
            audioSource.Play();
            _currentlyPlaying = this;
        }

        UpdateButtonLabel();
    }

    private void UpdateButtonLabel()
    {
        if (playPauseButton == null) return;
        var text = playPauseButton.GetComponentInChildren<TMPro.TextMeshProUGUI>();
        if (text != null) text.text = (audioSource != null && audioSource.isPlaying) ? "Pause" : "Play";
    }

    // Draw the trigger radius in the editor
    void OnDrawGizmosSelected()
    {
        Gizmos.color = new Color(0f, 0.6f, 1f, 0.25f);
        Gizmos.DrawSphere(transform.position, showRadius);
        Gizmos.color = new Color(0f, 0.6f, 1f, 0.9f);
        Gizmos.DrawWireSphere(transform.position, showRadius);
    }
}
