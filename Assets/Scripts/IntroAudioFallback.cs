using UnityEngine;

[RequireComponent(typeof(AudioSource))]
public class IntroAudioFallback : MonoBehaviour
{
    [Header("Clip to play as the intro narration")]
    public AudioClip introClip;

    [Tooltip("Play automatically when the scene starts")]
    public bool playOnStart = true;

    [Tooltip("Delay in seconds before starting playback")]
    public float startDelay = 0f;

    private AudioSource source;

    void Awake()
    {
        source = GetComponent<AudioSource>();
        // Narration should be 2D so it’s not positional
        source.spatialBlend = 0f;
        source.playOnAwake = false;
        source.loop = false;
        source.volume = 1.0f;
    }

    void Start()
    {
        if (playOnStart && introClip != null)
        {
            if (startDelay > 0f)
            {
                source.clip = introClip;
                source.PlayDelayed(startDelay);
            }
            else
            {
                source.PlayOneShot(introClip);
            }
        }
    }

    // Optional: Call from a moderator button to replay the intro
    public void Replay()
    {
        if (introClip == null) return;
        source.Stop();
        source.PlayOneShot(introClip);
    }
}