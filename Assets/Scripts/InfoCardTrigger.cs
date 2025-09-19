using UnityEngine;
using UnityEngine.InputSystem; // NEW

public class InfoCardTrigger : MonoBehaviour
{
    [Header("References")]
    public GameObject infoCard;
    public AudioSource audioSource;

    [Header("Car Info")]
    public string carName;
    public string modelYear;
    public string brandName;

    [Header("XR Input")]
    public InputActionProperty playAction; // assign "XR Controller → RightHand → PrimaryButton"

    private bool playerInside = false;

    private void Start()
    {
        if (infoCard != null)
        {
            infoCard.SetActive(false);
            foreach (var t in infoCard.GetComponentsInChildren<TMPro.TextMeshProUGUI>())
            {
                if (t.name == "CarNameText") t.text = carName;
                if (t.name == "ModelYearText") t.text = modelYear;
                if (t.name == "BrandNameText") t.text = brandName;
            }
        }

        if (playAction.reference != null)
            playAction.action.performed += OnPlayAction;
    }

    private void OnDestroy()
    {
        if (playAction.reference != null)
            playAction.action.performed -= OnPlayAction;
    }

    private void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag("Player"))
        {
            playerInside = true;
            infoCard.SetActive(true);
        }
    }

    private void OnTriggerExit(Collider other)
    {
        if (other.CompareTag("Player"))
        {
            playerInside = false;
            infoCard.SetActive(false);
        }
    }

    private void OnPlayAction(InputAction.CallbackContext ctx)
    {
        if (playerInside) PlayAudioInfo();
    }

    public void PlayAudioInfo()
    {
        if (audioSource != null && !audioSource.isPlaying)
            audioSource.Play();
    }
}
