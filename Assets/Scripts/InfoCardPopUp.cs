using UnityEngine;
using TMPro; // needed for TextMeshPro

public class InfoCardTrigger : MonoBehaviour
{
    [Header("References")]
    public GameObject infoCard;     // assign the prefab instance
    public AudioSource audioSource; // narration audio

    [Header("Car Info")]
    public string carName;
    public string modelYear;
    public string brandName;

    private void Start()
    {
        if (infoCard != null)
        {
            infoCard.SetActive(false);

            // Auto-fill text fields
            TextMeshProUGUI[] texts = infoCard.GetComponentsInChildren<TextMeshProUGUI>();
            foreach (var t in texts)
            {
                if (t.name == "CarNameText") t.text = carName;
                if (t.name == "ModelYearText") t.text = modelYear;
                if (t.name == "BrandNameText") t.text = brandName;
            }
        }
    }

    private void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag("Player"))
        {
            infoCard.SetActive(true);
        }
    }

    private void OnTriggerExit(Collider other)
    {
        if (other.CompareTag("Player"))
        {
            infoCard.SetActive(false);
        }
    }

    // Called by Hear More button
    public void PlayAudioInfo()
    {
        if (audioSource != null)
            audioSource.Play();
    }
}
