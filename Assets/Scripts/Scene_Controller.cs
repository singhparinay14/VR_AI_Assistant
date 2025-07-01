using UnityEngine;
using UnityEngine.SceneManagement;

public class SceneLoader : MonoBehaviour
{
    public void LoadGalleryScene()
    {
        SceneManager.LoadScene("Gallery_AI");
    }
}
