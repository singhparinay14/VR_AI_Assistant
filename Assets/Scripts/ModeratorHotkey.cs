using UnityEngine;
using UnityEngine.SceneManagement;

public class ModeratorHotkeys : MonoBehaviour
{
    void Update()
    {
        if (Input.GetKeyDown(KeyCode.F1))
            SceneManager.LoadScene("Gallery_AI", LoadSceneMode.Single);

        if (Input.GetKeyDown(KeyCode.F2))
            SceneManager.LoadScene("Gallery_NonAI", LoadSceneMode.Single);

        if (Input.GetKeyDown(KeyCode.F3))
            SceneManager.LoadScene("Welcome", LoadSceneMode.Single);
    }
}
