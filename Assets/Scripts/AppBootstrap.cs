using UnityEngine;

public enum AppMode { Welcome, AI, Offline }

public class AppBootstrap : MonoBehaviour
{
    public static AppBootstrap I { get; private set; }
    public AppMode mode = AppMode.Welcome;

    void Awake()
    {
        if (I != null) { Destroy(gameObject); return; }
        I = this;
        DontDestroyOnLoad(gameObject);
    }

    public void SetMode(AppMode m) => mode = m;
}
