using System.Diagnostics;
using UnityEngine;
using Debug = UnityEngine.Debug;

public static class APIKeyLoader
{
    private static string cachedKey;

    public static string LoadAPIKey()
    {
        if (!string.IsNullOrEmpty(cachedKey))
            return cachedKey;

        // Looks for Assets/Resources/openai_key.txt (no extension in the Load call)
        TextAsset keyFile = Resources.Load<TextAsset>("openai_key");
        if (keyFile != null)
        {
            cachedKey = keyFile.text.Trim();
            return cachedKey;
        }

        Debug.LogError("Could not load OpenAI API key. Ensure 'openai_key.txt' exists in Assets/Resources/ and contains only the key.");
        return string.Empty;
    }
}