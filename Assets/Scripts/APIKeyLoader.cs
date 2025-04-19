using UnityEngine;

public static class APIKeyLoader
{
    private static string cachedKey;

    public static string LoadAPIKey()
    {
        if (!string.IsNullOrEmpty(cachedKey))
            return cachedKey;

        TextAsset keyFile = Resources.Load<TextAsset>("openai_key");
        if (keyFile != null)
        {
            cachedKey = keyFile.text.Trim();
            return cachedKey;
        }

        Debug.LogError("Could not load OpenAI API key. Ensure 'openai_key.txt' exists in the Resources folder.");
        return string.Empty;
    }
}
