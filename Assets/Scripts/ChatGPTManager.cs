using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Networking;
using Newtonsoft.Json;

[System.Serializable]
public class Message
{
    public string role;
    public string content;
}

[System.Serializable]
public class Choice
{
    public Message message;
}

[System.Serializable]
public class OpenAIResponse
{
    public List<Choice> choices;
}

public class ChatGPTManager : MonoBehaviour
{
    private string openAI_APIKey;
    private readonly string openAI_Endpoint = "https://api.openai.com/v1/chat/completions";

    private void Awake()
    {
        openAI_APIKey = APIKeyLoader.LoadAPIKey();
    }

    public IEnumerator SendMessageToOpenAI(string userMessage, System.Action<string> callback)
    {
        var requestData = new
        {
            model = "gpt-3.5-turbo",
            messages = new[] {
                new { role = "system", content = "You are a helpful assistant in VR." },
                new { role = "user", content = userMessage }
            },
            temperature = 0.7f
        };

        string jsonData = JsonConvert.SerializeObject(requestData);
        byte[] jsonToSend = System.Text.Encoding.UTF8.GetBytes(jsonData);

        UnityWebRequest request = new UnityWebRequest(openAI_Endpoint, "POST");
        request.uploadHandler = new UploadHandlerRaw(jsonToSend);
        request.downloadHandler = new DownloadHandlerBuffer();
        request.SetRequestHeader("Content-Type", "application/json");
        request.SetRequestHeader("Authorization", $"Bearer {openAI_APIKey}");

        yield return request.SendWebRequest();

        if (request.result != UnityWebRequest.Result.Success)
        {
            Debug.LogError("Error: " + request.error);
            callback("Sorry, something went wrong.");
        }
        else
        {
            string responseJson = request.downloadHandler.text;
            var response = JsonConvert.DeserializeObject<OpenAIResponse>(responseJson);
            string aiResponse = response.choices[0].message.content;
            callback(aiResponse);
        }
    }
}
