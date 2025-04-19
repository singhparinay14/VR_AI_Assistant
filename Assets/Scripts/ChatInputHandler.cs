using UnityEngine;
using TMPro;
using UnityEngine.UI;

public class ChatInputHandler : MonoBehaviour
{
    public TMP_InputField userInputField;
    public Button sendButton;
    public ChatGPTManager chatGPTManager;
    public BotAI botAI;

    void Start()
    {
        sendButton.onClick.AddListener(SendMessageToChatGPT);
        userInputField.onSubmit.AddListener((string _) => SendMessageToChatGPT());
    }

    void SendMessageToChatGPT()
    {
        string userMessage = userInputField.text.Trim();
        if (string.IsNullOrEmpty(userMessage))
            return;

        userInputField.text = "";

        botAI.DisplayUserMessage(userMessage);

        StartCoroutine(chatGPTManager.SendMessageToOpenAI(userMessage, botAI.DisplayAIMessage));
    }
}
