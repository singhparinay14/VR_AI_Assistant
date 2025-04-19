using UnityEngine;
using TMPro;
using UnityEngine.UI;

public class BotAI : MonoBehaviour
{
    public TextMeshProUGUI chatText;
    public ScrollRect scrollRect;

    public void DisplayUserMessage(string message)
    {
        chatText.text += "\n<color=#00ff00><b>You:</b></color> " + message;
        ScrollToBottom();
    }

    public void DisplayAIMessage(string message)
    {
        chatText.text += "\n<color=#00ffff><b>AI:</b></color> " + message;
        ScrollToBottom();
    }

    private void ScrollToBottom()
    {
        Canvas.ForceUpdateCanvases();
        scrollRect.verticalNormalizedPosition = 0f;
        Canvas.ForceUpdateCanvases();
    }
}
