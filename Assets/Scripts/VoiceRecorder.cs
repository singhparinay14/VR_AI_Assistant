using System.Collections;
using System.IO;
using UnityEngine;

[RequireComponent(typeof(AudioSource))]
public class VoiceRecorder : MonoBehaviour
{
    public AudioSource audioSource;
    public string recordedFilePath;

    private AudioClip recordedClip;
    private bool isRecording = false;
    private const int recordingLength = 10; // seconds
    private const int sampleRate = 16000;

    public void StartRecording()
    {
        if (isRecording) return;

        isRecording = true;
        recordedClip = Microphone.Start(null, false, recordingLength, sampleRate);
        Debug.Log("Recording started...");
    }

    public void StopRecording()
    {
        if (!isRecording) return;

        isRecording = false;
        Microphone.End(null);
        Debug.Log("Recording stopped.");

        SaveWav(recordedClip);
    }

    private void SaveWav(AudioClip clip)
    {
        string filename = "recorded_audio";
        bool success = SavWav.Save(filename, clip);

        if (success)
        {
            recordedFilePath = Path.Combine(Application.persistentDataPath, filename + ".wav");
            Debug.Log("Audio saved to: " + recordedFilePath);
        }
        else
        {
            Debug.LogError("Failed to save audio.");
        }
    }
}
