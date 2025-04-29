using UnityEngine;
using System.Collections.Generic;
using System.Linq;

public class VisionContextProvider : MonoBehaviour
{
    public YoloObjectDetector detector;   // assign in Inspector

    string currentContext = "nothing";

    void OnEnable()
    {
        detector.OnDetections += Handle;
    }
    void OnDisable()
    {
        detector.OnDetections -= Handle;
    }

    void Handle(List<DetectionInfo> list)
    {
        if (list == null || list.Count == 0)
        {
            currentContext = "nothing";
            return;
        }

        // e.g. "a blue mug, a white chair"
        currentContext = string.Join(", ",
            list.Select(d => $"a {d.colour} {d.label}"));
    }

    public string GetContext()
    {
        return currentContext;
    }
}
