using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using UnityEngine;
using Debug = UnityEngine.Debug;

public class VisionSummaryLogger : MonoBehaviour
{
    [Header("Sources")]
    public YoloObjectDetector[] carDetectors;  // drag your 1 car detector (the one that says "initialized with 2 cameras")
    public PropsDetector[] propDetectors;      // drag the props detector

    [Header("Logging")]
    [Tooltip("0 = disable periodic logs; e.g., 30 logs about twice a second at 60 FPS")]
    public int logEveryNFrames = 30;
    public KeyCode logOnKey = KeyCode.F9;
    public bool onlyOnChange = false;   // if true, suppress identical lines

    private readonly Dictionary<UnityEngine.Object, List<DetectionInfo>> latestBySource = new();
    private string lastPrinted = "";

    IEnumerable<DetectionInfo> DedupByLabelAndProximity(IEnumerable<DetectionInfo> input, float radiusMeters)
    {
        const string NoHit = "nohit";
        float r2 = radiusMeters * radiusMeters;

        // 1) If we have a collider name ("surface"), use it as a stable key.
        var withSurface = input
            .Where(d => !string.IsNullOrEmpty(d.surface) && d.surface != NoHit)
            .GroupBy(d => (d.label, d.surface))
            .Select(g => g.OrderByDescending(x => x.confidence).First());

        // 2) For the rest (no/unstable surface), cluster by nearness within radius per label.
        var remaining = input
            .Where(d => string.IsNullOrEmpty(d.surface) || d.surface == NoHit)
            .OrderByDescending(d => d.confidence)
            .ToList();

        var chosen = new List<DetectionInfo>(withSurface);
        foreach (var d in remaining)
        {
            bool near = chosen.Any(c => c.label == d.label &&
                                        (c.worldPos - d.worldPos).sqrMagnitude <= r2);
            if (!near) chosen.Add(d);
        }
        return chosen;
    }


    void OnEnable()
    {
        latestBySource.Clear();

        if (carDetectors != null)
        {
            foreach (var d in carDetectors)
            {
                if (!d) continue;
                d.OnDetections += (list) => latestBySource[d] = list ?? new List<DetectionInfo>();
                latestBySource[d] = d.GetLatestDetections() ?? new List<DetectionInfo>();
            }
        }
        if (propDetectors != null)
        {
            foreach (var d in propDetectors)
            {
                if (!d) continue;
                d.OnDetections += (list) => latestBySource[d] = list ?? new List<DetectionInfo>();
                latestBySource[d] = d.GetLatestDetections() ?? new List<DetectionInfo>();
            }
        }
    }

    void Update()
    {
        if (logEveryNFrames > 0 && Time.frameCount % logEveryNFrames == 0)
            PrintSummary();

        if (logOnKey != KeyCode.None && Input.GetKeyDown(logOnKey))
            PrintSummary();
    }

    void PrintSummary()
    {
        var merged = latestBySource.Values
            .Where(v => v != null)
            .SelectMany(v => v);

        // Deduplicate across cameras: same label + same surface, or same label within ~0.75m
        var deduped = DedupByLabelAndProximity(merged, 0.75f).ToList();

        string msg;
        if (deduped.Count == 0)
        {
            msg = "[Vision] none";
        }
        else
        {
            var parts = deduped
                .GroupBy(d => d.label)
                .OrderByDescending(g => g.Count())
                .Select(g => $"{g.Key}:{g.Count()}")
                .ToArray();

            msg = "[Vision] " + string.Join(", ", parts);
        }

        if (!onlyOnChange || msg != lastPrinted)
        {
            Debug.Log(msg);
            lastPrinted = msg;
        }
    }

}
