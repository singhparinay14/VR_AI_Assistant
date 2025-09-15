using UnityEngine;
using Unity.Sentis;
using System;
using System.Collections.Generic;
using System.Linq;

/// <summary>
/// Holds YOLO detection info
/// </summary>
public struct DetectionInfo
{
    public string label;
    public Rect bbox;
    public string colour;
    public Vector3 worldPos;
    public float distance;
    public string relDir;
    public string surface;
    public float confidence;
}

public class YoloObjectDetector : MonoBehaviour
{
    [Header("Model & Vision")]
    public ModelAsset modelAsset;
    public Camera[] visionCameras;             // Multiple cameras supported
    public int inputWidth = 640;
    public int inputHeight = 640;
    [Range(0f, 1f)] public float confidenceThreshold = 0.5f;

    // Persistent RTs matching visionCameras (assigned/created at runtime)
    [Tooltip("Optional: assign per-camera RTs here; otherwise they are created in Awake().")]
    public RenderTexture[] visionRTs;

    private Worker worker;
    private Model model;
    private Texture2D readTex;

    public event Action<List<DetectionInfo>> OnDetections;

    // === Your fine-tuned labels ===
    private readonly string[] carLabels = {
        "Ford Mustang GT Convertible 2020",
        "Audi R8 2014",
        "Audi RS6 Avant 2020",
        "BMW X5 2015",
        "Ferrari F8 Tributo 2020",
        "Ferrari F40",
        "Lamborghini Gallardo 2010",
        "Porsche 911 Turbo S 2021",
        "Mercedes AMG GT 2019",
        "Lamborghini Aventador SVJ"
    };

    private List<DetectionInfo> latestDetections = new();

    void Awake()
    {
        if (visionCameras == null) visionCameras = Array.Empty<Camera>();
        if (visionRTs == null || visionRTs.Length != visionCameras.Length)
            visionRTs = new RenderTexture[visionCameras.Length];

        // Create/bind one persistent RT per camera and assign once
        for (int i = 0; i < visionCameras.Length; i++)
        {
            var cam = visionCameras[i];
            if (!cam) continue;

            if (!visionRTs[i])
            {
                var rt = new RenderTexture(inputWidth, inputHeight, 24, RenderTextureFormat.ARGB32);
                rt.name = $"VisionRT_{i}";
                rt.Create();
                visionRTs[i] = rt;
            }

            cam.targetTexture = visionRTs[i];   // bind permanently
        }
    }

    void Start()
    {
        model = ModelLoader.Load(modelAsset);
        worker = new Worker(model, BackendType.GPUCompute);
        readTex = new Texture2D(inputWidth, inputHeight, TextureFormat.RGB24, false);

        UnityEngine.Debug.Log($"YoloObjectDetector initialized with {visionCameras.Length} cameras.");
    }

    void Update()
    {
        if (visionCameras == null || visionCameras.Length == 0) return;

        List<DetectionInfo> combinedDetections = new();

        for (int i = 0; i < visionCameras.Length; i++)
        {
            Camera cam = visionCameras[i];
            RenderTexture rt = (visionRTs != null && i < visionRTs.Length) ? visionRTs[i] : null;
            if (!cam || !rt) continue;

            // If the cameras are not auto-rendering, you can force-render:
            // cam.Render();

            // Read pixels from the persistent RT
            var prev = RenderTexture.active;
            RenderTexture.active = rt;
            readTex.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0);
            readTex.Apply();
            RenderTexture.active = prev;

            var input = new Tensor<float>(new TensorShape(1, 3, inputHeight, inputWidth));
            TextureConverter.ToTensor(readTex, input, new TextureTransform());
            worker.Schedule(input);
            input.Dispose();

            // Parse detections for this camera
            List<DetectionInfo> detections = ParseDetections(worker, cam);
            combinedDetections.AddRange(detections);
        }

        latestDetections = combinedDetections;

        if (latestDetections.Count > 0)
        {
            OnDetections?.Invoke(latestDetections);
        }
    }

    // IoU (Intersection over Union) helper
    private float IoU(Rect a, Rect b)
    {
        float interX = Mathf.Max(a.xMin, b.xMin);
        float interY = Mathf.Max(a.yMin, b.yMin);
        float interW = Mathf.Min(a.xMax, b.xMax) - interX;
        float interH = Mathf.Min(a.yMax, b.yMax) - interY;

        if (interW <= 0 || interH <= 0) return 0f;

        float intersection = interW * interH;
        float union = a.width * a.height + b.width * b.height - intersection;
        return intersection / union;
    }

    // Non-Maximum Suppression
    private List<DetectionInfo> ApplyNMS(List<DetectionInfo> detections, float iouThreshold = 0.45f)
    {
        var results = new List<DetectionInfo>();

        // Sort by confidence (highest first)
        var sorted = detections.OrderByDescending(d => d.confidence).ToList();

        while (sorted.Count > 0)
        {
            var best = sorted[0];
            results.Add(best);
            sorted.RemoveAt(0);

            // Remove overlapping boxes of the same class
            sorted = sorted.Where(d =>
                !(d.label == best.label && IoU(d.bbox, best.bbox) > iouThreshold)
            ).ToList();
        }

        return results;
    }

    private List<DetectionInfo> ParseDetections(Worker worker, Camera cam)
    {
        List<DetectionInfo> detections = new();

        var output = worker.PeekOutput() as Tensor<float>;
        if (output == null) return detections;

        using var cpuOutput = output.ReadbackAndClone();

        int numAttrs = cpuOutput.shape[1];   // 14
        int numBoxes = cpuOutput.shape[2];   // 8400

        for (int i = 0; i < numBoxes; i++)
        {
            float cx = cpuOutput[0, 0, i];
            float cy = cpuOutput[0, 1, i];
            float w = cpuOutput[0, 2, i];
            float h = cpuOutput[0, 3, i];

            // apply sigmoid to objectness
            float obj = Sigmoid(cpuOutput[0, 4, i]);

            // find best class
            int bestClass = -1;
            float bestScore = 0f;
            for (int c = 5; c < numAttrs; c++)
            {
                float score = Sigmoid(cpuOutput[0, c, i]);
                if (score > bestScore)
                {
                    bestScore = score;
                    bestClass = c - 5;
                }
            }

            // final confidence
            float confidence = obj * bestScore;
            confidence = Mathf.Sqrt(confidence);
            if (confidence < confidenceThreshold) continue;

            if (bestClass < 0 || bestClass >= carLabels.Length) continue;

            string label = carLabels[bestClass];

            // scale bbox to pixel space
            float x = (cx - w / 2f) / inputWidth * cam.pixelWidth;
            float y = (cy - h / 2f) / inputHeight * cam.pixelHeight;
            float bw = w / inputWidth * cam.pixelWidth;
            float bh = h / inputHeight * cam.pixelHeight;

            Rect bbox = new Rect(x, y, bw, bh);

            // raycast from bbox center
            Vector3 screenPoint = new Vector3(x + bw / 2f, y + bh / 2f, cam.nearClipPlane + 1f);
            Ray ray = cam.ScreenPointToRay(screenPoint);
            if (Physics.Raycast(ray, out RaycastHit hit))
            {
                var det = new DetectionInfo
                {
                    label = label,
                    bbox = bbox,
                    colour = SampleColour(bbox, cam),
                    worldPos = hit.point,
                    distance = hit.distance,
                    relDir = GetRelativeDirection(cam, hit.point),
                    surface = hit.collider.gameObject.name,
                    confidence = confidence
                };

                detections.Add(det);

                UnityEngine.Debug.Log($"[YOLO] Detected {det.label} conf={det.confidence:F2} color={det.colour} at {det.worldPos}");

                // Debug visualization
                Vector3 debugPos = det.worldPos + Vector3.up * 2f;
                Color debugCol = Color.gray;
                if (ColorUtility.TryParseHtmlString(det.colour, out Color parsed))
                    debugCol = parsed;

                UnityEngine.Debug.DrawLine(det.worldPos, debugPos, debugCol, 2f);
            }
        }

        return ApplyNMS(detections, 0.6f);
    }

    // sigmoid helper
    private float Sigmoid(float x)
    {
        return 1f / (1f + Mathf.Exp(-x));
    }

    private string SampleColour(Rect bbox, Camera cam)
    {
        // Clamp bbox inside texture
        int x = Mathf.Clamp((int)bbox.x, 0, readTex.width - 1);
        int y = Mathf.Clamp((int)bbox.y, 0, readTex.height - 1);
        int w = Mathf.Clamp((int)bbox.width, 1, readTex.width - x);
        int h = Mathf.Clamp((int)bbox.height, 1, readTex.height - y);

        // Focus on the central/top region of the car (hood/roof area)
        int cropX = x + w / 4;                // middle section
        int cropW = w / 2;
        int cropY = y + h / 4;                // ignore bottom (floor/reflections)
        int cropH = h / 3;                    // only take ~⅓ height

        Color[] pixels = readTex.GetPixels(cropX, cropY, cropW, cropH);
        if (pixels.Length == 0) return "unknown";

        float sumH = 0f, sumS = 0f, sumV = 0f;
        int count = 0;

        foreach (Color c in pixels)
        {
            Color.RGBToHSV(c, out float hue, out float sat, out float val);

            // Filter out dull or extreme pixels (background, glass, shadows, bright reflections)
            if (sat < 0.25f) continue;
            if (val < 0.25f || val > 0.9f) continue;

            // Optional: ignore floor/wall tones (cyan-ish / blue-ish range)
            if (hue > 0.5f && hue < 0.65f) continue;

            sumH += hue;
            sumS += sat;
            sumV += val;
            count++;
        }

        if (count == 0) return "gray"; // fallback

        float avgH = sumH / count;
        float avgS = sumS / count;
        float avgV = sumV / count;

        return GetClosestColorName(avgH, avgS, avgV);
    }

    private string GetClosestColorName(float hue, float sat, float val)
    {
        if (val < 0.2f) return "black";
        if (val > 0.9f && sat < 0.2f) return "white";
        if (sat < 0.25f) return "gray";

        if (hue < 0.05f || hue > 0.95f) return "red";
        if (hue < 0.15f) return "orange";
        if (hue < 0.25f) return "yellow";
        if (hue < 0.45f) return "green";
        if (hue < 0.60f) return "cyan";
        if (hue < 0.75f) return "blue";
        if (hue < 0.90f) return "purple";

        return "gray";
    }

    private string GetRelativeDirection(Camera cam, Vector3 worldPos)
    {
        Vector3 local = cam.transform.InverseTransformPoint(worldPos);
        if (local.x < -0.2f) return "left";
        if (local.x > 0.2f) return "right";
        return "center";
    }

    public bool HasLabel(string word)
    {
        foreach (var label in carLabels)
        {
            if (label.ToLower().Contains(word.ToLower()))
                return true;
        }
        return false;
    }

    public List<DetectionInfo> GetLatestDetections()
    {
        return latestDetections ?? new List<DetectionInfo>();
    }

    void OnDestroy()
    {
        worker?.Dispose();
        // If we created RTs at runtime, you may optionally release them here.
        // for (int i = 0; i < visionRTs?.Length; i++) if (visionRTs[i]) visionRTs[i].Release();
    }
}
