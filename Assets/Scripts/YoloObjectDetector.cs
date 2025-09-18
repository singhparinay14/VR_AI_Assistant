using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Unity.Sentis;
using UnityEngine;
using Debug = UnityEngine.Debug;

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
    [Header("Performance")]
    [Tooltip("Max inferences per second for THIS detector (across its cameras).")]
    public float targetInferenceFPS = 10f;

    [Tooltip("Spread cameras over frames to avoid spikes.")]
    public bool staggerCameras = true;

    [Tooltip("Limit post-NMS results per camera to reduce raycasts and color sampling.")]
    public int maxDetectionsPerCamera = 5;

    [Tooltip("Compute color names from pixels (heavy). Turn OFF unless you need it right now).")]
    public bool enableColourSampling = false;

    [Header("Debug")]
    public bool showDebugLines = false;     // OFF by default
    public bool logDetections = false;      // OFF by default
    [Range(0f, 5f)] public float debugLineDuration = 0f; // 0 = draw for one frame

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

    // (Optional) reusable transform for TextureConverter
    private readonly TextureTransform _texTransform = new TextureTransform();

    // Throttle / stagger timing
    private float _nextTick = 0f;
    private int _frameCounter = 0;

    // Only used if enableColourSampling == true
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
        "Tesla Cybertruck"
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
                var rt = new RenderTexture(inputWidth, inputHeight, 24, RenderTextureFormat.ARGB32)
                {
                    name = $"VisionRT_{i}",
                    useMipMap = false,
                    antiAliasing = 1
                };
                rt.Create();
                visionRTs[i] = rt;
            }

            cam.targetTexture = visionRTs[i];   // bind permanently
            // Camera-side perf tweaks
            cam.allowHDR = false;
            cam.allowMSAA = false;
            cam.clearFlags = CameraClearFlags.SolidColor;
            cam.backgroundColor = Color.black;
            // Consider culling masks to include only needed layers
        }
    }

    void Start()
    {
        model = ModelLoader.Load(modelAsset);
        worker = new Worker(model, BackendType.GPUCompute);

        // Only allocate CPU texture if color sampling will be used
        if (enableColourSampling)
            readTex = new Texture2D(inputWidth, inputHeight, TextureFormat.RGB24, false);

        UnityEngine.Debug.Log($"YoloObjectDetector initialized with {visionCameras.Length} cameras.");
    }

    void Update()
    {
        if (visionCameras == null || visionCameras.Length == 0) return;

        // Throttle overall detector rate
        if (Time.time < _nextTick) return;
        _nextTick = Time.time + 1f / Mathf.Max(1f, targetInferenceFPS);

        var combinedDetections = new List<DetectionInfo>(16);

        // Stagger: process one camera per tick to avoid spikes
        int startIndex = 0;
        int camerasThisTick = visionCameras.Length;
        if (staggerCameras && visionCameras.Length > 1)
        {
            startIndex = _frameCounter++ % visionCameras.Length;
            camerasThisTick = 1;
        }

        for (int j = 0; j < camerasThisTick; j++)
        {
            int i = (startIndex + j) % visionCameras.Length;

            Camera cam = visionCameras[i];
            RenderTexture rt = (visionRTs != null && i < visionRTs.Length) ? visionRTs[i] : null;
            if (!cam || !rt) continue;

            // Create a short-lived input tensor (Sentis pattern: Schedule + Dispose)
            var input = new Tensor<float>(new TensorShape(1, 3, inputHeight, inputWidth));
            TextureConverter.ToTensor(rt, input, _texTransform);
            worker.Schedule(input);
            input.Dispose();

            // If (and only if) we need color sampling, refresh readTex once for this camera
            if (enableColourSampling && readTex != null)
            {
                var prev = RenderTexture.active;
                RenderTexture.active = rt;
                readTex.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0);
                readTex.Apply();
                RenderTexture.active = prev;
            }

            // Parse detections for this camera
            List<DetectionInfo> detections = ParseDetections(worker, cam);

            // Keep only the top-K (post-NMS) to reduce downstream work
            if (maxDetectionsPerCamera > 0)
                detections = detections
                    .OrderByDescending(d => d.confidence)
                    .Take(maxDetectionsPerCamera)
                    .ToList();

            combinedDetections.AddRange(detections);
        }

        latestDetections = combinedDetections;
        OnDetections?.Invoke(latestDetections);
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
        int raycastsDone = 0;
        int raycastCap = (maxDetectionsPerCamera > 0) ? maxDetectionsPerCamera : int.MaxValue;

        var detections = new List<DetectionInfo>(16);

        var output = worker.PeekOutput() as Tensor<float>;
        if (output == null) return detections;

        using var cpu = output.ReadbackAndClone();

        // Support both layouts:
        //  - channels-first: [1, A, B]  (A=numAttrs, B=numBoxes)
        //  - channels-last : [1, B, A]
        int A = cpu.shape[1];
        int B = cpu.shape[2];
        bool channelsFirst = (A <= B);     // e.g., 15 vs 8400
        int numBoxes = channelsFirst ? B : A;
        int numAttrs = channelsFirst ? A : B;

        // Our car model may be 4 box + [obj?] + numClasses (10)
        // hasObjness if numAttrs == 5 + classes; else assume 4 + classes
        int classes = carLabels.Length;
        bool hasObj = (numAttrs == (5 + classes));
        int clsStart = hasObj ? 5 : 4;

        for (int i = 0; i < numBoxes; i++)
        {
            float cx, cy, w, h, obj = 1f;

            if (channelsFirst)
            {
                cx = cpu[0, 0, i];
                cy = cpu[0, 1, i];
                w = cpu[0, 2, i];
                h = cpu[0, 3, i];
                if (hasObj) obj = Sigmoid(cpu[0, 4, i]);
            }
            else
            {
                cx = cpu[0, i, 0];
                cy = cpu[0, i, 1];
                w = cpu[0, i, 2];
                h = cpu[0, i, 3];
                if (hasObj) obj = Sigmoid(cpu[0, i, 4]);
            }

            // Best class
            int bestClass = -1; float bestScore = 0f;
            for (int c = clsStart; c < numAttrs; c++)
            {
                float s = channelsFirst ? cpu[0, c, i] : cpu[0, i, c];
                s = Sigmoid(s);
                if (s > bestScore) { bestScore = s; bestClass = c - clsStart; }
            }

            float confidence = hasObj ? Mathf.Sqrt(obj * bestScore) : bestScore;
            if (confidence < confidenceThreshold) continue;
            if (bestClass < 0 || bestClass >= classes) continue;

            string label = carLabels[bestClass];

            // Scale bbox to camera pixel space
            float x = (cx - w * 0.5f) / inputWidth * cam.pixelWidth;
            float y = (cy - h * 0.5f) / inputHeight * cam.pixelHeight;
            float bw = w / inputWidth * cam.pixelWidth;
            float bh = h / inputHeight * cam.pixelHeight;

            Rect bbox = new Rect(x, y, bw, bh);

            if (raycastsDone >= raycastCap) continue;

            // Raycast from bbox center
            Vector3 screenPoint = new Vector3(x + bw * 0.5f, y + bh * 0.5f, 0f);
            Ray ray = cam.ScreenPointToRay(screenPoint);
            if (Physics.Raycast(ray, out RaycastHit hit))
            {
                raycastsDone++;

                var det = new DetectionInfo
                {
                    label = label,
                    bbox = bbox,
                    colour = enableColourSampling ? SampleColour(bbox, cam) : "gray",
                    worldPos = hit.point,
                    distance = hit.distance,
                    relDir = GetRelativeDirection(cam, hit.point),
                    surface = hit.collider.gameObject.name,
                    confidence = confidence
                };

                detections.Add(det);

                if (logDetections)
                    Debug.Log($"[YOLO-cars] {det.label} {det.confidence:0.00} @ {det.worldPos}");
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
        // Only called when enableColourSampling == true.
        // We refresh readTex once per camera per tick in Update().
        if (readTex == null) return "gray";

        // Clamp bbox inside texture
        int x = Mathf.Clamp((int)bbox.x, 0, readTex.width - 1);
        int y = Mathf.Clamp((int)bbox.y, 0, readTex.height - 1);
        int w = Mathf.Clamp((int)bbox.width, 1, readTex.width - x);
        int h = Mathf.Clamp((int)bbox.height, 1, readTex.height - y);

        // Focus on central/top region
        int cropX = x + w / 4;
        int cropW = w / 2;
        int cropY = y + h / 4;
        int cropH = h / 3;

        Color[] pixels = readTex.GetPixels(cropX, cropY, cropW, cropH);
        if (pixels.Length == 0) return "unknown";

        float sumH = 0f, sumS = 0f, sumV = 0f;
        int count = 0;

        foreach (Color c in pixels)
        {
            Color.RGBToHSV(c, out float hue, out float sat, out float val);

            // Filter out dull or extreme pixels
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
