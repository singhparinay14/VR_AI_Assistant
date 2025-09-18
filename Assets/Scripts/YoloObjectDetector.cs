using System;
using System.Collections.Generic;
using System.Linq;
using Unity.Sentis;
using UnityEngine;
using Debug = UnityEngine.Debug;

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
    public float targetInferenceFPS = 10f;
    public bool staggerCameras = true;
    public int maxDetectionsPerCamera = 5;
    public bool enableColourSampling = false;

    [Header("Debug")]
    public bool showDebugLines = false;
    public bool logDetections = false;
    [Range(0f, 5f)] public float debugLineDuration = 0f;

    [Header("Debug Rays (visualize casts)")]
    [Tooltip("Draw rays from each camera through kept boxes (helps verify ray hits).")]
    public bool drawRays = false;
    [Tooltip("Only draw rays for final (post-NMS) boxes.")]
    public bool drawAfterNMS = true;
    public int maxRaysDrawn = 20;
    public float rayDuration = 0.1f;
    public Color rayHitColor = Color.cyan;
    public Color rayMissColor = Color.red;
    public Color hitMarkerColor = Color.yellow;

    [Header("Raycast")]
    [Tooltip("Flip Y when mapping bbox center to ray (YOLO top-left -> Unity bottom-left).")]
    public bool invertYForRay = true;
    [Tooltip("Layers the bbox-center ray can hit.")]
    public LayerMask hitMask = ~0;

    [Header("Model & Vision")]
    public ModelAsset modelAsset;
    public Camera[] visionCameras;
    public int inputWidth = 640;
    public int inputHeight = 640;
    [Range(0f, 1f)] public float confidenceThreshold = 0.5f;

    [Header("Aggregation")]
    [Tooltip("How long (seconds) to keep each camera's last results when building the union.")]
    public float cacheTTL = 0.6f;
    private List<DetectionInfo>[] _camCache;
    private float[] _camCacheTime;

    [Tooltip("Optional: assign per-camera RTs here; otherwise they are created in Awake().")]
    public RenderTexture[] visionRTs;

    private Worker worker;
    private Model model;
    private readonly TextureTransform _texTransform = new TextureTransform();
    private float _nextTick = 0f;
    private int _frameCounter = 0;
    private Texture2D readTex;

    public event Action<List<DetectionInfo>> OnDetections;

    // Your fine-tuned labels (keep order aligned with the model export)
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
                visionRTs[i] = rt; // IMPORTANT
            }

            cam.targetTexture = visionRTs[i];
            // camera perf hints
            cam.allowHDR = false;
            cam.allowMSAA = false;
            cam.clearFlags = CameraClearFlags.SolidColor;
            cam.backgroundColor = Color.black;
        }

        // init per-camera caches
        _camCache = new List<DetectionInfo>[visionCameras.Length];
        _camCacheTime = new float[visionCameras.Length];
        for (int k = 0; k < visionCameras.Length; k++)
        {
            _camCache[k] = new List<DetectionInfo>();
            _camCacheTime[k] = -999f;
        }
    }

    void Start()
    {
        model = ModelLoader.Load(modelAsset);
        worker = new Worker(model, BackendType.GPUCompute);

        if (enableColourSampling)
            readTex = new Texture2D(inputWidth, inputHeight, TextureFormat.RGB24, false);

        Debug.Log($"YoloObjectDetector initialized with {visionCameras.Length} cameras.");
    }

    void Update()
    {
        if (visionCameras == null || visionCameras.Length == 0) return;

        // throttle per-detector
        if (Time.time < _nextTick) return;
        _nextTick = Time.time + 1f / Mathf.Max(1f, targetInferenceFPS);

        // stagger: process one camera per tick to avoid spikes
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

            // feed RT directly (no CPU ReadPixels for inference)
            var input = new Tensor<float>(new TensorShape(1, 3, inputHeight, inputWidth));
            TextureConverter.ToTensor(rt, input, _texTransform);
            worker.Schedule(input);
            input.Dispose();

            // Only if we need colour sampling: refresh readTex once for this camera
            if (enableColourSampling && readTex != null)
            {
                var prev = RenderTexture.active;
                RenderTexture.active = rt;
                readTex.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0);
                readTex.Apply();
                RenderTexture.active = prev;
            }

            // parse and cache this camera's detections
            var detections = ParseDetections(worker, cam);

            if (maxDetectionsPerCamera > 0)
                detections = detections.OrderByDescending(d => d.confidence)
                                       .Take(maxDetectionsPerCamera).ToList();

            _camCache[i] = detections;
            _camCacheTime[i] = Time.time;
        }

        // Build the union from recent per-camera caches (cacheTTL)
        var combinedDetections = new List<DetectionInfo>(32);
        for (int k = 0; k < visionCameras.Length; k++)
            if (Time.time - _camCacheTime[k] <= cacheTTL)
                combinedDetections.AddRange(_camCache[k]);

        latestDetections = combinedDetections;
        OnDetections?.Invoke(latestDetections); // always publish, even if empty
    }

    private List<DetectionInfo> ParseDetections(Worker worker, Camera cam)
    {
        int raycastsDone = 0;
        int raycastCap = (maxDetectionsPerCamera > 0) ? maxDetectionsPerCamera : int.MaxValue;

        var detections = new List<DetectionInfo>(16);

        var output = worker.PeekOutput() as Tensor<float>;
        if (output == null) return detections;

        using var cpu = output.ReadbackAndClone();

        int A = cpu.shape[1];
        int B = cpu.shape[2];
        bool channelsFirst = (A <= B);
        int numBoxes = channelsFirst ? B : A;
        int numAttrs = channelsFirst ? A : B;

        int classes = carLabels.Length;
        bool hasObj = (numAttrs == (5 + classes));
        int clsStart = hasObj ? 5 : 4;

        for (int i = 0; i < numBoxes; i++)
        {
            float cx, cy, w, h, obj = 1f;

            if (channelsFirst)
            {
                cx = cpu[0, 0, i]; cy = cpu[0, 1, i];
                w = cpu[0, 2, i]; h = cpu[0, 3, i];
                if (hasObj) obj = Sigmoid(cpu[0, 4, i]);
            }
            else
            {
                cx = cpu[0, i, 0]; cy = cpu[0, i, 1];
                w = cpu[0, i, 2]; h = cpu[0, i, 3];
                if (hasObj) obj = Sigmoid(cpu[0, i, 4]);
            }

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

            // model space -> camera pixel space
            float x = (cx - w * 0.5f) / inputWidth * cam.pixelWidth;
            float y = (cy - h * 0.5f) / inputHeight * cam.pixelHeight;
            float bw = w / inputWidth * cam.pixelWidth;
            float bh = h / inputHeight * cam.pixelHeight;
            Rect bbox = new Rect(x, y, bw, bh);

            if (raycastsDone >= raycastCap) continue;

            // build a viewport ray from the bbox center (optionally flip Y)
            float vx = (x + bw * 0.5f) / cam.pixelWidth;
            float vy = (y + bh * 0.5f) / cam.pixelHeight;
            if (invertYForRay) vy = 1f - vy;

            Ray ray = cam.ViewportPointToRay(new Vector3(vx, vy, 0f));
            float castLen = GetCastLength(cam);
            bool hitOk = Physics.Raycast(ray, out RaycastHit hit, castLen, hitMask);
            raycastsDone++;

            var det = new DetectionInfo
            {
                label = label,
                bbox = bbox,
                colour = enableColourSampling ? SampleColour(bbox, cam) : "gray",
                worldPos = hitOk ? hit.point : cam.transform.position + cam.transform.forward * 2f,
                distance = hitOk ? hit.distance : -1f,
                relDir = GetRelativeDirection(cam, hitOk ? hit.point : cam.transform.position + cam.transform.forward * 2f),
                surface = hitOk ? hit.collider.gameObject.name : "nohit",
                confidence = confidence
            };

            detections.Add(det);

            if (logDetections)
                Debug.Log($"[YOLO-cars] {det.label} {det.confidence:0.00} hit={hitOk} @ {det.worldPos}");
        }

        // NMS on bbox (per label)
        var kept = ApplyNMS(detections, 0.6f);

#if UNITY_EDITOR
        if (drawRays)
        {
            int drawn = 0;
            foreach (var d in kept.OrderByDescending(k => k.confidence))
            {
                if (drawn >= Mathf.Max(1, maxRaysDrawn)) break;

                // rebuild the ray from bbox center for visualization
                float vx = (d.bbox.x + d.bbox.width * 0.5f) / cam.pixelWidth;
                float vy = (d.bbox.y + d.bbox.height * 0.5f) / cam.pixelHeight;
                if (invertYForRay) vy = 1f - vy;

                var r2 = cam.ViewportPointToRay(new Vector3(vx, vy, 0f));
                // draw short segment; the hit line was drawn in Update if needed
                Debug.DrawRay(r2.origin, r2.direction * 3f, rayHitColor, rayDuration);

                // small cross at the stored worldPos (if we had a hit)
                if (d.distance >= 0f)
                {
                    Debug.DrawLine(d.worldPos + Vector3.up * 0.05f, d.worldPos - Vector3.up * 0.05f, hitMarkerColor, rayDuration);
                    Debug.DrawLine(d.worldPos + Vector3.right * 0.05f, d.worldPos - Vector3.right * 0.05f, hitMarkerColor, rayDuration);
                    Debug.DrawLine(d.worldPos + Vector3.forward * 0.05f, d.worldPos - Vector3.forward * 0.05f, hitMarkerColor, rayDuration);
                }

                drawn++;
            }
        }
#endif

        return kept;
    }

    private float GetCastLength(Camera cam)
    {
        // Cap the ray length so we don't hit far-away skybox or stray colliders
        return Mathf.Min(50f, Mathf.Max(2f, cam.farClipPlane));
    }

    private float Sigmoid(float x) => 1f / (1f + Mathf.Exp(-x));

    private string SampleColour(Rect bbox, Camera cam)
    {
        if (readTex == null) return "gray";

        int x = Mathf.Clamp((int)bbox.x, 0, readTex.width - 1);
        int y = Mathf.Clamp((int)bbox.y, 0, readTex.height - 1);
        int w = Mathf.Clamp((int)bbox.width, 1, readTex.width - x);
        int h = Mathf.Clamp((int)bbox.height, 1, readTex.height - y);

        int cropX = x + w / 4;
        int cropW = w / 2;
        int cropY = y + h / 4;
        int cropH = h / 3;

        Color[] pixels = readTex.GetPixels(cropX, cropY, cropW, cropH);
        if (pixels.Length == 0) return "unknown";

        float sumH = 0f, sumS = 0f, sumV = 0f; int count = 0;
        foreach (Color c in pixels)
        {
            Color.RGBToHSV(c, out float hue, out float sat, out float val);
            if (sat < 0.25f) continue;
            if (val < 0.25f || val > 0.9f) continue;
            if (hue > 0.5f && hue < 0.65f) continue; // optional cyan exclusion
            sumH += hue; sumS += sat; sumV += val; count++;
        }
        if (count == 0) return "gray";

        float avgH = sumH / count, avgS = sumS / count, avgV = sumV / count;
        return GetClosestColorName(avgH, avgS, avgV);
    }

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

    private List<DetectionInfo> ApplyNMS(List<DetectionInfo> detections, float iouThreshold = 0.45f)
    {
        var results = new List<DetectionInfo>();
        // NMS per label to avoid cross-class suppression
        var byLabel = detections.GroupBy(d => d.label);
        foreach (var grp in byLabel)
        {
            var list = grp.OrderByDescending(d => d.confidence).ToList();
            while (list.Count > 0)
            {
                var best = list[0];
                results.Add(best);
                list.RemoveAt(0);
                list.RemoveAll(d => IoU(best.bbox, d.bbox) > iouThreshold);
            }
        }
        return results;
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
        => carLabels.Any(l => l.ToLower().Contains(word.ToLower()));

    public List<DetectionInfo> GetLatestDetections()
        => latestDetections ?? new List<DetectionInfo>();

    void OnDestroy() => worker?.Dispose();
}
