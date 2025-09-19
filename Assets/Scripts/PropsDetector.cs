using UnityEngine;
using Unity.Sentis;
using System;
using System.Collections.Generic;
using System.Linq;
using Debug = UnityEngine.Debug; // resolve Debug ambiguity
using UnityEngine.Experimental.Rendering; // GraphicsFormat

public class PropsDetector : MonoBehaviour
{
    [Header("Performance")]
    [Tooltip("Max inferences per second for THIS detector (across its cameras).")]
    public float targetInferenceFPS = 10f;
    [Tooltip("Spread cameras across ticks to avoid spikes.")]
    public bool staggerCameras = true;
    [Tooltip("After filtering, cap detections per camera to reduce raycasts. 0 = unlimited.")]
    public int maxDetectionsPerCamera = 5;
    [Tooltip("Compute color from pixels (heavy). Turn OFF unless needed.")]
    public bool enableColourSampling = false;

    [Header("Debug Viz")]
    public bool drawRays = true;
    public bool drawAfterNMS = true;     // draw rays only for final kept boxes
    public int maxRaysDrawn = 20;        // cap per frame
    public float rayDuration = 0.15f;
    public Color rayHitColor = Color.green;
    public Color rayMissColor = Color.red;
    public Color hitMarkerColor = Color.yellow;
    public LayerMask hitMask = ~0;       // everything
    public string[] debugOnlyTheseLabels; // leave empty to draw all kept labels

    [Header("Raycast Mapping")]
    public bool invertYForRay = true;         // flip Y (YOLO top-left -> Unity bottom-left)
    public bool useCameraRangeForRays = true; // automatic: farClip - nearClip
    public float rayLength = 100f;            // fallback manual length if auto is off

    [Header("Model & Vision")]
    public ModelAsset modelAsset;          // drag yolov8n (Model Asset) here
    public Camera[] visionCameras;         // one or more cameras
    public int inputWidth = 640;
    public int inputHeight = 640;
    [Range(0f, 1f)] public float confidenceThreshold = 0.25f;

    // === Cache union across cameras ===
    [Header("Aggregation")]
    [Tooltip("How long (seconds) to keep each camera's last results when building the union.")]
    public float cacheTTL = 0.6f;

    private List<DetectionInfo>[] _camCache;
    private float[] _camCacheTime;

    [Tooltip("Optional: assign per-camera RTs here; otherwise they are created in Awake().")]
    public RenderTexture[] visionRTs;

    [Header("Filtering")]
    [Tooltip("Only these labels are kept (COCO-80). Leave empty to allow all.")]
    public string[] allowedLabels = new string[] {
        "bench","chair","couch","potted plant","dining table","tv","clock","bicycle","laptop","bottle"
    };
    [Tooltip("Discard boxes smaller than this pixel area on the 640x640 input.")]
    public int minBoxAreaPx = 30 * 30;       // 900 px = ~30x30
    [Tooltip("Discard boxes shorter than this many pixels.")]
    public int minBoxHeightPx = 18;

    [Header("Logging")]
    public bool verbose = false;             // per-detection logs
    public int printEveryNFrames = 10;       // counts summary cadence

    private Worker worker;
    private Model model;

    // Only used if enableColourSampling == true
    private Texture2D readTex;

    // Runtime labels actually used everywhere in this script
    private string[] cocoLabels;

    // Throttle / stagger timing
    private float _nextTick = 0f;
    private int _frameCounter = 0;

    public bool HasLabel(string word)
    {
        if (string.IsNullOrWhiteSpace(word)) return false;
        string w = word.ToLower();
        // Match whole labels or partials (e.g., "table" matches "dining table")
        return cocoLabels.Any(l => l.ToLower().Contains(w));
    }

    public event Action<List<DetectionInfo>> OnDetections;

    // COCO-80 labels (Ultralytics order)
    [Header("Labels")]
    [SerializeField] private TextAsset cocoLabelsText; // drag your coco_labels_80.txt here

    // Built-in fallback (Ultralytics COCO-80 order)
    private static readonly string[] CocoLabelsFallback = new string[] {
        "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
        "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
        "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
        "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",
        "wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange",
        "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed",
        "dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
        "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
    };

    private HashSet<int> allowSet;           // resolved class indices
    private List<DetectionInfo> latestDetections = new();

    void Awake()
    {
        if (visionCameras == null) visionCameras = Array.Empty<Camera>();
        if (visionRTs == null || visionRTs.Length != visionCameras.Length)
            visionRTs = new RenderTexture[visionCameras.Length];

        // Create/bind one persistent RT per camera and assign once (URP RenderGraph needs depth)
        for (int i = 0; i < visionCameras.Length; i++)
        {
            var cam = visionCameras[i];
            if (!cam) continue;

            if (!visionRTs[i])
            {
                var desc = new RenderTextureDescriptor(inputWidth, inputHeight)
                {
                    graphicsFormat = GraphicsFormat.R8G8B8A8_UNorm,
                    depthStencilFormat = GraphicsFormat.D24_UNorm_S8_UInt,
                    msaaSamples = 1,
                    sRGB = true,
                    mipCount = 1,
                    useMipMap = false,
                    autoGenerateMips = false
                };
                var rt = new RenderTexture(desc) { name = $"PropsVisionRT_{i}" };
                rt.Create();
                visionRTs[i] = rt;
            }
            cam.targetTexture = visionRTs[i]; // bind permanently
            // Camera perf hints
            cam.allowHDR = false;
            cam.allowMSAA = false;
            cam.clearFlags = CameraClearFlags.SolidColor;
            cam.backgroundColor = Color.black;
        }

        // init caches ONCE (outside the camera loop)
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
        if (modelAsset == null)
        {
            Debug.LogError("PropsDetector: assign modelAsset (yolov8n Model Asset).");
            enabled = false; return;
        }

        // Load labels
        if (cocoLabelsText != null)
        {
            var parsed = cocoLabelsText.text
                .Split(new[] { '\n', '\r', ',' }, StringSplitOptions.RemoveEmptyEntries)
                .Select(s => s.Trim())
                .Where(s => !string.IsNullOrEmpty(s))
                .ToArray();

            cocoLabels = (parsed.Length == 80) ? parsed : CocoLabelsFallback;
            if (parsed.Length != 80) Debug.LogWarning($"PropsDetector: labels file has {parsed.Length} entries, expected 80. Using built-in fallback.");
        }
        else cocoLabels = CocoLabelsFallback;

        // Resolve allowlist indices
        allowSet = (allowedLabels == null || allowedLabels.Length == 0)
            ? null
            : new HashSet<int>(allowedLabels.Select(n => Array.IndexOf(cocoLabels, n))
                                            .Where(idx => idx >= 0));

        model = ModelLoader.Load(modelAsset);
        worker = new Worker(model, BackendType.GPUCompute);

        if (enableColourSampling)
            readTex = new Texture2D(inputWidth, inputHeight, TextureFormat.RGB24, false);

        Debug.Log($"PropsDetector initialized with {visionCameras.Length} camera(s).");
    }

    void Update()
    {
        if (visionCameras == null || visionCameras.Length == 0) return;

        // Throttle
        if (Time.time < _nextTick) return;
        _nextTick = Time.time + 1f / Mathf.Max(1f, targetInferenceFPS);

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

            if (rt.width != inputWidth || rt.height != inputHeight)
            {
                Debug.LogWarning($"PropsDetector: RT {rt.name} is {rt.width}x{rt.height}, expected {inputWidth}x{inputHeight}.");
                continue;
            }

            // Feed RT directly into tensor (no CPU ReadPixels here)
            var input = new Tensor<float>(new TensorShape(1, 3, inputHeight, inputWidth));
            TextureConverter.ToTensor(rt, input, new TextureTransform());
            worker.Schedule(input);
            input.Dispose();

            if (enableColourSampling && readTex != null)
            {
                var prev = RenderTexture.active;
                RenderTexture.active = rt;
                readTex.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0);
                readTex.Apply();
                RenderTexture.active = prev;
            }

            // Parse detections for this camera and cache
            var dets = ParseDetectionsProps(worker, cam);
            _camCache[i] = dets;
            _camCacheTime[i] = Time.time;
        }

        // Build union from recent caches
        var combinedDetections = new List<DetectionInfo>(32);
        for (int k = 0; k < visionCameras.Length; k++)
            if (Time.time - _camCacheTime[k] <= cacheTTL)
                combinedDetections.AddRange(_camCache[k]);

        latestDetections = combinedDetections;
        OnDetections?.Invoke(latestDetections);   // always publish, even if empty

        // Optional periodic summary
        if (printEveryNFrames > 0 && Time.frameCount % printEveryNFrames == 0)
        {
            var counts = latestDetections.GroupBy(d => d.label)
                .Select(g => $"{g.Key}:{g.Count()}").ToArray();
            Debug.Log(counts.Length == 0 ? "[YOLO-props] none" : "[YOLO-props] " + string.Join(", ", counts));
        }
    }

    private float GetCastLength(Camera cam)
        => useCameraRangeForRays ? Mathf.Max(0.1f, cam.farClipPlane - cam.nearClipPlane) : rayLength;

    private static bool IsLikelyFloorOrWall(RaycastHit hit)
    {
        if (hit.collider == null) return true;
        var b = hit.collider.bounds;
        return (b.size.y < 0.05f) || (b.size.x > 50f) || (b.size.z > 50f);
    }

    private List<DetectionInfo> ParseDetectionsProps(Worker worker, Camera cam)
    {
        List<DetectionInfo> detections = new();

        var output = worker.PeekOutput() as Tensor<float>;
        if (output == null) return detections;

        using var cpu = output.ReadbackAndClone();

        // YOLOv8n ONNX: (1,84,8400) or (1,8400,84)
        bool channelsFirst = (cpu.shape[1] == 84);
        int numBoxes = channelsFirst ? cpu.shape[2] : cpu.shape[1];
        const int numAttrs = 84; // 4 box + 80 cls

        var candidates = new List<(int cls, float conf, float cx, float cy, float w, float h)>(256);

        for (int i = 0; i < numBoxes; i++)
        {
            // box
            float bx, by, bw, bh;
            if (channelsFirst) { bx = cpu[0, 0, i]; by = cpu[0, 1, i]; bw = cpu[0, 2, i]; bh = cpu[0, 3, i]; }
            else { bx = cpu[0, i, 0]; by = cpu[0, i, 1]; bw = cpu[0, i, 2]; bh = cpu[0, i, 3]; }

            // best class
            int bestClass = -1; float bestScore = 0f;
            for (int c = 4; c < numAttrs; c++)
            {
                float s = channelsFirst ? cpu[0, c, i] : cpu[0, i, c];
                s = Sigmoid(s);
                if (s > bestScore) { bestScore = s; bestClass = c - 4; }
            }
            if (bestClass < 0) continue;

            // Allowlist
            if (allowSet != null && !allowSet.Contains(bestClass)) continue;

            // Confidence + size gates (model-pixel space)
            float conf = bestScore;
            if (conf < confidenceThreshold) continue;
            if (bh < minBoxHeightPx) continue;
            if ((bw * bh) < minBoxAreaPx) continue;

            candidates.Add((bestClass, conf, bx, by, bw, bh));
        }

        // Keep top-K candidates before raycasts
        int TOPK = 300;
        if (maxDetectionsPerCamera > 0) TOPK = Mathf.Min(TOPK, Mathf.Max(1, maxDetectionsPerCamera * 10));
        if (candidates.Count > TOPK)
            candidates = candidates.OrderByDescending(c => c.conf).Take(TOPK).ToList();

        int raycastsDone = 0;
        int raycastCap = (maxDetectionsPerCamera > 0) ? maxDetectionsPerCamera : int.MaxValue;

        foreach (var c in candidates)
        {
            if (raycastsDone >= raycastCap) break;

            string label = cocoLabels[c.cls];

            // model(px) -> camera(px)
            float px = (c.cx - c.w * 0.5f) / inputWidth * cam.pixelWidth;
            float py = (c.cy - c.h * 0.5f) / inputHeight * cam.pixelHeight;
            float pw = c.w / inputWidth * cam.pixelWidth;
            float ph = c.h / inputHeight * cam.pixelHeight;
            Rect bbox = new Rect(px, py, pw, ph);

            // bbox center -> viewport (0..1), with optional Y flip
            float vx = (px + pw * 0.5f) / cam.pixelWidth;
            float vy = (py + ph * 0.5f) / cam.pixelHeight;
            if (invertYForRay) vy = 1f - vy;

            Ray ray = cam.ViewportPointToRay(new Vector3(vx, vy, 0f));
            float castLen = GetCastLength(cam);
            bool hitOk = Physics.Raycast(ray, out RaycastHit hit, castLen, hitMask, QueryTriggerInteraction.Collide);
            raycastsDone++;

            // Robust fallback (avoid floors/walls; try spherecast and side rays)
            float vw = pw / cam.pixelWidth;
            if ((!hitOk || IsLikelyFloorOrWall(hit)) && vw > 0.02f)
            {
                const float radius = 0.15f;
                if (Physics.SphereCast(ray, radius, out RaycastHit sh, castLen, hitMask, QueryTriggerInteraction.Collide) && !IsLikelyFloorOrWall(sh))
                {
                    hit = sh;
                    hitOk = true;
                }
                else
                {
                    float vxL = Mathf.Clamp01(vx - vw * 0.25f);
                    float vxR = Mathf.Clamp01(vx + vw * 0.25f);
                    Ray rL = cam.ViewportPointToRay(new Vector3(vxL, vy, 0f));
                    Ray rR = cam.ViewportPointToRay(new Vector3(vxR, vy, 0f));

                    bool okL = Physics.Raycast(rL, out RaycastHit hL, castLen, hitMask, QueryTriggerInteraction.Collide) && !IsLikelyFloorOrWall(hL);
                    bool okR = Physics.Raycast(rR, out RaycastHit hR, castLen, hitMask, QueryTriggerInteraction.Collide) && !IsLikelyFloorOrWall(hR);

                    if (okL && (!hitOk || hL.distance < hit.distance)) { ray = rL; hit = hL; hitOk = true; }
                    if (okR && (!hitOk || hR.distance < hit.distance)) { ray = rR; hit = hR; hitOk = true; }
                }
            }

            // ✅ Use root to stabilize surface identity
            Transform root = hitOk ? hit.collider.transform.root : null;
            string surfaceName = hitOk ? (root != null ? root.name : hit.collider.name) : "nohit";

            var det = new DetectionInfo
            {
                label = label,
                bbox = bbox,
                colour = enableColourSampling && readTex != null ? SampleColour(bbox) : "gray",
                worldPos = hitOk ? hit.point : cam.transform.position + cam.transform.forward * 2f,
                distance = hitOk ? hit.distance : -1f,
                relDir = GetRelativeDirection(cam, hitOk ? hit.point : cam.transform.position + cam.transform.forward * 2f),
                surface = surfaceName,
                confidence = c.conf
            };
            detections.Add(det);

            if (verbose)
                Debug.Log($"[YOLO-props] {det.label} conf={det.confidence:0.00} hit={(hitOk ? "OK" : "MISS")} surf={det.surface}");
        }

        var kept = ApplyNMS(detections, 0.45f);

#if UNITY_EDITOR
        if (drawRays && drawAfterNMS)
        {
            int drawn = 0;
            HashSet<string> allowDbg = (debugOnlyTheseLabels != null && debugOnlyTheseLabels.Length > 0)
                ? new HashSet<string>(debugOnlyTheseLabels)
                : null;

            foreach (var d in kept.OrderByDescending(k => k.confidence))
            {
                if (drawn >= Mathf.Max(1, maxRaysDrawn)) break;
                if (allowDbg != null && !allowDbg.Contains(d.label)) continue;

                float vx = (d.bbox.x + d.bbox.width * 0.5f) / cam.pixelWidth;
                float vy = (d.bbox.y + d.bbox.height * 0.5f) / cam.pixelHeight;
                if (invertYForRay) vy = 1f - vy;

                Ray ray = cam.ViewportPointToRay(new Vector3(vx, vy, 0f));
                float castLen = GetCastLength(cam);
                bool hitOk = d.distance >= 0f;

                Debug.DrawRay(ray.origin, ray.direction * (hitOk ? d.distance : castLen),
                              hitOk ? rayHitColor : rayMissColor, rayDuration);

                if (hitOk)
                {
                    Debug.DrawLine(d.worldPos + Vector3.up * 0.05f,    d.worldPos - Vector3.up * 0.05f,    hitMarkerColor, rayDuration);
                    Debug.DrawLine(d.worldPos + Vector3.right * 0.05f, d.worldPos - Vector3.right * 0.05f, hitMarkerColor, rayDuration);
                    Debug.DrawLine(d.worldPos + Vector3.forward * 0.05f, d.worldPos - Vector3.forward * 0.05f, hitMarkerColor, rayDuration);
                }

                drawn++;
            }
        }
#endif

        return kept;
    }

    // === Helpers ===
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
        var byLabel = detections.GroupBy(d => d.label);
        foreach (var grp in byLabel)
        {
            var g = grp.OrderByDescending(d => d.confidence).ToList();
            while (g.Count > 0)
            {
                var best = g[0]; g.RemoveAt(0);
                results.Add(best);
                g.RemoveAll(d => IoU(best.bbox, d.bbox) > iouThreshold);
            }
        }
        return results;
    }

    private float Sigmoid(float x) => 1f / (1f + Mathf.Exp(-x));

    private string SampleColour(Rect bbox)
    {
        int x = Mathf.Clamp((int)bbox.x, 0, readTex.width - 1);
        int y = Mathf.Clamp((int)bbox.y, 0, readTex.height - 1);
        int w = Mathf.Clamp((int)bbox.width, 1, readTex.width - x);
        int h = Mathf.Clamp((int)bbox.height, 1, readTex.height - y);

        int cropX = x + w / 4;
        int cropW = Mathf.Max(1, w / 2);
        int cropY = y + h / 4;
        int cropH = Mathf.Max(1, h / 3);

        Color[] pixels = readTex.GetPixels(cropX, cropY, cropW, cropH);
        if (pixels.Length == 0) return "gray";

        float sumH = 0f, sumS = 0f, sumV = 0f; int count = 0;
        foreach (Color c in pixels)
        {
            Color.RGBToHSV(c, out float hue, out float sat, out float val);
            if (sat < 0.25f) continue;
            if (val < 0.25f || val > 0.9f) continue;
            sumH += hue; sumS += sat; sumV += val; count++;
        }
        if (count == 0) return "gray";

        float avgH = sumH / count, avgS = sumS / count, avgV = sumV / count;
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

    public List<DetectionInfo> GetLatestDetections() => latestDetections ?? new List<DetectionInfo>();

    void OnDestroy()
    {
        worker?.Dispose();
    }
}