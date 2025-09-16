using UnityEngine;
using Unity.Sentis;
using System;
using System.Collections.Generic;
using System.Linq;
using Debug = UnityEngine.Debug; // resolve Debug ambiguity
using UnityEngine.Experimental.Rendering;      // for GraphicsFormat


public class PropsDetector : MonoBehaviour
{
    [Header("Debug Viz")]
    public bool drawRays = true;
    public bool drawAfterNMS = true;     // draw rays only for final kept boxes
    public int maxRaysDrawn = 20;        // cap per frame
    public float rayLength = 20f;
    public float rayDuration = 0.15f;
    public Color rayHitColor = Color.green;
    public Color rayMissColor = Color.red;
    public Color hitMarkerColor = Color.yellow;
    public LayerMask hitMask = ~0;       // everything
    public string[] debugOnlyTheseLabels; // leave empty to draw all kept labels


    [Header("Model & Vision")]
    public ModelAsset modelAsset;          // drag yolov8n (Model Asset) here
    public Camera[] visionCameras;         // one or more cameras
    public int inputWidth = 640;
    public int inputHeight = 640;
    [Range(0f, 1f)] public float confidenceThreshold = 0.25f;

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
    private Texture2D readTex;

    public event Action<List<DetectionInfo>> OnDetections;

    // COCO-80 labels (Ultralytics order)
    private readonly string[] cocoLabels = new string[] {
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

        // Create/bind one persistent RT per camera and assign once (URP RG needs depth)
        for (int i = 0; i < visionCameras.Length; i++)
        {
            var cam = visionCameras[i];
            if (!cam) continue;

            if (!visionRTs[i])
            {
                var desc = new RenderTextureDescriptor(inputWidth, inputHeight)
                {
                    graphicsFormat = GraphicsFormat.R8G8B8A8_UNorm,
                    depthStencilFormat = GraphicsFormat.D24_UNorm_S8_UInt, // required by URP Render Graph
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
            cam.targetTexture = visionRTs[i];   // bind permanently
        }
    }

    void Start()
    {
        if (modelAsset == null)
        {
            Debug.LogError("PropsDetector: assign modelAsset (yolov8n Model Asset).");
            enabled = false; return;
        }

        // Resolve allowlist indices
        allowSet = (allowedLabels == null || allowedLabels.Length == 0)
            ? null
            : new HashSet<int>(allowedLabels.Select(n => Array.IndexOf(cocoLabels, n)).Where(idx => idx >= 0));

        model = ModelLoader.Load(modelAsset);
        worker = new Worker(model, BackendType.GPUCompute);
        readTex = new Texture2D(inputWidth, inputHeight, TextureFormat.RGB24, false);

        Debug.Log($"PropsDetector initialized with {visionCameras.Length} camera(s).");
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

            if (rt.width != inputWidth || rt.height != inputHeight)
            {
                Debug.LogWarning($"PropsDetector: RT {rt.name} is {rt.width}x{rt.height}, expected {inputWidth}x{inputHeight}.");
                continue;
            }

            // Read pixels from RT
            var prev = RenderTexture.active;
            RenderTexture.active = rt;
            readTex.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0);
            readTex.Apply();
            RenderTexture.active = prev;

            // Build input tensor and run
            var input = new Tensor<float>(new TensorShape(1, 3, inputHeight, inputWidth));
            TextureConverter.ToTensor(readTex, input, new TextureTransform()); // matches your car detector
            worker.Schedule(input);
            input.Dispose();

            // Parse detections for this camera
            combinedDetections.AddRange(ParseDetectionsProps(worker, cam));
        }

        latestDetections = combinedDetections;

        if (latestDetections.Count > 0)
            OnDetections?.Invoke(latestDetections);

        // === Print a compact count summary every N frames ===
        if (Time.frameCount % Mathf.Max(1, printEveryNFrames) == 0)
        {
            var counts = latestDetections
                .GroupBy(d => d.label)
                .Select(g => $"{g.Key}:{g.Count()}")
                .ToArray();
            if (counts.Length == 0) Debug.Log("[YOLO-props] none");
            else Debug.Log("[YOLO-props] " + string.Join(", ", counts));
        }
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
            // --- box ---
            float cx, cy, w, h;
            if (channelsFirst) { cx = cpu[0, 0, i]; cy = cpu[0, 1, i]; w = cpu[0, 2, i]; h = cpu[0, 3, i]; }
            else { cx = cpu[0, i, 0]; cy = cpu[0, i, 1]; w = cpu[0, i, 2]; h = cpu[0, i, 3]; }

            // --- best class (Ultralytics v8: class probs; objectness already folded) ---
            int bestClass = -1; float bestScore = 0f;
            for (int c = 4; c < numAttrs; c++)
            {
                float s = channelsFirst ? cpu[0, c, i] : cpu[0, i, c];
                s = Sigmoid(s); // safe
                if (s > bestScore) { bestScore = s; bestClass = c - 4; }
            }
            if (bestClass < 0) continue;

            // Apply allowlist if set
            if (allowSet != null && !allowSet.Contains(bestClass)) continue;

            // Confidence gate
            float conf = bestScore;
            if (conf < confidenceThreshold) continue;

            // Size filters (on model input pixels)
            if (h < minBoxHeightPx) continue;
            if ((w * h) < minBoxAreaPx) continue;

            candidates.Add((bestClass, conf, cx, cy, w, h));
        }

        // Keep top-K to avoid NMS overload
        const int TOPK = 300;
        if (candidates.Count > TOPK)
            candidates = candidates.OrderByDescending(c => c.conf).Take(TOPK).ToList();

        // Convert to screen space + (optional) raycast
        foreach (var c in candidates)
        {
            string label = cocoLabels[c.cls];

            float x = (c.cx - c.w / 2f) / inputWidth * cam.pixelWidth;
            float y = (c.cy - c.h / 2f) / inputHeight * cam.pixelHeight;
            float bw = c.w / inputWidth * cam.pixelWidth;
            float bh = c.h / inputHeight * cam.pixelHeight;
            Rect bbox = new Rect(x, y, bw, bh);

            Vector3 screenPoint = new Vector3(x + bw / 2f, y + bh / 2f, cam.nearClipPlane + 1f);
            Ray ray = cam.ScreenPointToRay(screenPoint);
            bool hitOk = Physics.Raycast(ray, out RaycastHit hit, rayLength, hitMask);

            //if (drawRays)
            //{
            //    var col = hitOk ? rayHitColor : rayMissColor;
            //    // draw the ray; stop at hit distance if we hit, otherwise full length
            //    Debug.DrawRay(ray.origin, ray.direction * (hitOk ? hit.distance : rayLength), col, rayDuration);
            //    if (hitOk)
            //    {
            //        // little “spark” at the hit point
            //        Debug.DrawLine(hit.point + Vector3.up * 0.05f, hit.point - Vector3.up * 0.05f, hitMarkerColor, rayDuration);
            //        Debug.DrawLine(hit.point + Vector3.right * 0.05f, hit.point - Vector3.right * 0.05f, hitMarkerColor, rayDuration);
            //        Debug.DrawLine(hit.point + Vector3.forward * 0.05f, hit.point - Vector3.forward * 0.05f, hitMarkerColor, rayDuration);
            //    }
            //}

            var det = new DetectionInfo
            {
                label = label,
                bbox = bbox,
                colour = SampleColour(bbox),
                worldPos = hitOk ? hit.point : cam.transform.position + cam.transform.forward * 2f,
                distance = hitOk ? hit.distance : -1f,
                relDir = GetRelativeDirection(cam, hitOk ? hit.point : cam.transform.position + cam.transform.forward * 2f),
                surface = hitOk ? hit.collider.gameObject.name : "nohit",
                confidence = c.conf
            };
            detections.Add(det);


            if (verbose)
                Debug.Log($"[YOLO-props] {det.label} conf={det.confidence:0.00} box=({(int)bbox.x},{(int)bbox.y},{(int)bbox.width},{(int)bbox.height}) hit={hitOk}");
        }

        var kept = ApplyNMS(detections, 0.45f);

        // --- Debug rays for kept boxes (clean & capped) ---
        if (drawRays && drawAfterNMS)
        {
            int drawn = 0;
            // optional label filter
            HashSet<string> allowDbg = (debugOnlyTheseLabels != null && debugOnlyTheseLabels.Length > 0)
                ? new HashSet<string>(debugOnlyTheseLabels)
                : null;

            foreach (var d in kept.OrderByDescending(k => k.confidence))
            {
                if (drawn >= Mathf.Max(1, maxRaysDrawn)) break;
                if (allowDbg != null && !allowDbg.Contains(d.label)) continue;

                // Center of bbox -> screen point -> ray
                Vector3 screenPoint = new Vector3(
                    d.bbox.x + d.bbox.width * 0.5f,
                    d.bbox.y + d.bbox.height * 0.5f,
                    cam.nearClipPlane + 1f
                );
                Ray ray = cam.ScreenPointToRay(screenPoint);

                bool hitOk = Physics.Raycast(ray, out RaycastHit hit, rayLength, hitMask);
                var col = hitOk ? rayHitColor : rayMissColor;
                Debug.DrawRay(ray.origin, ray.direction * (hitOk ? hit.distance : rayLength), col, rayDuration);

                if (hitOk)
                {
                    Debug.DrawLine(hit.point + Vector3.up * 0.05f, hit.point - Vector3.up * 0.05f, hitMarkerColor, rayDuration);
                    Debug.DrawLine(hit.point + Vector3.right * 0.05f, hit.point - Vector3.right * 0.05f, hitMarkerColor, rayDuration);
                    Debug.DrawLine(hit.point + Vector3.forward * 0.05f, hit.point - Vector3.forward * 0.05f, hitMarkerColor, rayDuration);
                }

                drawn++;
            }
        }

        return kept;

    }

    // === Helpers matching your style ===

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
