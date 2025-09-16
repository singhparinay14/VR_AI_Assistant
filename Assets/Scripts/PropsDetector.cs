using UnityEngine;
using Unity.Sentis;
using System;
using System.Collections.Generic;
using System.Linq;
using Debug = UnityEngine.Debug; 
using UnityEngine.Experimental.Rendering;

/// Props-only detector using YOLOv8n (COCO-80) on Unity 6 + Sentis.
/// Mirrors your YoloObjectDetector style.
public class PropsDetector : MonoBehaviour
{
    [Header("Model & Vision")]
    public ModelAsset modelAsset;          // drag yolov8n (Model Asset) here
    public Camera[] visionCameras;         // one or more cameras
    public int inputWidth = 640;
    public int inputHeight = 640;
    [Range(0f, 1f)] public float confidenceThreshold = 0.25f;

    [Tooltip("Optional: assign per-camera RTs here; otherwise they are created in Awake().")]
    public RenderTexture[] visionRTs;

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
                var desc = new RenderTextureDescriptor(inputWidth, inputHeight)
                {
                    graphicsFormat = GraphicsFormat.R8G8B8A8_UNorm,  // color
                    depthStencilFormat = GraphicsFormat.D24_UNorm_S8_UInt, // <-- REQUIRED for URP RG
                    msaaSamples = 1,
                    sRGB = true,
                    mipCount = 1,
                    useMipMap = false,
                    autoGenerateMips = false
                };

                var rt = new RenderTexture(desc)
                {
                    name = $"PropsVisionRT_{i}"
                };
                rt.Create();

                visionRTs[i] = rt;
            }
            cam.targetTexture = visionRTs[i];
        }
    }

    void Start()
    {
        if (modelAsset == null)
        {
            Debug.LogError("PropsDetector: assign modelAsset (yolov8n Model Asset).");
            enabled = false; return;
        }

        model = ModelLoader.Load(modelAsset);
        worker = new Worker(model, BackendType.GPUCompute); // same as your car detector
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

            // Ensure correct size (YOLOv8 expects 640x640)
            if (rt.width != inputWidth || rt.height != inputHeight)
            {
                Debug.LogWarning($"PropsDetector: RT {rt.name} is {rt.width}x{rt.height}, expected {inputWidth}x{inputHeight}.");
                continue;
            }

            // Read pixels from the persistent RT
            var prev = RenderTexture.active;
            RenderTexture.active = rt;
            readTex.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0);
            readTex.Apply();
            RenderTexture.active = prev;

            // Build input tensor and run
            var input = new Tensor<float>(new TensorShape(1, 3, inputHeight, inputWidth));
            TextureConverter.ToTensor(readTex, input, new TextureTransform()); // no flipY in your Sentis; default works
            worker.Schedule(input);
            input.Dispose();

            // Parse detections for this camera
            List<DetectionInfo> detections = ParseDetectionsProps(worker, cam);
            combinedDetections.AddRange(detections);
        }

        latestDetections = combinedDetections;

        if (latestDetections.Count > 0)
            OnDetections?.Invoke(latestDetections);
    }

    private List<DetectionInfo> ParseDetectionsProps(Worker worker, Camera cam)
    {
        List<DetectionInfo> detections = new();

        var output = worker.PeekOutput() as Tensor<float>;
        if (output == null) return detections;

        using var cpu = output.ReadbackAndClone(); // CPU tensor

        // YOLOv8n ONNX (Ultralytics): either (1,84,8400) or (1,8400,84)
        bool channelsFirst = (cpu.shape[1] == 84); // (N,C,B)
        int numBoxes = channelsFirst ? cpu.shape[2] : cpu.shape[1];
        int numAttrs = 84; // 4 box + 80 cls

        for (int i = 0; i < numBoxes; i++)
        {
            // --- read box ---
            float cx, cy, w, h;
            if (channelsFirst) { cx = cpu[0, 0, i]; cy = cpu[0, 1, i]; w = cpu[0, 2, i]; h = cpu[0, 3, i]; }
            else { cx = cpu[0, i, 0]; cy = cpu[0, i, 1]; w = cpu[0, i, 2]; h = cpu[0, i, 3]; }

            // --- best class ---
            int bestClass = -1; float bestScore = 0f;
            for (int c = 4; c < numAttrs; c++)
            {
                float s = channelsFirst ? cpu[0, c, i] : cpu[0, i, c];
                s = Sigmoid(s); // safe; most exports are post-sigmoid anyway
                if (s > bestScore) { bestScore = s; bestClass = c - 4; }
            }
            if (bestScore < confidenceThreshold || bestClass < 0 || bestClass >= cocoLabels.Length) continue;

            string label = cocoLabels[bestClass];

            // --- bbox to pixel space (same math as your car detector) ---
            float x = (cx - w / 2f) / inputWidth * cam.pixelWidth;
            float y = (cy - h / 2f) / inputHeight * cam.pixelHeight;
            float bw = w / inputWidth * cam.pixelWidth;
            float bh = h / inputHeight * cam.pixelHeight;
            Rect bbox = new Rect(x, y, bw, bh);

            // --- raycast from bbox center to get world pos ---
            Vector3 screenPoint = new Vector3(x + bw / 2f, y + bh / 2f, cam.nearClipPlane + 1f);
            Ray ray = cam.ScreenPointToRay(screenPoint);
            if (Physics.Raycast(ray, out RaycastHit hit))
            {
                var det = new DetectionInfo
                {
                    label = label,
                    bbox = bbox,
                    colour = SampleColour(bbox),
                    worldPos = hit.point,
                    distance = hit.distance,
                    relDir = GetRelativeDirection(cam, hit.point),
                    surface = hit.collider.gameObject.name,
                    confidence = bestScore
                };

                detections.Add(det);

                // Debug line
                Vector3 debugPos = det.worldPos + Vector3.up * 2f;
                Color debugCol = Color.gray;
                if (ColorUtility.TryParseHtmlString(det.colour, out Color parsed)) debugCol = parsed;
                Debug.DrawLine(det.worldPos, debugPos, debugCol, 2f);
            }
        }

        return ApplyNMS(detections, 0.45f);
    }

    // === Helpers (same style as your car detector) ===

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
        var sorted = detections.OrderByDescending(d => d.confidence).ToList();
        while (sorted.Count > 0)
        {
            var best = sorted[0];
            results.Add(best);
            sorted.RemoveAt(0);
            sorted = sorted.Where(d => !(d.label == best.label && IoU(d.bbox, best.bbox) > iouThreshold)).ToList();
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
        // optional: release RTs you created here
        // for (int i = 0; i < visionRTs?.Length; i++) if (visionRTs[i]) visionRTs[i].Release();
    }
}
