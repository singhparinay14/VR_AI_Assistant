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
    public bool drawRays = true;
    public bool drawAfterNMS = true;
    public int maxRaysDrawn = 20;
    public float rayDuration = 0.15f;
    public Color rayHitColor = Color.green;
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

    // Fine-tuned car labels (order must match the model)
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

    struct Raw
    {
        public int cls;
        public float conf, cxN, cyN, wN, hN;
    }

    private List<DetectionInfo> latestDetections = new();

    void Awake()
    {
        if (visionCameras == null) visionCameras = Array.Empty<Camera>();
        if (visionRTs == null || visionRTs.Length != visionCameras.Length)
            visionRTs = new RenderTexture[visionCameras.Length];

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

            cam.targetTexture = visionRTs[i];
            cam.allowHDR = false;
            cam.allowMSAA = false;
            cam.clearFlags = CameraClearFlags.SolidColor;
            cam.backgroundColor = Color.black;
        }

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

        if (Time.time < _nextTick) return;
        _nextTick = Time.time + 1f / Mathf.Max(1f, targetInferenceFPS);

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

            var input = new Tensor<float>(new TensorShape(1, 3, inputHeight, inputWidth));
            TextureConverter.ToTensor(rt, input, _texTransform);
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

            var detections = ParseDetections(worker, cam);

            if (maxDetectionsPerCamera > 0)
                detections = detections.OrderByDescending(d => d.confidence)
                                       .Take(maxDetectionsPerCamera).ToList();

            _camCache[i] = detections;
            _camCacheTime[i] = Time.time;
        }

        var combinedDetections = new List<DetectionInfo>(32);
        for (int k = 0; k < visionCameras.Length; k++)
            if (Time.time - _camCacheTime[k] <= cacheTTL)
                combinedDetections.AddRange(_camCache[k]);

        latestDetections = combinedDetections;
        OnDetections?.Invoke(latestDetections);
    }

    private List<DetectionInfo> ParseDetections(Worker worker, Camera cam)
    {
        var detections = new List<DetectionInfo>(16);

        var output = worker.PeekOutput() as Tensor<float>;
        if (output == null) return detections;

        using var cpu = output.ReadbackAndClone();

        // --- infer layout ---
        int A = cpu.shape[1];
        int B = cpu.shape[2];
        int numAttrs = Mathf.Min(A, B);
        int numBoxes = Mathf.Max(A, B);
        bool channelsFirst = (A < B);

        int classes = carLabels.Length;
        bool hasObj = (numAttrs == (5 + classes));
        int clsStart = hasObj ? 5 : 4;

        // 1) Collect raw predictions (no raycasts yet)

        var raw = new List<Raw>(numBoxes);

        for (int i = 0; i<numBoxes; i++)
        {
            // bbox
            float cx, cy, w, h, obj = 1f;
            if (channelsFirst)
            {
                cx = cpu[0, 0, i]; cy = cpu[0, 1, i]; w = cpu[0, 2, i]; h = cpu[0, 3, i];
                if (hasObj) obj = 1f / (1f + Mathf.Exp(-cpu[0, 4, i]));
            }
            else
            {
                cx = cpu[0, i, 0]; cy = cpu[0, i, 1]; w = cpu[0, i, 2]; h = cpu[0, i, 3];
                if (hasObj) obj = 1f / (1f + Mathf.Exp(-cpu[0, i, 4]));
            }

            // best class
            int bestClass = -1; float bestScore = 0f;
            for (int c = clsStart; c < numAttrs; c++)
            {
                float s = channelsFirst ? cpu[0, c, i] : cpu[0, i, c];
                s = 1f / (1f + Mathf.Exp(-s));
                if (s > bestScore) { bestScore = s; bestClass = c - clsStart; }
            }

            float conf = hasObj ? Mathf.Sqrt(obj * bestScore) : bestScore;
            if (conf < confidenceThreshold) continue;
            if (bestClass < 0 || bestClass >= classes) continue;

            // normalize regardless of export style (pixels vs normalized)
            bool inPixels = (Mathf.Abs(cx) > 2f || Mathf.Abs(cy) > 2f || Mathf.Abs(w) > 2f || Mathf.Abs(h) > 2f);
            float cxN = inPixels ? cx / inputWidth : cx;
            float cyN = inPixels ? cy / inputHeight : cy;
            float wN = inPixels ? w / inputWidth : w;
            float hN = inPixels ? h / inputHeight : h;

            raw.Add(new Raw { cls = bestClass, conf = conf, cxN = cxN, cyN = cyN, wN = wN, hN = hN });
        }

        if (raw.Count == 0) return detections;

        // 2) Keep only the strongest few before any raycasts (perf!)
        int cap = (maxDetectionsPerCamera > 0) ? Mathf.Max(1, maxDetectionsPerCamera * 2) : Mathf.Min(50, raw.Count);
        raw = raw.OrderByDescending(r => r.conf).Take(cap).ToList();

        // 3) Raycast the kept predictions
        int raycastsDone = 0;
        int raycastCap = (maxDetectionsPerCamera > 0) ? maxDetectionsPerCamera : int.MaxValue;
        float castLen = GetCastLength(cam);

        foreach (var r in raw)
        {
            if (raycastsDone >= raycastCap) break;

            // viewport center (normalized) → ray
            float vx = r.cxN;
            float vy = invertYForRay ? 1f - r.cyN : r.cyN;

            Ray ray = cam.ViewportPointToRay(new Vector3(vx, vy, 0f));
            bool hitOk = Physics.Raycast(ray, out RaycastHit hit, castLen, hitMask, QueryTriggerInteraction.Collide);
            raycastsDone++;

            // Fallback: small spherecast along the same ray to catch body panels near the center-line
            if ((!hitOk || IsLikelyFloorOrWall(hit)) && r.wN > 0.02f)
            {
                const float radius = 0.20f; // ~20cm
                if (Physics.SphereCast(ray, radius, out RaycastHit sh, castLen, hitMask, QueryTriggerInteraction.Collide))
                {
                    hit = sh;
                    hitOk = true;
                }
                else
                {
                    // last try: left/right thirds of the bbox
                    float vxL = Mathf.Clamp01(r.cxN - r.wN * 0.25f);
                    float vxR = Mathf.Clamp01(r.cxN + r.wN * 0.25f);
                    Ray rL = cam.ViewportPointToRay(new Vector3(vxL, vy, 0f));
                    Ray rR = cam.ViewportPointToRay(new Vector3(vxR, vy, 0f));

                    bool okL = Physics.Raycast(rL, out RaycastHit hL, castLen, hitMask, QueryTriggerInteraction.Collide);
                    bool okR = Physics.Raycast(rR, out RaycastHit hR, castLen, hitMask, QueryTriggerInteraction.Collide);

                    if (okL && (!hitOk || hL.distance < hit.distance)) { ray = rL; hit = hL; hitOk = true; }
                    if (okR && (!hitOk || hR.distance < hit.distance)) { ray = rR; hit = hR; hitOk = true; }
                }
            }

            // Normalize identity to the object's root so duplicates from sub-colliders collapse
            Transform root = hitOk ? hit.collider.transform.root : null;
            string surfaceName = hitOk ? (root != null ? root.name : hit.collider.name) : "nohit";


            // pixel bbox (for UI/logging)
            float xPix = (r.cxN - r.wN * 0.5f) * cam.pixelWidth;
            float yPix = (r.cyN - r.hN * 0.5f) * cam.pixelHeight;
            float wPix = r.wN * cam.pixelWidth;
            float hPix = r.hN * cam.pixelHeight;
            Rect bbox = new Rect(xPix, yPix, wPix, hPix);

            var det = new DetectionInfo
            {
                label = carLabels[r.cls],
                bbox = bbox,
                colour = enableColourSampling ? SampleColour(bbox, cam) : "gray",
                worldPos = hitOk ? hit.point : (cam.transform.position + cam.transform.forward * 2f),
                distance = hitOk ? hit.distance : -1f,
                relDir = GetRelativeDirection(cam, hitOk ? hit.point : (cam.transform.position + cam.transform.forward * 2f)),
                surface = surfaceName,
                confidence = r.conf
            };

            detections.Add(det);

            if (logDetections)
                Debug.Log($"[YOLO-cars] {det.label} {det.confidence:0.50} hit={hitOk} @ {det.worldPos} (vx={vx:0.02}, vy={vy:0.02})");
        }

        // 4) NMS per label
        var kept = ApplyNMS(detections, 0.7f);

        #if UNITY_EDITOR
            if (drawRays && drawAfterNMS)
            {
                int drawn = 0;
                foreach (var d in kept.OrderByDescending(k => k.confidence))
                {
                    if (drawn >= Mathf.Max(1, maxRaysDrawn)) break;

                    // rebuild the ray from bbox center for viz
                    float vx = (d.bbox.x + d.bbox.width * 0.5f) / cam.pixelWidth;
                    float vy = (d.bbox.y + d.bbox.height * 0.5f) / cam.pixelHeight;
                    if (invertYForRay) vy = 1f - vy;

                    var r2 = cam.ViewportPointToRay(new Vector3(vx, vy, 0f));
                    float length = (d.distance >= 0f) ? d.distance : castLen;  // no extra Physics.Raycast here
                    Debug.DrawRay(r2.origin, r2.direction * length, (d.distance >= 0f) ? rayHitColor : rayMissColor, rayDuration);

                    if (d.distance >= 0f)
                    {
                        Debug.DrawLine(d.worldPos + Vector3.up * 0.05f,       d.worldPos - Vector3.up * 0.05f,       hitMarkerColor, rayDuration);
                        Debug.DrawLine(d.worldPos + Vector3.right * 0.05f,    d.worldPos - Vector3.right * 0.05f,    hitMarkerColor, rayDuration);
                        Debug.DrawLine(d.worldPos + Vector3.forward * 0.05f,  d.worldPos - Vector3.forward * 0.05f,  hitMarkerColor, rayDuration);
                    }
                    drawn++;
                }
            }
        #endif

        return kept;
    }


    private static bool IsLikelyFloorOrWall(RaycastHit hit)
    {
        if (hit.collider == null) return true;
        var b = hit.collider.bounds;
        // very thin in one dimension -> large plane (floor/wall)
        return (b.size.y < 0.05f) || (b.size.x > 50f) || (b.size.z > 50f);
    }

    private float GetCastLength(Camera cam)
        => Mathf.Max(0.1f, cam.farClipPlane - cam.nearClipPlane);

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
    {
        if (string.IsNullOrWhiteSpace(word)) return false;
        return carLabels.Any(l => l.IndexOf(word, StringComparison.OrdinalIgnoreCase) >= 0);
    }

    public List<DetectionInfo> GetLatestDetections()
        => latestDetections ?? new List<DetectionInfo>();

    void OnDestroy() => worker?.Dispose();
}
