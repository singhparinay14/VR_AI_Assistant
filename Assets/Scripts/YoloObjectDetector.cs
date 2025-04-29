using UnityEngine;
using Unity.Sentis;
using System;
using System.Collections.Generic;
using System.Linq;

public struct DetectionInfo
{
    public string label;   // COCO class name
    public Rect bbox;    // pixel rect in the resized 416x416 image
    public string colour;  // coarse colour word
}

public class YoloObjectDetector : MonoBehaviour
{
    public event Action<List<DetectionInfo>> OnDetections;   // subscribers get the list each frame

    [Header("Model & Vision")]
    public ModelAsset modelAsset;
    public RenderTexture visionTexture;
    [Range(0f, 1f)]
    public float confidenceThreshold = 0.5f;

    private Worker worker;
    private Model model;

    [Header("Input Dimensions")]
    public int inputWidth = 416;
    public int inputHeight = 416;

    // reusable readable texture for colour sampling
    private Texture2D readTex;

    // COCO labels (80 classes)
    private readonly string[] cocoLabels =
    {
        "person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat","traffic light",
        "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
        "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
        "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
        "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
        "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","sofa",
        "pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse","remote","keyboard",
        "cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors",
        "teddy bear","hair drier","toothbrush"
    };

    void Start()
    {
        if (modelAsset == null)
        {
            Debug.LogError("YoloObjectDetector: modelAsset not set.");
            enabled = false;
            return;
        }

        model = ModelLoader.Load(modelAsset);
        worker = new Worker(model, BackendType.GPUCompute);

        readTex = new Texture2D(inputWidth, inputHeight, TextureFormat.RGB24, false);

        Debug.Log("YoloObjectDetector initialised.");
    }

    void Update()
    {
        RunDetection();
    }

    void RunDetection()
    {
        if (visionTexture == null)
        {
            Debug.LogWarning("Vision texture is not set.");
            return;
        }

        // ---------- build input tensor ----------
        var transform = new TextureTransform().SetDimensions(inputWidth, inputHeight, 3);
        var inputShape = new TensorShape(1, 3, inputHeight, inputWidth);
        using Tensor<float> input = new Tensor<float>(inputShape);

        TextureConverter.ToTensor(visionTexture, input, transform);

        // ---------- run network ----------
        worker.Schedule(input);

        using Tensor<float> cpuOut =
            worker.PeekOutput().ReadbackAndClone() as Tensor<float>;

        // ---------- make a CPU copy of the frame for colour sampling ----------
        RenderTexture.active = visionTexture;
        readTex.ReadPixels(new Rect(0, 0, inputWidth, inputHeight), 0, 0);
        readTex.Apply();
        RenderTexture.active = null;

        // ---------- parse detections ----------
        var detections = ParseDetections(cpuOut, readTex);

        // broadcast to anyone listening (for chat context)
        OnDetections?.Invoke(detections);

        // simple log for debugging
        if (detections.Count > 0)
        {
            Debug.Log("Objects detected: " +
                      string.Join(", ", detections.Select(d => d.colour + " " + d.label).Distinct()));
            foreach (var d in detections)
            {
                Debug.Log($"DETECT {d.label} -> {d.colour}");

            }
        }
        else
            Debug.Log("No objects detected in this frame.");
    }

    // =========================================================================================
    // Helpers
    // =========================================================================================

    List<DetectionInfo> ParseDetections(Tensor<float> t, Texture2D srcTex)
    {
        var list = new List<DetectionInfo>();

        bool attrsFirst = t.shape[1] == 84;
        int boxes = attrsFirst ? t.shape[2] : t.shape[1];

        for (int b = 0; b < boxes; b++)
        {
            // ----- class selection -----
            int bestClass = 0;
            float bestScore = 0f;

            for (int c = 0; c < 80; c++)
            {
                float score = attrsFirst ? t[0, 4 + c, b] : t[0, b, 4 + c];
                if (score > bestScore)
                {
                    bestScore = score;
                    bestClass = c;
                }
            }

            if (bestScore < confidenceThreshold)
                continue;

            // ----- bbox (x,y,w,h) normalised -----
            float x = attrsFirst ? t[0, 0, b] : t[0, b, 0];
            float y = attrsFirst ? t[0, 1, b] : t[0, b, 1];
            float w = attrsFirst ? t[0, 2, b] : t[0, b, 2];
            float h = attrsFirst ? t[0, 3, b] : t[0, b, 3];

            var rect = new Rect(
                (x - w * 0.5f) * inputWidth,
                (y - h * 0.5f) * inputHeight,
                w * inputWidth,
                h * inputHeight);

            // ----- colour sampling -----
            string colWord = SampleColour(srcTex, rect);

            list.Add(new DetectionInfo
            {
                label = cocoLabels[bestClass],
                bbox = rect,
                colour = colWord
            });
        }
        return list;
    }

    string SampleColour(Texture2D tex, Rect r)
    {
        int x0 = Mathf.Clamp(Mathf.RoundToInt(r.x), 0, tex.width - 1);
        int y0 = Mathf.Clamp(Mathf.RoundToInt(r.y), 0, tex.height - 1);
        int x1 = Mathf.Clamp(Mathf.RoundToInt(r.x + r.width), 0, tex.width - 1);
        int y1 = Mathf.Clamp(Mathf.RoundToInt(r.y + r.height), 0, tex.height - 1);

        bool linear = QualitySettings.activeColorSpace == ColorSpace.Linear;

        // best pixel seen so far
        float bestS = 0f, bestH = 0f, bestV = 0f;

        for (int y = y0; y <= y1; y += 2)           // step 2 px for speed
            for (int x = x0; x <= x1; x += 2)
            {
                Color c = tex.GetPixel(x, y);
                if (linear) c = c.gamma;                // convert to sRGB

                Color.RGBToHSV(c, out float h, out float s, out float v);

                if (s > bestS && v > 0.15f)             // ignore very dark pixels
                {
                    bestS = s;
                    bestH = h;
                    bestV = v;
                }
            }

        if (bestS < 0.18f) return "gray";           // not enough saturation
        if (bestV > 0.90f && bestS < 0.20f) return "white";
        if (bestV < 0.12f) return "black";

        if (bestH < 15f || bestH >= 345f) return "red";
        if (bestH < 45f) return "orange";
        if (bestH < 70f) return "yellow";
        if (bestH < 170f) return "green";
        if (bestH < 205f) return "cyan";
        if (bestH < 255f) return "blue";
        if (bestH < 295f) return "purple";
        return "pink";
    }



    void OnDestroy()
    {
        worker?.Dispose();
    }
}
