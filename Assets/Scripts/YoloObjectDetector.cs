using UnityEngine;
using Unity.Sentis;
using System.Collections.Generic;
using System.Linq;

public class YoloObjectDetector : MonoBehaviour
{
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
        model = ModelLoader.Load(modelAsset);
        worker = new Worker(model, BackendType.GPUCompute);
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


        if(modelAsset == null)
        {
            Debug.LogWarning("Model asset is not set.");
            return;
        }

        // prepare transform to resize to inputWidth x inputHeight, 3 channels
        var transform = new TextureTransform()
            .SetDimensions(inputWidth, inputHeight, 3);

        // allocate input tensor in NCHW layout
        var inputShape = new TensorShape(1, 3, inputHeight, inputWidth);
        using Tensor<float> input = new Tensor<float>(inputShape);

        // fill tensor from texture (new void overload) :contentReference[oaicite:0]{index=0}
        TextureConverter.ToTensor(visionTexture, input, transform);

        // schedule inference
        worker.Schedule(input);

        // fetch output and sync GPU->CPU
        using Tensor<float> cpuOut =
            worker.PeekOutput().ReadbackAndClone() as Tensor<float>;

        // parse and log detections
        var labels = ParseDetections(cpuOut);
        if (labels.Count > 0)
            Debug.Log("Objects detected: " + string.Join(", ", labels.Distinct()));
        else
            Debug.Log("No objects detected in this frame.");
    }

    List<string> ParseDetections(Tensor output)
    {
        var results = new List<string>();
        var t = output as Tensor<float>;
        if (t == null) return results;

        bool attrsFirst = t.shape[1] == 84;
        int boxes = attrsFirst ? t.shape[2] : t.shape[1];

        for (int b = 0; b < boxes; b++)
        {
            int bestClass = 0;
            float bestScore = 0f;

            for (int c = 0; c < 80; c++)
            {
                float score = attrsFirst
                    ? t[0, 4 + c, b]
                    : t[0, b, 4 + c];

                if (score > bestScore)
                {
                    bestScore = score;
                    bestClass = c;
                }
            }

            if (bestScore >= confidenceThreshold)
            {
                results.Add(cocoLabels[bestClass]);
            }
        }

        return results;
    }

    void OnDestroy()
    {
        worker?.Dispose();
    }
}
