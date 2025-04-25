using UnityEngine;
using Unity.Sentis;
using System.Collections.Generic;
using System.Linq;

public class YoloObjectDetector : MonoBehaviour
{
    public NNModel modelAsset;
    public RenderTexture inputTexture;
    public Texture2D readableTexture;

    public float confidenceThreshold = 0.5f;
    public int inputWidth = 416;
    public int inputHeight = 416;

    IWorker worker;
    Model model;

    string[] cocoLabels = new string[]
    {
        "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
        "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
        "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
        "teddy bear", "hair drier", "toothbrush"
    };

    void Start()
    {
        model = ModelLoader.Load(modelAsset);
        worker = WorkerFactory.CreateWorker(BackendType.GPUCompute, model);

        readableTexture = new Texture2D(inputWidth, inputHeight, TextureFormat.RGB24, false);
        Debug.Log("YOLO model loaded and worker initialized.");
    }

    void Update()
    {
        RunDetection();
    }

    void RunDetection()
    {
        // Copy RenderTexture to Texture2D
        RenderTexture.active = inputTexture;
        readableTexture.ReadPixels(new Rect(0, 0, inputWidth, inputHeight), 0, 0);
        readableTexture.Apply();
        RenderTexture.active = null;

        // Prepare input tensor
        var input = TextureConverter.ToTensor(readableTexture, channels: 3);
        TensorShape shape = new TensorShape(1, 3, inputHeight, inputWidth); // NCHW
        input = input.Reshape(shape);

        // Run inference
        worker.Execute(input);
        Tensor output = worker.PeekOutput();

        // Parse YOLO results
        List<string> detections = ParseDetections(output);
        if (detections.Count > 0)
        {
            string context = string.Join(", ", detections.Distinct());
            Debug.Log("Objects detected: " + context);
            // Send context string to GPT input system if needed
        }

        input.Dispose();
        output.Dispose();
    }

    List<string> ParseDetections(Tensor output)
    {
        List<string> result = new List<string>();

        int numPreds = output.shape.length;

        for (int i = 0; i < output.shape[1]; i++)
        {
            float confidence = output[0, i, 4];
            if (confidence < confidenceThreshold)
                continue;

            float[] classScores = new float[cocoLabels.Length];
            for (int j = 0; j < classScores.Length; j++)
            {
                classScores[j] = output[0, i, 5 + j];
            }

            int bestClass = classScores.ToList().IndexOf(classScores.Max());
            string label = cocoLabels[bestClass];
            result.Add(label);
        }

        return result;
    }

    void OnDestroy()
    {
        worker?.Dispose();
    }
}
