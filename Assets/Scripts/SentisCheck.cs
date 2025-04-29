using UnityEngine;
using Unity.Sentis;

public class SentisCheck : MonoBehaviour
{
    void Start()
    {
        var tensor = new Tensor<float>(new TensorShape(1, 2, 3));
        //var shape = new TensorShape(1);
        //Tensor tensor = new TensorFloat32(shape); // older Sentis class
        tensor.Dispose();
        Debug.Log("Sentis is recognized and working!");
    }
}
