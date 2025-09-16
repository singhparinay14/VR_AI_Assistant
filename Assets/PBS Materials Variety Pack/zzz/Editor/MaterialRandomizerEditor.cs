#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;

[CustomEditor(typeof(MaterialRandomizerScript))]
public class MaterialRandomizerEditor : Editor
{
    public override void OnInspectorGUI()
    {
        DrawDefaultInspector();

        var myScript = (MaterialRandomizerScript)target;

        if (GUILayout.Button("Find Materials"))
        {
            myScript.findMaterials();
        }
        if (GUILayout.Button("Find Spheres"))
        {
            myScript.findMaterialSpheres();
        }
        if (GUILayout.Button("Randomize Materials"))
        {
            myScript.randomizeMaterials();
        }
    }
}
#endif