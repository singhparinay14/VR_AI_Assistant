using System.Collections.Generic;
using UnityEngine;

public class MaterialRandomizerScript : MonoBehaviour
{
    public List<Material> materials;
    public List<GameObject> gameObjects;

    public void randomizeMaterials()
    {
        if (materials == null || gameObjects == null) return;

        var materialsCopy = new List<Material>(materials);
        foreach (var go in gameObjects)
        {
            if (go == null) continue;
            if (materialsCopy.Count == 0)
            {
                // No more materials to assign
                return;
            }

            // Use UnityEngine.Random explicitly to avoid ambiguity with System.Random
            int chosen = UnityEngine.Random.Range(0, materialsCopy.Count);

            var mr = go.GetComponent<MeshRenderer>();
            if (mr != null)
            {
                mr.material = materialsCopy[chosen];
            }

            materialsCopy.RemoveAt(chosen);
        }
    }

#if UNITY_EDITOR
    // Editor-only: requires UnityEditor.AssetDatabase
    public void findMaterials()
    {
        var guids = UnityEditor.AssetDatabase.FindAssets("t:Material", new[] { "Assets/PBS Materials Variety Pack/" });
        if (materials == null) materials = new List<Material>();
        materials.Clear();

        foreach (string id in guids)
        {
            string path = UnityEditor.AssetDatabase.GUIDToAssetPath(id);
            if (!path.Contains("coming_soon"))
            {
                var mat = UnityEditor.AssetDatabase.LoadMainAssetAtPath(path) as Material;
                if (mat != null)
                {
                    materials.Add(mat);
                }
            }
        }
    }
#endif

    public void findMaterialSpheres()
    {
        if (gameObjects == null) gameObjects = new List<GameObject>();
        gameObjects.Clear();

        foreach (var gameObj in FindObjectsByType<GameObject>(FindObjectsSortMode.None))
        {
            if (gameObj != null && gameObj.name == "Material Sphere")
            {
                gameObjects.Add(gameObj);
            }
        }
    }
}