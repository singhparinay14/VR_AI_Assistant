//using System.Diagnostics;
using UnityEngine;
using UnityEngine.AI;

[RequireComponent(typeof(LineRenderer))]
public class PathDrawer : MonoBehaviour
{
    private LineRenderer lineRenderer;
    private NavMeshPath navPath;
    public Transform startPoint; // usually the AI or player

    void Awake()
    {
        lineRenderer = GetComponent<LineRenderer>();
        navPath = new NavMeshPath();
    }

    public void DrawPathTo(Vector3 targetPosition)
    {
        if (startPoint == null)
        {
            Debug.LogWarning("PathDrawer: Start point not assigned.");
            return;
        }

        NavMeshPath path = new NavMeshPath();
        if (NavMesh.CalculatePath(startPoint.position, targetPosition, NavMesh.AllAreas, path))
        {
            lineRenderer.positionCount = path.corners.Length;
            lineRenderer.SetPositions(path.corners);
            Debug.Log("PathDrawer: Drawing path with " + path.corners.Length + " corners.");
        }
        else
        {
            Debug.LogWarning("PathDrawer: Could not calculate path.");
            lineRenderer.positionCount = 0;
        }

        // Debug line (optional)
        Debug.DrawLine(startPoint.position, targetPosition, Color.red, 5f);
    }


    public void ClearPath()
    {
        lineRenderer.positionCount = 0;
    }
}
