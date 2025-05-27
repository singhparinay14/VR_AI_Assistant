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

        if (NavMesh.CalculatePath(startPoint.position, targetPosition, NavMesh.AllAreas, navPath))
        {
            lineRenderer.positionCount = navPath.corners.Length;
            lineRenderer.SetPositions(navPath.corners);
        }
        else
        {
            Debug.LogWarning("PathDrawer: Could not calculate path.");
            lineRenderer.positionCount = 0;
        }
    }

    public void ClearPath()
    {
        lineRenderer.positionCount = 0;
    }
}
