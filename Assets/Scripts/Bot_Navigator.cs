using UnityEngine;
using UnityEngine.AI;

public class BotNavigator : MonoBehaviour
{
    public NavMeshAgent agent;

    [SerializeField] private PathDrawer pathDrawer; // 💡 Reference to the path visualizer

    private bool isMoving = false;

    public void MoveToTarget(Vector3 targetPos)
    {
        agent.SetDestination(targetPos);
        isMoving = true;

        if (pathDrawer != null)
            pathDrawer.DrawPathTo(targetPos);
    }

    private void Update()
    {
        if (isMoving && !agent.pathPending && agent.remainingDistance <= agent.stoppingDistance)
        {
            if (!agent.hasPath || agent.velocity.sqrMagnitude == 0f)
            {
                if (pathDrawer != null)
                    pathDrawer.ClearPath();

                isMoving = false;
            }
        }
    }
}
