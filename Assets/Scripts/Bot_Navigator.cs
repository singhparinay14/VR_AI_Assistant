using UnityEngine;
using UnityEngine.AI;

public class BotNavigator : MonoBehaviour
{
    public NavMeshAgent agent;
    [SerializeField] private PathDrawer pathDrawer;
    [SerializeField] private Animator animator;

    private bool isMoving = false;

    public void MoveToTarget(Vector3 targetPos)
    {
        if (agent == null) return;

        agent.SetDestination(targetPos);
        isMoving = true;

        // Ensure the path is drawn from the bot's current position
        if (pathDrawer != null)
        {
            if (pathDrawer.startPoint == null)
                pathDrawer.startPoint = this.transform; // draw from bot by default

            pathDrawer.DrawPathTo(targetPos);
        }
    }

    private void Update()
    {
        // Set Speed parameter for animation
        if (animator != null && agent != null)
        {
            float speed = agent.velocity.magnitude;
            animator.SetFloat("Speed", speed);
        }

        // Stop movement check
        if (isMoving && agent != null && !agent.pathPending && agent.remainingDistance <= agent.stoppingDistance)
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