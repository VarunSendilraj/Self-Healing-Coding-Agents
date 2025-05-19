from self_healing_agents.schemas import CriticReport, CRITIC_STATUS_SUCCESS

DEFAULT_SUCCESS_THRESHOLD = 0.8 # As per PRD, e.g. score below a success threshold

def should_trigger_self_healing(
    report: CriticReport,
    success_threshold: float = DEFAULT_SUCCESS_THRESHOLD
) -> bool:
    """
    Determines whether the self-healing mechanism should be triggered based on the Critic's report.

    Args:
        report: The structured report from the Critic agent.
        success_threshold: The minimum score to be considered a success without triggering self-healing.

    Returns:
        True if self-healing should be triggered, False otherwise.
    """
    if not isinstance(report, CriticReport):
        raise TypeError("Input 'report' must be an instance of CriticReport.")

    if not 0.0 <= success_threshold <= 1.0:
        raise ValueError("success_threshold must be between 0.0 and 1.0.")

    # Trigger if the status indicates any failure
    if report.status != CRITIC_STATUS_SUCCESS:
        return True
    
    # Trigger if the score is below the success threshold, even if status is nominally SUCCESS
    # (This could happen if tests pass but score is still low due to other quality metrics in a more advanced scorer)
    if report.score < success_threshold:
        return True
        
    return False 