import unittest
from self_healing_agents.orchestrator import should_trigger_self_healing, DEFAULT_SUCCESS_THRESHOLD
from self_healing_agents.schemas import (
    CriticReport,
    CRITIC_STATUS_SUCCESS,
    CRITIC_STATUS_FAILURE_SYNTAX,
    CRITIC_STATUS_FAILURE_RUNTIME,
    CRITIC_STATUS_FAILURE_LOGIC
)

class TestSelfHealingTriggerLogic(unittest.TestCase):

    def test_trigger_on_syntax_failure(self):
        report = CriticReport(status=CRITIC_STATUS_FAILURE_SYNTAX, score=0.0)
        self.assertTrue(should_trigger_self_healing(report))

    def test_trigger_on_runtime_failure(self):
        report = CriticReport(status=CRITIC_STATUS_FAILURE_RUNTIME, score=0.2)
        self.assertTrue(should_trigger_self_healing(report))

    def test_trigger_on_logic_failure(self):
        report = CriticReport(status=CRITIC_STATUS_FAILURE_LOGIC, score=0.5)
        self.assertTrue(should_trigger_self_healing(report))

    def test_trigger_on_low_score_despite_success_status(self):
        # Edge case: status is SUCCESS, but score is below threshold
        report = CriticReport(status=CRITIC_STATUS_SUCCESS, score=DEFAULT_SUCCESS_THRESHOLD - 0.1)
        self.assertTrue(should_trigger_self_healing(report))

    def test_no_trigger_on_success_above_threshold(self):
        report = CriticReport(status=CRITIC_STATUS_SUCCESS, score=DEFAULT_SUCCESS_THRESHOLD)
        self.assertFalse(should_trigger_self_healing(report))

    def test_no_trigger_on_success_well_above_threshold(self):
        report = CriticReport(status=CRITIC_STATUS_SUCCESS, score=0.95)
        self.assertFalse(should_trigger_self_healing(report))

    def test_custom_threshold_trigger(self):
        custom_threshold = 0.9
        report = CriticReport(status=CRITIC_STATUS_SUCCESS, score=custom_threshold - 0.05)
        self.assertTrue(should_trigger_self_healing(report, success_threshold=custom_threshold))

    def test_custom_threshold_no_trigger(self):
        custom_threshold = 0.7
        report = CriticReport(status=CRITIC_STATUS_SUCCESS, score=custom_threshold + 0.05)
        self.assertFalse(should_trigger_self_healing(report, success_threshold=custom_threshold))
        
    def test_score_at_threshold_is_not_failure(self):
        report = CriticReport(status=CRITIC_STATUS_SUCCESS, score=DEFAULT_SUCCESS_THRESHOLD)
        self.assertFalse(should_trigger_self_healing(report))

    def test_invalid_report_type(self):
        with self.assertRaises(TypeError):
            should_trigger_self_healing({"status": CRITIC_STATUS_FAILURE_LOGIC, "score": 0.1}) # type: ignore

    def test_invalid_threshold_too_low(self):
        report = CriticReport(status=CRITIC_STATUS_SUCCESS, score=0.9)
        with self.assertRaises(ValueError):
            should_trigger_self_healing(report, success_threshold=-0.1)

    def test_invalid_threshold_too_high(self):
        report = CriticReport(status=CRITIC_STATUS_SUCCESS, score=0.9)
        with self.assertRaises(ValueError):
            should_trigger_self_healing(report, success_threshold=1.1)

if __name__ == '__main__':
    unittest.main() 