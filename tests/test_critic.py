import pytest
from typing import List, Dict, Any, Optional

# Assuming the CriticAgent and status constants are in src.self_healing_agents.agents
# Adjust the import path based on your actual project structure and how you run tests.
# For example, if tests/ is a top-level directory and src/ is also top-level:
from self_healing_agents.agents import (
    CriticAgent,
    STATUS_CRITICAL_SYNTAX_ERROR,
    STATUS_CRITICAL_RUNTIME_ERROR,
    STATUS_LOGICAL_ERROR,
    STATUS_SUCCESS,
    NO_TESTS_FOUND
)

@pytest.fixture
def critic_agent() -> CriticAgent:
    """Pytest fixture to provide a CriticAgent instance."""
    return CriticAgent() # No LLM service needed for this method


def test_critical_syntax_error(critic_agent: CriticAgent):
    execution_details = {
        'error_type': 'SyntaxError', 
        'error_message': 'invalid syntax',
        'traceback': 'File "<string>", line 1\n  bad syntax\n^'
    }
    test_results = None
    report = critic_agent.analyze_results(execution_details, test_results)
    assert report['overall_status'] == STATUS_CRITICAL_SYNTAX_ERROR
    assert report['execution_error_type'] == 'SyntaxError'
    assert report['execution_error_message'] == 'invalid syntax'
    assert report['execution_traceback'] == 'File "<string>", line 1\n  bad syntax\n^'
    assert report['num_tests_total'] == 0
    assert report['quantitative_score'] == 0.0
    assert 'Critical syntax error: invalid syntax.' in report['concise_summary']
    assert 'Details: File "<string>", line 1' in report['concise_summary']
    assert report['all_test_details'] is None

def test_critical_runtime_error(critic_agent: CriticAgent):
    execution_details = {
        'error_type': 'ZeroDivisionError', 
        'error_message': 'division by zero',
        'traceback': 'Traceback (most recent call last):\nFile "<string>", line 1, in <module>\nZeroDivisionError: division by zero'
    }
    test_input_data = [
        {'name': 'test_div', 'status': 'error_during_exec', 'input': (1,0), 'expected': None, 'actual': None, 'error_message': 'ZeroDivisionError'}
    ]
    report = critic_agent.analyze_results(execution_details, test_input_data)
    assert report['overall_status'] == STATUS_CRITICAL_RUNTIME_ERROR
    assert report['execution_error_type'] == 'ZeroDivisionError'
    assert report['execution_traceback'] is not None
    assert 'division by zero' in report['execution_traceback']
    assert report['num_tests_total'] == 1
    assert report['num_tests_failed'] == 1
    assert report['quantitative_score'] == 0.1
    assert 'Critical runtime error: ZeroDivisionError - division by zero.' in report['concise_summary']
    assert 'Traceback: Traceback (most recent call last):' in report['concise_summary']
    assert report['all_test_details'] == test_input_data


def test_logical_error_some_tests_failed(critic_agent: CriticAgent):
    execution_details = None
    test_input_data = [
        {'name': 'test_add_ok', 'status': 'passed', 'input': (1,1), 'expected': 2, 'actual': 2, 'error_message': None},
        {'name': 'test_add_bad', 'status': 'failed', 'input': (1,2), 'expected': 4, 'actual': 3, 'error_message': 'AssertionError: 3 != 4'}
    ]
    report = critic_agent.analyze_results(execution_details, test_input_data)
    assert report['overall_status'] == STATUS_LOGICAL_ERROR
    assert report['execution_error_type'] is None
    assert report['execution_traceback'] is None
    assert report['num_tests_total'] == 2
    assert report['num_tests_passed'] == 1
    assert report['num_tests_failed'] == 1
    assert len(report['failed_test_details']) == 1
    assert report['failed_test_details'][0]['name'] == 'test_add_bad'
    assert report['quantitative_score'] == 0.55
    assert 'Code executed, but 1 of 2 test(s) failed.' in report['concise_summary']
    assert "First failed test: 'test_add_bad' - AssertionError: 3 != 4." in report['concise_summary']
    assert report['all_test_details'] == test_input_data

def test_success_all_tests_passed(critic_agent: CriticAgent):
    execution_details = None
    test_input_data = [
        {'name': 'test_add_ok', 'status': 'passed', 'input': (1,1), 'expected': 2, 'actual': 2, 'error_message': None},
        {'name': 'test_sub_ok', 'status': 'passed', 'input': (5,2), 'expected': 3, 'actual': 3, 'error_message': None}
    ]
    report = critic_agent.analyze_results(execution_details, test_input_data)
    assert report['overall_status'] == STATUS_SUCCESS
    assert report['execution_error_type'] is None
    assert report['execution_traceback'] is None
    assert report['quantitative_score'] == 1.0
    assert 'Code executed successfully and all 2 test(s) passed.' in report['concise_summary']
    assert report['all_test_details'] == test_input_data

def test_no_execution_errors_no_tests_run(critic_agent: CriticAgent):
    execution_details = None
    test_input_data = [] # No tests
    report = critic_agent.analyze_results(execution_details, test_input_data)
    assert report['overall_status'] == NO_TESTS_FOUND
    assert report['execution_traceback'] is None
    assert report['quantitative_score'] == 0.3
    assert 'Code executed without critical errors, but no tests were run to verify correctness.' in report['concise_summary']
    assert report['all_test_details'] == test_input_data

def test_no_execution_errors_none_test_results(critic_agent: CriticAgent):
    execution_details = None
    test_results = None # No tests
    report = critic_agent.analyze_results(execution_details, test_results)
    assert report['overall_status'] == NO_TESTS_FOUND
    assert report['execution_traceback'] is None
    assert report['num_tests_total'] == 0
    assert report['quantitative_score'] == 0.3
    assert 'Code executed without critical errors, but no tests were run to verify correctness.' in report['concise_summary']
    assert report['all_test_details'] is None

def test_critical_syntax_error_with_test_data_present(critic_agent: CriticAgent):
    execution_details = {
        'error_type': 'SyntaxError', 
        'error_message': 'EOL while scanning string literal',
        'traceback': 'File "<string>", line 1\n  print("hello)'
    }
    test_input_data = [
        {'name': 'test_string', 'status': 'untested', 'input': ("hello"), 'expected': "hello", 'actual': None, 'error_message': None}
    ]
    report = critic_agent.analyze_results(execution_details, test_input_data)
    assert report['overall_status'] == STATUS_CRITICAL_SYNTAX_ERROR
    assert report['execution_traceback'] is not None
    assert report['quantitative_score'] == 0.0
    assert 'Critical syntax error: EOL while scanning string literal.' in report['concise_summary']
    assert report['all_test_details'] == test_input_data

def test_logical_error_all_tests_failed(critic_agent: CriticAgent):
    execution_details = None
    test_input_data = [
        {'name': 'test_mul_bad', 'status': 'failed', 'input': (2,2), 'expected': 5, 'actual': 4, 'error_message': 'AssertionError: Actual 4 != Expected 5'},
        {'name': 'test_div_bad', 'status': 'failed', 'input': (10,2), 'expected': 6, 'actual': 5, 'error_message': 'AssertionError: Actual 5 != Expected 6'}
    ]
    report = critic_agent.analyze_results(execution_details, test_input_data)
    assert report['overall_status'] == STATUS_LOGICAL_ERROR
    assert report['execution_traceback'] is None
    assert report['quantitative_score'] == 0.2
    assert 'Code executed, but 2 of 2 test(s) failed.' in report['concise_summary']
    assert "First failed test: 'test_mul_bad' - AssertionError: Actual 4 != Expected 5." in report['concise_summary']
    assert report['all_test_details'] == test_input_data

def test_logical_error_one_third_passed(critic_agent: CriticAgent):
    execution_details = None
    test_input_data = [
        {'name': 'test_1_passed', 'status': 'passed', 'input': (1,1), 'expected': 2, 'actual': 2, 'error_message': None},
        {'name': 'test_2_failed', 'status': 'failed', 'input': (1,2), 'expected': 4, 'actual': 3, 'error_message': 'AssertionError: Output 3 not equal to expected 4'},
        {'name': 'test_3_failed', 'status': 'failed', 'input': (1,3), 'expected': 5, 'actual': 4, 'error_message': 'AssertionError: Output 4 not equal to expected 5'}
    ]
    report = critic_agent.analyze_results(execution_details, test_input_data)
    assert report['overall_status'] == STATUS_LOGICAL_ERROR
    assert report['quantitative_score'] == round(0.2 + 0.7 * (1/3), 4)
    assert 'Code executed, but 2 of 3 test(s) failed.' in report['concise_summary']
    assert "First failed test: 'test_2_failed' - AssertionError: Output 3 not equal to expected 4." in report['concise_summary']
    assert report['all_test_details'] == test_input_data

if __name__ == '__main__':
    pytest.main() 