import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from measure_deployments import summarize


def test_injected_failures_are_not_counted_as_successful_deployments():
    rows = [dict(scenario='failed_start', deployment_success=False, expected_behavior_passed=True,
                 deployment_seconds=float(i), scenario_check_seconds=float(i)+.2,
                 sampled_unavailable_seconds=0.) for i in range(1, 21)]
    result = summarize(rows)['failed_start']
    assert result['N'] == 20
    assert result['deployment_success_rate'] == 0
    assert result['expected_behavior_pass_rate'] == 1
    assert result['deployment_seconds']['mean'] == 10.5
    assert result['deployment_seconds']['median'] == 10.5
    assert result['deployment_seconds']['P95'] == 19
    assert result['deployment_seconds']['min'] == 1
    assert result['deployment_seconds']['max'] == 20
    assert result['sampled_unavailable_seconds']['max'] == 0
    assert summarize(rows)['successful_replacement']['N'] == 0
