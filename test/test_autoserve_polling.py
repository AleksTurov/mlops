from types import SimpleNamespace
from unittest.mock import Mock

import mlflow
import pytest
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST

from app import mlflow_autoserve as autoserve


class StopPolling(BaseException):
    """Stop the infinite loop at a known boundary without entering its error handler."""


@pytest.fixture
def loop(monkeypatch):
    registry = Mock()
    docker = Mock()
    deploy = Mock()
    log = Mock()
    monkeypatch.setenv('MLFLOW_SERVE_ALIASES', 'champion')
    monkeypatch.setattr(autoserve, '_start_health_endpoint', lambda: None)
    monkeypatch.setattr(autoserve, 'get_settings', lambda: SimpleNamespace(
        mlflow_tracking_uri='http://test-registry', mlflow_s3_endpoint_url='',
        aws_access_key_id='', aws_secret_access_key=''))
    monkeypatch.setattr(autoserve.mlflow, 'set_tracking_uri', lambda _: None)
    monkeypatch.setattr(autoserve.mlflow.tracking, 'MlflowClient', lambda: registry)
    monkeypatch.setattr(autoserve.docker, 'from_env', lambda: docker)
    monkeypatch.setattr(autoserve, '_resolve_trace_experiment', lambda **kwargs: '')
    monkeypatch.setattr(autoserve, '_ensure_container', deploy)
    monkeypatch.setattr(autoserve, 'logger', log)
    return registry, docker, deploy, log


def stop_after_two_polls(monkeypatch, deploy):
    sleeps = 0
    def sleep(_):
        nonlocal sleeps
        sleeps += 1
        if sleeps == 1:
            deploy.assert_not_called()
        else:
            raise StopPolling()
    monkeypatch.setattr(autoserve.time, 'sleep', sleep)


@pytest.mark.parametrize('failure_at', ['model_listing', 'alias_lookup'])
def test_registry_outage_is_logged_and_next_poll_recovers(loop, monkeypatch, failure_at):
    registry, docker, deploy, log = loop
    models = [SimpleNamespace(name='iris')]
    if failure_at == 'model_listing':
        registry.search_registered_models.side_effect = [ConnectionError('offline'), models]
        registry.get_model_version_by_alias.return_value = SimpleNamespace(version='2')
    else:
        registry.search_registered_models.return_value = models
        registry.get_model_version_by_alias.side_effect = [
            ConnectionError('offline'), SimpleNamespace(version='2')]
    stop_after_two_polls(monkeypatch, deploy)
    with pytest.raises(StopPolling):
        autoserve.main()
    deploy.assert_called_once()
    assert deploy.call_args.kwargs['model_name'] == 'iris'
    assert deploy.call_args.kwargs['version'] == '2'
    log.exception.assert_called_once()
    docker.containers.remove.assert_not_called()
    docker.containers.run.assert_not_called()


def test_partial_alias_scan_is_not_reconciled_before_next_complete_poll(loop, monkeypatch):
    registry, _, deploy, log = loop
    registry.search_registered_models.side_effect = [
        [SimpleNamespace(name='iris'), SimpleNamespace(name='other')],
        [SimpleNamespace(name='iris')]]
    registry.get_model_version_by_alias.side_effect = [
        SimpleNamespace(version='2'), ConnectionError('offline'), SimpleNamespace(version='2')]
    stop_after_two_polls(monkeypatch, deploy)
    with pytest.raises(StopPolling):
        autoserve.main()
    deploy.assert_called_once()
    log.exception.assert_called_once()


def test_failed_alias_scan_does_not_prevent_other_alias_processing(loop, monkeypatch):
    registry, _, deploy, log = loop
    monkeypatch.setenv('MLFLOW_SERVE_ALIASES', 'champion,challenger')
    registry.search_registered_models.side_effect = [ConnectionError('offline'), [SimpleNamespace(name='iris')]]
    registry.get_model_version_by_alias.return_value = SimpleNamespace(version='1')
    def stop(_):
        raise StopPolling()
    monkeypatch.setattr(autoserve.time, 'sleep', stop)
    with pytest.raises(StopPolling):
        autoserve.main()
    deploy.assert_called_once()
    assert deploy.call_args.kwargs['alias'] == 'challenger'
    log.exception.assert_called_once()


def test_missing_alias_is_skipped_without_hiding_registry_failures():
    registry = Mock()
    registry.search_registered_models.return_value = [SimpleNamespace(name='iris')]
    registry.get_model_version_by_alias.side_effect = mlflow.exceptions.MlflowException(
        'Alias not found', error_code=RESOURCE_DOES_NOT_EXIST)
    assert list(autoserve._iter_models_with_alias(registry, 'challenger')) == []
    registry.get_model_version_by_alias.side_effect = mlflow.exceptions.MlflowException('Registry unavailable')
    with pytest.raises(mlflow.exceptions.MlflowException, match='Registry unavailable'):
        list(autoserve._iter_models_with_alias(registry, 'champion'))
