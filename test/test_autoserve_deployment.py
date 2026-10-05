from types import SimpleNamespace
from unittest.mock import Mock

import docker
import pytest
from app import mlflow_autoserve as a


class Container:
    def __init__(self, pool, name, version, image='image'):
        self.pool, self.name, self.status = pool, name, 'running'
        self.labels = {'mlflow_version': version, 'mlflow_port': '8080'}
        self.image = image
        self.removed = False
        self.stopped = False

    def reload(self):
        pass

    def rename(self, name):
        if name in self.pool.items:
            raise docker.errors.APIError('name already exists')
        self.pool.items.pop(self.name)
        self.name = name
        self.pool.items[name] = self

    def remove(self, force=False):
        self.removed = True
        self.pool.items.pop(self.name, None)

    def stop(self, **kwargs):
        self.stopped = True
        self.status = 'exited'


class Pool:
    def __init__(self):
        self.items = {}
        self.run_count = 0

    def get(self, name):
        if name not in self.items:
            raise docker.errors.NotFound(name)
        return self.items[name]

    def run(self, **kwargs):
        self.run_count += 1
        container = Container(self, kwargs['name'], kwargs['labels']['mlflow_version'])
        container.labels = kwargs['labels']
        if container.name in self.items:
            raise docker.errors.APIError('duplicate name')
        self.items[container.name] = container
        return container


@pytest.fixture
def setup(monkeypatch):
    monkeypatch.setenv('MLFLOW_SERVE_ENABLE_GPU', 'false')
    pool = Pool()
    old = Container(pool, 'mlflow-serve-model-champion', '1')
    pool.items[old.name] = old
    dc = SimpleNamespace(containers=pool)
    registry = Mock()
    registry.get_model_version_by_alias.return_value = SimpleNamespace(version='2')
    monkeypatch.setattr(a, 'MlflowClient', lambda: registry)
    monkeypatch.setattr(a, '_build_image_name', lambda *args: 'image')
    monkeypatch.setattr(a, '_image_exists', lambda *args: True)
    monkeypatch.setattr(a, '_wait_ready', lambda *args: None)
    return dc, old, registry


def deploy(dc, alias='champion', version='2'):
    return a._ensure_container(dc, 'model', alias, version, 'test', 'image', 'network', 5000,
                               {'MLFLOW_MODELS_WORKERS': '1'}, 'docker-image', 'local', 1, 0, '')


def test_build_failure_keeps_serving_and_restores_alias(setup, monkeypatch):
    dc, old, registry = setup
    monkeypatch.setattr(a, '_image_exists', lambda *args: False)
    def build(**kwargs):
        assert not old.removed and not old.stopped
        raise RuntimeError('build failed')
    monkeypatch.setattr(a, '_build_model_image_with_retries', build)
    with pytest.raises(RuntimeError, match='build failed'):
        deploy(dc)
    assert dc.containers.get(old.name) is old
    assert not old.removed and not old.stopped and dc.containers.run_count == 0
    registry.set_registered_model_alias.assert_called_once_with('model', 'champion', '1')


def test_candidate_ready_before_old_is_replaced(setup, monkeypatch):
    dc, old, _ = setup
    calls = []
    def ready(candidate, *args):
        calls.append(candidate.name)
        assert not old.stopped and not old.removed
        assert old.status == 'running'
    monkeypatch.setattr(a, '_wait_ready', ready)
    deploy(dc)
    assert calls == ['mlflow-serve-model-champion-candidate', 'mlflow-serve-model-champion']
    assert old.removed
    assert dc.containers.get('mlflow-serve-model-champion').labels['mlflow_version'] == '2'


def test_startup_failure_leaves_original_untouched(setup, monkeypatch):
    dc, old, registry = setup
    monkeypatch.setattr(a, '_wait_ready', Mock(side_effect=TimeoutError('not ready')))
    with pytest.raises(TimeoutError):
        deploy(dc)
    assert dc.containers.get('mlflow-serve-model-champion') is old
    assert not old.removed and not old.stopped
    assert len(dc.containers.items) == 1


def test_failed_handover_restores_original_name(setup, monkeypatch):
    dc, old, _ = setup
    def ready(candidate, *args):
        if candidate is not old and candidate.name == 'mlflow-serve-model-champion':
            raise TimeoutError('handover failed')
    monkeypatch.setattr(a, '_wait_ready', ready)
    with pytest.raises(TimeoutError):
        deploy(dc)
    assert dc.containers.get('mlflow-serve-model-champion') is old
    assert not old.removed and not old.stopped


def test_failure_does_not_overwrite_newer_alias(setup, monkeypatch):
    dc, old, registry = setup
    registry.get_model_version_by_alias.return_value = SimpleNamespace(version='3')
    monkeypatch.setattr(a, '_wait_ready', Mock(side_effect=TimeoutError('not ready')))
    with pytest.raises(TimeoutError):
        deploy(dc)
    registry.set_registered_model_alias.assert_not_called()


def test_alias_restore_failure_keeps_original_container(setup, monkeypatch):
    dc, old, registry = setup
    registry.set_registered_model_alias.side_effect = RuntimeError('registry offline')
    monkeypatch.setattr(a, '_wait_ready', Mock(side_effect=TimeoutError('not ready')))
    with pytest.raises(TimeoutError):
        deploy(dc)
    assert not old.removed


def test_real_probe_timeout(monkeypatch):
    container = SimpleNamespace(name='unhealthy', status='restarting', reload=lambda: None, attrs={'NetworkSettings': {'Networks': {'network': {'IPAddress': ''}}}})
    monkeypatch.setenv('MLFLOW_SERVE_READY_TIMEOUT_SECONDS', '.01')
    monkeypatch.setattr(a.time, 'sleep', lambda _: None)
    with pytest.raises(TimeoutError, match='/ping'):
        a._wait_ready(container, 'network', 8080)


def test_superseded_candidate_is_not_activated(setup, monkeypatch):
    dc, old, registry = setup
    registry.get_model_version_by_alias.return_value = SimpleNamespace(version='3')
    with pytest.raises(RuntimeError, match='superseded'):
        deploy(dc)
    assert dc.containers.get('mlflow-serve-model-champion') is old
    assert len(dc.containers.items) == 1
    registry.set_registered_model_alias.assert_not_called()


def test_failed_docker_start_cleans_partially_created_candidate(setup, monkeypatch):
    dc, old, _ = setup
    real_run = dc.containers.run
    def run(**kwargs):
        real_run(**kwargs)
        raise docker.errors.APIError('cannot start')
    monkeypatch.setattr(dc.containers, 'run', run)
    with pytest.raises(docker.errors.APIError):
        deploy(dc)
    assert len(dc.containers.items) == 1
    assert dc.containers.get('mlflow-serve-model-champion') is old
