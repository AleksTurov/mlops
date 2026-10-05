"""Check real replacement and failure containment in an isolated Docker environment."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import docker
import mlflow
import requests
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from app import mlflow_autoserve as controller

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-image', default=os.getenv('MLFLOW_SERVE_BUILD_BASE_IMAGE', 'mlops-demo-mlflow-autoserve'))
    parser.add_argument('--output', type=Path, help='Fresh result directory; default: a temporary directory')
    parser.add_argument('--ready-timeout', type=float, default=15)
    args = parser.parse_args()
    if args.ready_timeout <= 0:
        parser.error('ready-timeout must be positive')
    root=ROOT
    work=args.output.resolve() if args.output else Path(tempfile.mkdtemp(prefix='mlops-minimal-check-'))
    if work.exists() and any(work.iterdir()):
        parser.error('output directory must be empty to preserve earlier measurements')
    work.mkdir(parents=True, exist_ok=True)
    check_id=uuid.uuid4().hex[:8]
    model='minimal_check_'+check_id
    network_name='mlops-minimal-check-'+check_id
    name=controller._sanitize_name(f'mlflow-serve-{model}-champion')
    base=args.base_image
    os.environ['MLFLOW_SERVE_BUILD_BASE_IMAGE']=base
    os.environ['MLFLOW_SERVE_ENABLE_GPU']='false'
    os.environ['MLFLOW_SERVE_READY_TIMEOUT_SECONDS']=str(args.ready_timeout)
    mlflow.set_tracking_uri('sqlite:///' + str(work/'registry.db'))
    registry=MlflowClient()
    experiment=registry.create_experiment(model,artifact_location=(work/'artifacts').as_uri())
    mlflow.set_experiment(experiment_id=experiment)
    real=docker.from_env()
    real.images.get(base)
    network=None

    # Isolate ownership labels only: the old server's global janitor selects mlflow_serve=true.
    # All lifecycle operations, builds, readiness and registry calls below remain real.
    class Containers:
        def __init__(self):
            self.collection=real.containers
        def get(self,*args,**kwargs):
            return self.collection.get(*args,**kwargs)
        def run(self,**kwargs):
            kwargs['labels']=dict(kwargs['labels'],mlflow_serve='minimal-check',minimal_check=check_id)
            return self.collection.run(**kwargs)

    class Client:
        images=real.images
        containers=Containers()

    client=Client()
    images=set()
    results={'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root).decode().strip(),
             'source_sha256':hashlib.sha256((root/'src/app/mlflow_autoserve.py').read_bytes()).hexdigest(),
             'base_image':base,'base_image_id':real.images.get(base).id,
             'test_type':'real Docker + private SQLite MLflow registry, direct _ensure_container calls',
             'isolation':'unique network and model; test-only serving ownership label',
             'sampling_interval_seconds':.1,'readiness_timeout_seconds':args.ready_timeout,'scenarios':[]}
    result_file=work/'result.json'
    print(f'Results directory: {work}', flush=True)
    peer=None
    versions=[]


    def save():
        result_file.write_text(json.dumps(results,indent=2))


    def get_serving():
        try:
            c=real.containers.get(name)
            c.reload()
            ip=c.attrs['NetworkSettings']['Networks'][network_name]['IPAddress']
            return c,f'http://{ip}:8080'
        except docker.errors.NotFound:
            return None,None


    def ping():
        c,url=get_serving()
        try:
            return c is not None and requests.get(url+'/ping',timeout=1).status_code==200
        except requests.RequestException:
            return False


    def deploy(version):
        image=controller._build_image_name(client,model,str(version))
        images.add(image)
        controller._ensure_container(client,model,'champion',str(version),network_name,base,network_name,
                                     5000,{'MLFLOW_MODELS_WORKERS':'1','DISABLE_NGINX':'true'},
                                     'docker-image','local',1,0,'')


    class Monitor:
        def __init__(self):
            self.done=threading.Event()
            self.samples=[]
            self.overlap=0
            self.thread=threading.Thread(target=self.run,daemon=True)
        def run(self):
            while not self.done.is_set():
                ok=ping()
                self.samples.append((time.monotonic(),ok))
                try:
                    real.containers.get(name+'-candidate')
                    self.overlap+=int(ok)
                except docker.errors.NotFound:
                    pass
                self.done.wait(.1)
        def __enter__(self):
            self.thread.start()
            return self
        def __exit__(self,*args):
            self.done.set()
            self.thread.join(timeout=3)
        def measurements(self):
            return {'ping_samples':len(self.samples),'failed_ping_samples':sum(not ok for _,ok in self.samples),
                    'healthy_old_during_candidate_samples':self.overlap,
                    'sampled_unavailable_seconds':sum(b[0]-a[0] for a,b in zip(self.samples,self.samples[1:]) if not a[1])}

    try:
        network=real.networks.create(network_name,labels={'minimal_check':check_id})
        iris=load_iris()
        registry.create_registered_model(model)
        for c in (1.,.5):
            fitted=LogisticRegression(C=c,max_iter=200).fit(iris.data,iris.target)
            with mlflow.start_run() as run:
                info=mlflow.sklearn.log_model(fitted,name='iris',input_example=iris.data[:2],pip_requirements=['scikit-learn==1.5.2'])
                versions.append(str(registry.create_model_version(model,info.model_uri,run_id=run.info.run_id).version))
        registry.set_registered_model_alias(model,'champion',versions[0])
        deploy(versions[0])
        print('Initial version serving',flush=True)
        first,_=get_serving()
        peer=real.containers.run(base,name='minimal-check-peer-'+check_id,entrypoint=['python'],
            command=['-c','import time; time.sleep(3600)'],detach=True,network=network_name,
            labels={'minimal_check':check_id})
        started=time.monotonic()
        with Monitor() as monitor:
            registry.set_registered_model_alias(model,'champion',versions[1])
            deploy(versions[1])
            time.sleep(.2)
        active,url=get_serving()
        assert active.id!=first.id and active.labels['mlflow_version']==versions[1] and ping()
        try:
            real.containers.get(first.id)
            raise AssertionError('old container retained after successful replacement')
        except docker.errors.NotFound:
            pass
        payload={'inputs':iris.data[:2].tolist()}
        code=('import requests,json,socket; '+f'host="{name}"; '+
              'r=requests.get("http://"+host+":8080/ping",timeout=3); assert r.status_code==200; '+
              f'r=requests.post("http://"+host+":8080/invocations",json={payload!r},timeout=10); '+
              'r.raise_for_status(); print(json.dumps({"dns_ip":socket.gethostbyname(host),"prediction":r.json()}))')
        dns=peer.exec_run(['python','-c',code])
        assert dns.exit_code==0,dns.output.decode()
        results['scenarios'].append(dict(scenario='successful_replacement',passed=True,
            elapsed_seconds=time.monotonic()-started,serving_version=versions[1],dns_and_prediction=json.loads(dns.output),
            **monitor.measurements()))
        save()
        print('PASS: successful replacement, Docker DNS, real predictions',flush=True)

        stable_id=active.id
        failed_build=str(registry.create_model_version(model,str(work/'missing-model')).version)
        versions.append(failed_build)
        started=time.monotonic()
        error=None
        with Monitor() as monitor:
            registry.set_registered_model_alias(model,'champion',failed_build)
            try:
                deploy(failed_build)
            except Exception as exc:
                error=f'{type(exc).__name__}: {exc}'
            time.sleep(.2)
        current,_=get_serving()
        assert error and current.id==stable_id and ping()
        assert str(registry.get_model_version_by_alias(model,'champion').version)==versions[1]
        results['scenarios'].append(dict(scenario='failed_build',passed=True,
            elapsed_seconds=time.monotonic()-started,error=error,old_container_unchanged=True,
            alias_restored_to=versions[1],**monitor.measurements()))
        save()
        print('PASS: image construction failure retains old container and restores alias',flush=True)

        valid=registry.get_model_version(model,versions[1])
        failed_start=str(registry.create_model_version(model,valid.source,run_id=valid.run_id).version)
        versions.append(failed_start)
        bad_image=controller._build_image_name(client,model,failed_start)
        images.add(bad_image)
        # Real cached image missing its model: image construction succeeds, serving startup fails.
        real.images.build(fileobj=io.BytesIO(f'FROM {base}\nRUN mkdir -p /opt/ml/model\n'.encode()),tag=bad_image,rm=True)
        started=time.monotonic()
        error=None
        with Monitor() as monitor:
            registry.set_registered_model_alias(model,'champion',failed_start)
            try:
                deploy(failed_start)
            except Exception as exc:
                error=f'{type(exc).__name__}: {exc}'
            time.sleep(.2)
        current,_=get_serving()
        assert error and current.id==stable_id and ping()
        assert str(registry.get_model_version_by_alias(model,'champion').version)==versions[1]
        try:
            real.containers.get(name+'-candidate')
            raise AssertionError('failed candidate was not removed')
        except docker.errors.NotFound:
            pass
        results['scenarios'].append(dict(scenario='failed_start',passed=True,
            elapsed_seconds=time.monotonic()-started,error=error,old_container_unchanged=True,
            alias_restored_to=versions[1],failed_candidate_removed=True,**monitor.measurements()))
        results['passed']=True
        save()
        print('PASS: startup failure retains old container and restores alias',flush=True)
    except Exception as exc:
        results['passed']=False
        results['test_error']=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        cleanup_errors = []
        for container in real.containers.list(all=True, filters={'label': 'minimal_check='+check_id}):
            try:
                container.remove(force=True)
            except docker.errors.DockerException as exc:
                cleanup_errors.append(str(exc))
        if network is not None:
            try:
                network.remove()
            except docker.errors.DockerException as exc:
                cleanup_errors.append(str(exc))
        for image in images:
            try:
                real.images.remove(image, force=True)
            except docker.errors.ImageNotFound:
                pass
            except docker.errors.DockerException as exc:
                cleanup_errors.append(str(exc))
        results['test_resources_removed'] = not cleanup_errors
        results['cleanup_errors'] = cleanup_errors
        save()
    if cleanup_errors:
        raise RuntimeError('Test resource cleanup failed; see result.json')
    print(json.dumps(results,indent=2),flush=True)
    print(f'Report: {result_file}')


if __name__ == '__main__':
    main()
