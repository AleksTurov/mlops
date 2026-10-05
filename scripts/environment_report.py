"""Record the measurement host and base image without secrets or server mutations."""
import argparse
import hashlib
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def command(args):
    try:
        result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=30)
        return result.stdout.strip() if result.returncode == 0 else 'unavailable'
    except (OSError, subprocess.TimeoutExpired):
        return 'unavailable'


def report(output, base_image):
    cpu = 'unavailable'
    try:
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('model name'):
                cpu = line.split(':', 1)[1].strip()
                break
    except OSError:
        pass
    rows = [f'Captured UTC: {datetime.now(timezone.utc).isoformat()}',
            f'OS: {platform.platform()}', f'Architecture: {platform.machine()}',
            f'CPU: {cpu}', f'Logical CPUs: {os.cpu_count()}',
            f'Process CPU affinity: {sorted(os.sched_getaffinity(0))}',
            f'Host RAM bytes: {os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")}',
            f'Host load average: {os.getloadavg()}', f'Python: {platform.python_version()}',
            f'Autoserve git commit: {command(["git", "rev-parse", "HEAD"])}',
            f'Git status: {command(["git", "status", "--short"])}',
            f'Autoserve source SHA256: {hashlib.sha256((ROOT / "src/app/mlflow_autoserve.py").read_bytes()).hexdigest()}',
            f'Docker: {command(["docker", "version", "--format", "{{json .}}"])}',
            f'Docker Compose: {command(["docker", "compose", "version"])}',
            f'Base image: {base_image}',
            f'Base image ID/digests: {command(["docker", "image", "inspect", base_image, "--format", "{{.Id}} {{json .RepoDigests}}"])}',
            'Benchmark CPU/memory limits: none; GPU disabled; shared host',
            'Registry: private SQLite; PostgreSQL/MinIO/Prometheus/Grafana not used by measurements']
    for package in ('mlflow', 'docker', 'requests', 'numpy', 'scikit-learn', 'pytest', 'matplotlib'):
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = 'unavailable'
        rows.append(f'Host package {package}: {version}')
    runtime = 'import platform,importlib.metadata as m; print("Python="+platform.python_version()); [print(p+"="+m.version(p)) for p in ("mlflow","scikit-learn","numpy","requests")]'
    rows.append('Base image runtime:\n' + command(['docker', 'run', '--rm', '--network', 'none',
                '--label', 'mlops_environment_report=true', '--entrypoint', 'python', base_image, '-c', runtime]))
    # Inventory configured images only; never exec into or restart server containers.
    try:
        config = json.loads(command(['docker', 'compose', 'config', '--format', 'json']))
        rows.append('Configured demo stack images (not benchmark services):')
        for name in ('mlflow-db', 'minio', 'prometheus', 'grafana'):
            image = config['services'][name]['image']
            identity = command(['docker', 'image', 'inspect', image, '--format', '{{.Id}} {{json .RepoDigests}}'])
            rows.append(f'{name}: {image}; {identity}')
    except (ValueError, KeyError):
        rows.append('Configured stack inventory: unavailable')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('\n'.join(rows) + '\n')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/tmp/mlops-environment.txt'))
    parser.add_argument('--base-image', default='mlops-demo-mlflow-autoserve')
    args = parser.parse_args()
    print(report(args.output, args.base_image))
