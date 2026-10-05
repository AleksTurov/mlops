"""Repeat the existing isolated runtime check and aggregate its measured results."""
import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile

from environment_report import report

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ('successful_replacement', 'failed_build', 'failed_start')


def summarize(rows):
    summary = {}
    for scenario in SCENARIOS:
        selected = [r for r in rows if r['scenario'] == scenario]
        item = {'N': len(selected),
                'deployment_success_rate': sum(r['deployment_success'] for r in selected) / len(selected) if selected else None,
                'expected_behavior_pass_rate': sum(r['expected_behavior_passed'] for r in selected) / len(selected) if selected else None}
        for field in ('deployment_seconds', 'scenario_check_seconds', 'sampled_unavailable_seconds'):
            values = sorted(r[field] for r in selected)
            item[field] = {'N': len(values), 'mean': statistics.mean(values), 'median': statistics.median(values),
                           'P95': values[math.ceil(.95 * len(values)) - 1], 'min': min(values), 'max': max(values)} if values else None
        summary[scenario] = item
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repetitions', type=int, default=20)
    parser.add_argument('--base-image', default='mlops-demo-mlflow-autoserve')
    parser.add_argument('--ready-timeout', type=float, default=15)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.repetitions < 1 or args.ready_timeout <= 0:
        parser.error('repetitions and readiness timeout must be positive')
    output = args.output.resolve() if args.output else Path(tempfile.mkdtemp(prefix='mlops-measurements-'))
    if output.exists() and any(output.iterdir()):
        parser.error('output directory must be empty to preserve measurements')
    output.mkdir(parents=True, exist_ok=True)
    report(output / 'environment.txt', args.base_image)
    metadata = {'started_at': datetime.now(timezone.utc).isoformat(), 'repetitions_requested': args.repetitions,
                'base_image': args.base_image, 'ready_timeout_seconds': args.ready_timeout,
                'sampling_interval_seconds': .1, 'polling_wait_included': False,
                'timing_window': 'before availability-monitor startup and alias assignment through controller return or failure after alias restoration',
                'scenario_check_seconds': 'also includes validation, sampling tail, and successful-case DNS/prediction checks',
                'successful_path': 'new absent model image; prepared base and Docker build cache retained',
                'failed_build': 'missing artifact; error occurs before Docker build executes',
                'failed_start': 'actual cached image without model, startup/readiness failure',
                'N_per_iteration': 1, 'deployment_success_for_injected_failures': False,
                'availability_method': 'Docker API resolves current stable name to IP; /ping samples at nominal 0.1s intervals; DNS checked after successful replacement'}
    (output / 'measurement_config.json').write_text(json.dumps(metadata, indent=2))
    rows = []

    def save():
        if rows:
            with (output / 'benchmark_results.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        (output / 'benchmark_summary.json').write_text(json.dumps(summarize(rows), indent=2))

    print(f'Measurements directory: {output}', flush=True)
    for iteration in range(1, args.repetitions + 1):
        directory = output / f'run-{iteration:02d}'
        command = [sys.executable, str(ROOT / 'scripts/check_deployment.py'), '--base-image', args.base_image,
                   '--ready-timeout', str(args.ready_timeout), '--output', str(directory)]
        with (output / f'run-{iteration:02d}.log').open('w') as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        path = directory / 'result.json'
        if path.exists():
            run = json.loads(path.read_text())
            for scenario in run['scenarios']:
                rows.append({'iteration': iteration, 'scenario': scenario['scenario'],
                             'deployment_seconds': scenario['deployment_seconds'],
                             'scenario_check_seconds': scenario['elapsed_seconds'],
                             'deployment_success': scenario['scenario'] == 'successful_replacement' and scenario['passed'],
                             'expected_behavior_passed': scenario['passed'],
                             'sampled_unavailable_seconds': scenario['sampled_unavailable_seconds'],
                             'ping_samples': scenario['ping_samples'], 'failed_ping_samples': scenario['failed_ping_samples'],
                             'healthy_old_during_candidate_samples': scenario['healthy_old_during_candidate_samples'],
                             'error': scenario.get('error'), 'source_sha256': run['source_sha256'],
                             'git_commit': run['commit'], 'base_image_id': run['base_image_id'],
                             'test_resources_removed': run.get('test_resources_removed', False)})
        save()
        if result.returncode or not path.exists() or not run.get('passed') or not run.get('test_resources_removed'):
            raise RuntimeError(f'Iteration {iteration} failed; partial results and log retained at {output}')
        print(f'Completed {iteration}/{args.repetitions}; {len(rows)} scenario measurements', flush=True)
    metadata['finished_at'] = datetime.now(timezone.utc).isoformat()
    (output / 'measurement_config.json').write_text(json.dumps(metadata, indent=2))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.boxplot([[r['deployment_seconds'] for r in rows if r['scenario'] == s] for s in SCENARIOS], tick_labels=SCENARIOS)
        ax.set_ylabel('Controller completion / failure containment (seconds)')
        fig.tight_layout()
        fig.savefig(output / 'deployment_latency.png', dpi=180)
        plt.close(fig)
    except ImportError:
        print('matplotlib unavailable; raw data and summary are saved', flush=True)
    print(json.dumps(summarize(rows), indent=2))


if __name__ == '__main__':
    main()
