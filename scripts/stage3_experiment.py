"""Fixed-budget counterfactual diagnostic with actual train/calibrate/load/exposure path."""
from __future__ import annotations
import argparse
import contextlib
import copy
import json
from pathlib import Path
import platform
import sys
import time
import torch
from kev_laya.checkpoint import load_checkpoint
from kev_laya.cli import main as cli
from kev_laya.counterfactual import PROTOCOL, freeze_counterfactual, regression_data
from kev_laya.data import load_suite
from kev_laya.encoding import ByteTokenizer, Limits
from kev_laya.evaluation import evaluate
from kev_laya.exposure import verify_exposure
from kev_laya.io import atomic_json, sha256_file
from kev_laya.model import BackboneConfig, DecisionEngine
from kev_laya.objectives import ObjectiveConfig
from kev_laya.provenance import hardware_identity
from kev_laya.registration import register
from kev_laya.telemetry import TelemetryWriter, artifact
from kev_laya.training import TrainSettings, train


def run(output: Path):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('refusing to overwrite recorded experiment')
    output.mkdir(parents=True, exist_ok=True)
    protocol = copy.deepcopy(PROTOCOL)
    protocol['script_sha256'] = sha256_file(Path(__file__))
    atomic_json(output / 'protocol.json', protocol)  # No model output has been inspected.
    freeze_counterfactual(output / 'suite')
    suite, manifest = load_suite(output / 'suite')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(protocol['seed'])
    model, tokenizer = DecisionEngine(BackboneConfig()), ByteTokenizer()
    limits = Limits(branch=512, aggregate=8192, max_questions=64)
    started = time.perf_counter()
    settings = TrainSettings(steps=protocol['sft_steps'], accumulation=protocol['accumulation'],
                             learning_rate=protocol['learning_rate'], seed=protocol['seed'],
                             choice_permutation=True, save_every=40)
    runs = {}
    runs['supervised-160'] = train(model, tokenizer, suite['train'], manifest, output / 'supervised-160',
         settings, ObjectiveConfig(ordinal=protocol['ordinal_coefficient']), limits,
         experiment_id=protocol['id'], run_id='supervised-160')
    for label, coefficient in (('supervised-200', 0.), ('pgps-200', protocol['pgps_coefficient'])):
        arm, tok, point = load_checkpoint(Path(runs['supervised-160']['checkpoint']))
        runs[label] = train(arm, tok, suite['train'], manifest, output / label,
            TrainSettings(steps=protocol['matched_extension_steps'], accumulation=protocol['accumulation'],
                          learning_rate=protocol['extension_learning_rate'], seed=protocol['seed'],
                          choice_permutation=True, save_every=40), ObjectiveConfig(ordinal=.1, reinforce=coefficient),
            limits, experiment_id=protocol['id'], run_id=label, parent_sha256=point['checkpoint_sha256'])
    # All weight-training arms have finished; no test-based selection or budget extension.
    calibrated = output / 'pgps-200' / 'calibrated.pt'
    with (output / 'calibration-cli.log').open('w', encoding='utf-8') as log, contextlib.redirect_stdout(log):
        cli(['calibrate', '--checkpoint', runs['pgps-200']['checkpoint'], '--suite', str(output / 'suite'),
             '--out', str(calibrated), '--experiment-id', protocol['id'], '--run-id', 'pgps-200-calibration'])
    reports = {}
    checkpoints = {k: Path(v['checkpoint']) for k, v in runs.items()} | {'pgps-200-calibrated': calibrated}
    for label, checkpoint in checkpoints.items():
        evaluated, tok, point = load_checkpoint(checkpoint)
        report = evaluate(evaluated, suite['test'], tok, limits, split='test', diagnostic=True)
        report.update(checkpoint_sha256=point['checkpoint_sha256'],
                      split_sha256=manifest['partitions']['test']['sha256'], protocol_sha256=sha256_file(output/'protocol.json'))
        path = output / (label + '-evaluation.json')
        atomic_json(path, report)
        reports[label] = report
        if label in runs:
            snapshot_path = Path(runs[label]['snapshot'])
            old = json.loads(snapshot_path.read_text())
            writer = TelemetryWriter(snapshot_path, old)
            metrics = old['metrics'] | {'validation/' + k: report['summary'][k] for k in ('nll', 'brier', 'ece', 'accuracy', 'ordinal_mae')}
            writer.update(phase='completed', metrics=metrics, artifacts=old['artifacts']+[artifact(path, 'evaluation')])
            register(output/'registry.json', snapshot_path, label)
    final, tok, point = load_checkpoint(calibrated)
    regression = evaluate(final, regression_data(), tok, limits, split='observed-regression')
    atomic_json(output / 'known-regressions.json', regression)
    # This is a documented old regression, not a held-out test or checkpoint selector.
    known = [r for r in regression['rows'] if r['record_id']=='known-regression/99999/red/1/red']
    exposure = verify_exposure(calibrated, expected_sha256=point['checkpoint_sha256'])
    atomic_json(output / 'calibrated-exposure.json', exposure)
    result = {'protocol_sha256': sha256_file(output/'protocol.json'), 'elapsed_seconds': time.perf_counter()-started,
              'environment': {'python':platform.python_version(), 'torch':torch.__version__, 'device':hardware_identity(torch.device('cpu')),
                              'dtype':'fp32', 'threads':torch.get_num_threads(), 'cuda_available':torch.cuda.is_available()},
              'checkpoints': {k:str(v) for k,v in checkpoints.items()}, 'training_runs':runs,
              'summaries': {k:v['summary'] for k,v in reports.items()},
              'fixture_threshold_pass': reports['pgps-200-calibrated']['summary']['accuracy'] >= protocol['accuracy_threshold'],
              'known_regression': known, 'regression_summary':regression['summary'], 'exposure': exposure,
              'old_short_smoke': protocol['prior_failure_preserved'],
              'actual_cost_usd':None, 'cost_reason':'measured CPU runtime; no billing or energy meter',
              'native_context':'unverified', 'general_predictive_quality':'unverified', 'live_overwatch':'unverified'}
    atomic_json(output/'experiment-report.json', result)
    print(json.dumps({'elapsed_seconds':result['elapsed_seconds'],
                      'accuracy':{k:v['summary']['accuracy'] for k,v in reports.items()},
                      'fixture_threshold_pass':result['fixture_threshold_pass'],
                      'known_regression':known, 'exposure_status':exposure['status']},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args().out)
