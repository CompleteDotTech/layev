"""Identical inputs/weights; a fresh process per case/mode; no speedup threshold."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import threading
import time
import torch
from kev_laya.checkpoint import load_checkpoint
from kev_laya.encoding import ByteTokenizer, Limits, encode_request
from kev_laya.execution import BatchPolicy
from kev_laya.model import BackboneConfig, DecisionEngine
from kev_laya.schema import SystemOneRequest, canonical


def make_request(count, distribution):
    qs = {}
    for i in range(count):
        kind = ('choice', 'score', 'noul')[i % 3] if distribution == 'ragged' else 'choice'
        q = {'type': kind, 'instructions': 'select value' + (' x' * ((i % 4) * 13) if distribution == 'ragged' else '')}
        if kind == 'choice':
            q['criteria'] = {str(j): None for j in range(2 + (i % 4 if distribution == 'ragged' else 0))}
        elif kind == 'score':
            q['criteria'] = [str(j) for j in range(2 + i % 7)]
        else:
            q['criteria'] = {'false':'absent', 'true':'present'}
        qs[f'id-{i}'] = q
    return SystemOneRequest(state='color=red; level=1; case=99999', questions=qs, model='kev-laya-preview')


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def worker(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(47)
    device = torch.device(args.device)
    if args.checkpoint:
        model, tokenizer, payload = load_checkpoint(args.checkpoint, args.device)
        identity = payload['model_id']
    else:
        model, tokenizer = DecisionEngine(BackboneConfig()).to(device), ByteTokenizer()
        identity = 'random-fixture-seed47'
    model.eval()
    policy = BatchPolicy(max_branches=8, max_padded_tokens=4096)
    encoding = encode_request(make_request(args.count, args.distribution), tokenizer,
                              Limits(model.cfg.max_position_embeddings, 65536))
    kwargs = {'policy':policy, 'reference':args.mode=='full_row_reference',
              'serial_reference':args.mode=='serial_cached_reference'}
    with torch.inference_mode():
        for _ in range(args.warmups): model(encoding, **kwargs)
        sync(device)
        if device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device)
        try:
            import psutil
            proc = psutil.Process()
            rss_start = proc.memory_info().rss
        except ImportError:
            proc, rss_start = None, None
        samples, stop = [], threading.Event()
        def sample():
            while not stop.is_set():
                samples.append(proc.memory_info().rss)
                stop.wait(.002)
        monitor = threading.Thread(target=sample) if proc else None
        if monitor: monitor.start()
        durations = []
        try:
            for _ in range(args.repetitions):
                sync(device)
                started = time.perf_counter_ns()
                model(encoding, **kwargs)
                sync(device)
                durations.append((time.perf_counter_ns()-started)/1e6)
        finally:
            stop.set()
            if monitor: monitor.join()
        observed = []
        def capture(_module, call, kw):
            ids = call[0]
            has_prefix = len(call)>1 and call[1] is not None
            observed.append({'batch_size':int(ids.shape[0]) if torch.is_tensor(ids) else 1,
                             'length':int(ids.shape[1]) if torch.is_tensor(ids) else len(ids),
                             'cached_prefix':has_prefix})
        handle = model.backbone.register_forward_pre_hook(capture, with_kwargs=True)
        logits, stats = model(encoding, **kwargs)
        handle.remove()
        assert stats['branch_passes'] + stats['prefix_passes'] == len(observed)
        assert [r['batch_size'] for r in observed if r['cached_prefix']] == ([] if args.mode=='full_row_reference' else stats['effective_batch_sizes'])
        try:
            import resource
            rss_hwm = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform=='darwin' else 1024)
        except ImportError:
            rss_hwm = None
        probs=[(z.double()/model.temperatures[b.question.type]).softmax(-1).cpu().tolist()
               for z,b in zip(logits,encoding.branches)]
        return {'mode':args.mode,'count':args.count,'distribution':args.distribution,'model_id':identity,
                'device':str(device),'dtype':str(next(model.parameters()).dtype),'torch':torch.__version__,
                'python':platform.python_version(),'threads':args.threads,
                'native_weights_loaded':model.native_weights_loaded,
                'input_sha256':hashlib.sha256(canonical({'state':list(encoding.state),'branches':[list(b.ids) for b in encoding.branches]}).encode()).hexdigest(),
                'state_tokens':len(encoding.state),'branch_lengths':[len(b.ids) for b in encoding.branches],
                'latency_ms':{'median':statistics.median(durations),'min':min(durations),'max':max(durations),'samples':durations},
                'rss_before_bytes':rss_start,'rss_peak_sampled_bytes':max(samples) if samples else None,
                'process_peak_rss_bytes':rss_hwm,'rss_method':'fresh process per case/mode; RSS sampled every 2ms after warmup; short peaks may be missed; process HWM includes load and warmup',
                'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
                'cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved(device) if device.type=='cuda' else None,
                'execution':stats,'observed_backbone_calls':observed,
                'logits':[z.cpu().tolist() for z in logits],'probabilities':probs}


def diff(a,b):
    pairs=[(x,y) for xs,ys in zip(a,b,strict=True) for x,y in zip(xs,ys,strict=True)]
    return {'max_absolute':max(abs(x-y) for x,y in pairs),
            'max_tolerance_ratio':max(abs(x-y)/(1e-5+1e-5*abs(y)) for x,y in pairs)}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--warmups',type=int,default=3)
    parser.add_argument('--repetitions',type=int,default=10)
    parser.add_argument('--counts',nargs='+',type=int,default=[1,4,16,32])
    parser.add_argument('--distributions',nargs='+',choices=['uniform','ragged'],default=['uniform','ragged'])
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--mode',default='batched')
    parser.add_argument('--count',type=int,default=4)
    parser.add_argument('--distribution',default='uniform')
    args=parser.parse_args()
    if args.threads<1 or args.repetitions<1 or args.warmups<0: parser.error('invalid measurement counts')
    if args.worker:
        print(json.dumps(worker(args),allow_nan=False));return
    if not args.out: parser.error('--out required')
    if args.out.exists(): raise FileExistsError('use a fresh benchmark report path')
    rows,comparisons=[],[]
    for distribution in args.distributions:
        for count in args.counts:
            case={}
            for mode in ['serial_cached_reference','batched','full_row_reference']:
                cmd=[sys.executable,str(Path(__file__).resolve()),'--worker','--mode',mode,'--count',str(count),
                     '--distribution',distribution,'--device',args.device,'--threads',str(args.threads),
                     '--warmups',str(args.warmups),'--repetitions',str(args.repetitions)]
                if args.checkpoint: cmd+=['--checkpoint',str(args.checkpoint.resolve())]
                run=subprocess.run(cmd,capture_output=True,text=True,timeout=180)
                if run.returncode:
                    raise RuntimeError(f'benchmark worker failed: {run.stderr}')
                row=json.loads(run.stdout)
                case[mode]=row;rows.append(row)
            a,b,c=(case[x] for x in ['batched','serial_cached_reference','full_row_reference'])
            assert a['input_sha256']==b['input_sha256']==c['input_sha256']
            logits=diff(a['logits'],c['logits']); probabilities=diff(a['probabilities'],c['probabilities'])
            assert logits['max_tolerance_ratio']<=1 and probabilities['max_tolerance_ratio']<=1
            comparisons.append({'count':count,'distribution':distribution,
                                'speedup_vs_serial_cached':b['latency_ms']['median']/a['latency_ms']['median'],
                                'speedup_vs_full_row':c['latency_ms']['median']/a['latency_ms']['median'],
                                'logits_vs_full_row':logits,'probabilities_vs_full_row':probabilities,
                                'logits_vs_serial_cached':diff(a['logits'],b['logits'])})
            print(f'{distribution} Q={count}: serial={b["latency_ms"]["median"]:.3f} ms, batch={a["latency_ms"]["median"]:.3f} ms',flush=True)
    result={'protocol':'parallel-questions-v1','evidence_class':'local-fixture-backbone',
            'native_cuda_tested':args.device.startswith('cuda'),
            'native_pretrained_tested':all(r['native_weights_loaded'] for r in rows),
            'device':args.device, 'batch_policy':BatchPolicy(max_branches=8,max_padded_tokens=4096).to_dict(),
            'warmups':args.warmups,'repetitions':args.repetitions,'rows':rows,'comparisons':comparisons}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')

if __name__=='__main__': main()
