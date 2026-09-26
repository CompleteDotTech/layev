"""Launch an actual loopback CLI server and check mixed, reordered, renamed rows.

Use PYTHONPATH pointing to an installed wheel to exercise installation separately
from source-tree tests. Only this explicit synthetic diagnostic stores responses.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import httpx
import torch
import kev_laya
from kev_laya.checkpoint import load_checkpoint
from kev_laya.encoding import Limits
from kev_laya.service import InferenceRuntime
from kev_laya.telemetry_contract import validate_snapshot
from benchmark_parallel import make_request


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    a.out.mkdir(parents=True)
    torch.set_num_threads(2)
    checkpoint=a.checkpoint.resolve();out=a.out.resolve()
    model,tok,point=load_checkpoint(checkpoint)
    runtime=InferenceRuntime(model,tok,point['model_id'],Limits(**point['provenance']['context_limits']))
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    command=[sys.executable,'-m','kev_laya','--threads','2','serve','--checkpoint',str(checkpoint),
             '--port',str(port),'--no-auth','--telemetry',str(out/'telemetry.json')]
    results=[]
    with (out/'server.log').open('w') as log:
        proc=subprocess.Popen(command,cwd=out,stdout=log,stderr=log,env=os.environ.copy())
        try:
            with httpx.Client(base_url=f'http://127.0.0.1:{port}',timeout=30) as client:
                for _ in range(200):
                    if proc.poll() is not None:raise RuntimeError('server exited; inspect server.log')
                    try:
                        if client.get('/v1/models').status_code==200:break
                    except httpx.TransportError:pass
                    time.sleep(.1)
                else:raise TimeoutError('loopback server did not become ready')
                for count in (1,3,7,20):
                    original=make_request(count,'ragged')
                    variants=[original, original.model_copy(update={'questions':dict(reversed(list(original.questions.items())))}),
                              original.model_copy(update={'questions':{f'renamed-{i}':v for i,v in enumerate(original.questions.values())}})]
                    for data in variants:
                        response=client.post('/v1/systemone',json=data.model_dump());response.raise_for_status()
                        direct,stats=runtime.predict_encoded(runtime.encode(data))
                        body=response.json()
                        assert body==direct.model_dump()
                        assert list(body['answers'])==list(data.questions)
                        assert response.headers['x-kev-laya-prefix-passes']=='1'
                        assert response.headers['x-kev-laya-branch-passes']==str(stats['branch_passes'])
                        assert response.headers['x-kev-laya-batch-sizes']==','.join(map(str,stats['effective_batch_sizes']))
                        results.append({'question_count':count,'model':body['model'],'input_tokens':body['usage']['input_tokens'],
                                        'batch_sizes':stats['effective_batch_sizes'],'prefix_passes':stats['prefix_passes'],
                                        'branch_passes':stats['branch_passes'],'compute_tokens':stats['compute_tokens'],
                                        'padding_tokens':stats['padding_tokens'],'exact_direct_match':True})
        finally:
            proc.terminate()
            try:proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill();proc.wait()
    snapshot=json.loads((out/'telemetry.json').read_text());validate_snapshot(snapshot)
    assert snapshot['serving']['requests']==12 and snapshot['serving']['errors']==0
    text=(out/'telemetry.json').read_text()
    assert 'color=red' not in text and 'select value' not in text
    result={'checkpoint_sha256':point['checkpoint_sha256'],'model_id':point['model_id'],
            'package_loaded_from':str(Path(kev_laya.__file__).resolve()),'requests':results,
            'retention_disabled':True,'telemetry_schema_version':snapshot['schema_version'],'telemetry_contract_valid':True,'loopback_http':True,
            'execution_totals':snapshot.get('extensions',{}).get('execution'),'total_requests':len(results),'multi_question_requests':sum(r['question_count']>1 for r in results)}
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
