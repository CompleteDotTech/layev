"""Opt-in loopback contract test plus separate, honestly retained quality observations.
Run with an installed wheel outside the source tree; no hosted API or paid job.
"""
from pathlib import Path
import argparse
import json
import os
import socket
import subprocess
import sys
import time
import httpx
import torch
import kev_laya
from kev_laya.checkpoint import load_checkpoint
from kev_laya.cli import limits_for
from kev_laya.data import load_suite
from kev_laya.schema import SystemOneRequest
from kev_laya.service import InferenceRuntime

p=argparse.ArgumentParser()
p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--suite',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
if a.out.exists(): raise FileExistsError('refusing to replace prior evidence')
a.out.mkdir(parents=True)
a.checkpoint=a.checkpoint.resolve(); a.suite=a.suite.resolve(); a.out=a.out.resolve()
torch.set_num_threads(2)
model,tok,point=load_checkpoint(a.checkpoint)
runtime=InferenceRuntime(model,tok,point['model_id'],limits_for(point))
suite,_=load_suite(a.suite)
with socket.socket() as s:
    s.bind(('127.0.0.1',0)); port=s.getsockname()[1]
command=[sys.executable,'-m','kev_laya','serve','--checkpoint',str(a.checkpoint),'--no-auth',
         '--port',str(port),'--telemetry',str(a.out/'telemetry.json'),'--run-id','installed-http-check']
result={'kind':'installed-wheel loopback HTTP contract and diagnostic',
        'installed_module':kev_laya.__file__,'working_directory':os.getcwd(),'command':command,
        'checkpoint_sha256':point['checkpoint_sha256'],'actual_model_id':point['model_id'],
        'not_native_or_independent_quality_evidence':True,'requests':[],'software_checks':{},'quality':{}}
with (a.out/'server.log').open('w') as log:
    child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
    try:
        with httpx.Client(base_url=f'http://127.0.0.1:{port}',trust_env=False,timeout=15) as client:
            for _ in range(100):
                if child.poll() is not None: raise RuntimeError('server exited before readiness')
                try:
                    models=client.get('/v1/models')
                    if models.status_code==200: break
                except httpx.HTTPError: pass
                time.sleep(.1)
            else: raise RuntimeError('readiness timeout')
            result['software_checks']['models_http_200']=models.status_code==200
            result['software_checks']['resolved_checkpoint_identity']=models.json().get('resolved_model')==point['model_id']
            correct=total=0; conformant=True
            for item in suite['test']:
                req=item.request.model_copy(update={'model':'kev-laya-preview'})
                expected,_=runtime.predict_encoded(runtime.encode(req))
                response=client.post('/v1/systemone',json=req.model_dump())
                data=response.json(); same=response.status_code==200 and data==expected.model_dump()
                conformant &= same
                for (key,q),target in zip(req.questions.items(),item.targets,strict=True):
                    answer=data['answers'][key]
                    if q.type=='noul': index=int(answer['noul']>=.5)
                    else: index=max(range(len(target)),key=lambda i:answer['probabilities'][q.options()[i][0]])
                    correct += index==max(range(len(target)),key=target.__getitem__);total+=1
                result['requests'].append({'record_id':item.meta['id'],'http_status':response.status_code,'matches_direct_execution':same})
            result['software_checks']['all_frozen_requests_match_direct_execution']=conformant
            result['quality']['recorded_fixture']={'correct':correct,'questions':total,'accuracy':correct/total,
                                                   'previously_inspected_fixture':True}
            probe={'model':'kev-laya-preview','state':'color=red; level=1; case=99999',
                   'questions':{'color':{'type':'choice','instructions':'color?','criteria':{'red':None,'blue':None}},
                                'level':{'type':'score','instructions':'level?','criteria':['0','1','2']}}}
            response=client.post('/v1/systemone',json=probe)
            expected,_=runtime.predict_encoded(runtime.encode(SystemOneRequest.model_validate(probe)))
            result['software_checks']['diagnostic_matches_direct_execution']=response.json()==expected.model_dump()
            result['quality']['out_of_fixture_probe']={'request':probe,'response':response.json(),
                'expected_color':'red','observed_color':response.json()['answers']['color']['choice'],
                'passed':response.json()['answers']['color']['choice']=='red',
                'interpretation':'A retained semantic failure is not a transport bug. It blocks general-quality claims.'}
            result['software_contract_passed']=all(result['software_checks'].values())
    except Exception as exc:
        result.update(software_contract_passed=False,error=f'{type(exc).__name__}: {exc}')
    finally:
        child.terminate()
        try: result['server_exit']=child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill();result['server_exit']=child.wait(timeout=5)
result['telemetry_created']=(a.out/'telemetry.json').is_file()
(a.out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('requests','command')},indent=2))
# This command is a software contract check; quality has its own explicit pass/fail above.
raise SystemExit(0 if result['software_contract_passed'] else 1)
