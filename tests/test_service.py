import json
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from kev_laya.schema import SystemOneRequest, canonical
from kev_laya.service import InferenceRuntime, ServeSettings, Quotas, create_app
from kev_laya.encoding import ByteTokenizer, Limits
from kev_laya.client import KevLayaClient

def app_for(tiny,**kwargs):
    tiny.training_steps=1
    runtime=InferenceRuntime(tiny,ByteTokenizer(),'kev-laya-test-v1',Limits(512,8192))
    return create_app(runtime,ServeSettings(api_keys=('test-not-secret',),**kwargs))

AUTH={'Authorization':'Bearer test-not-secret'}

def test_mixed_native_shapes_accounting_identity(request_data,tiny):
    app=app_for(tiny)
    with TestClient(app) as client:
        r=client.post('/v1/systemone',json=request_data,headers=AUTH)
        assert r.status_code==200,r.text
        body=r.json(); answers=body['answers']
        assert body['model']=='kev-laya-test-v1'
        assert list(answers)==list(request_data['questions'])
        assert answers['truth']['type']=='noul' and 0<=answers['truth']['noul']<=1
        for key in ('color','level'):
            p=answers[key]['probabilities'];assert sum(p.values())==pytest.approx(1,abs=1e-6)
            assert 0<=answers[key]['confidence']<=1
        assert answers['level']['score']==pytest.approx(sum(int(k)*v for k,v in answers['level']['probabilities'].items()))
        assert answers['level']['legend']['1']=='{"description":"medium"}'
        assert body['usage']['input_tokens']==body['usage']['state_tokens']+sum(body['usage']['branch_tokens'])
        assert body['usage']['forward_tokens']==body['usage']['input_tokens']
        assert body['usage']['generated_tokens']==0
        assert body['usage']['output_tokens']==len(canonical(answers).encode())
        assert body['confidence_definition']=='entropy-concentration-v1'
        names=[m['name'] for m in client.get('/v1/models',headers=AUTH).json()['models']]
        assert 'kev-laya-preview' in names and 'jev-latest' not in names

@pytest.mark.parametrize('headers',[{}, {'Authorization':'Bearer wrong'}, {'Authorization':'Basic anything'}])
def test_authentication(tiny,request_data,headers):
    with TestClient(app_for(tiny)) as c:
        assert c.get('/v1/models',headers=headers).status_code==401
        assert c.post('/v1/systemone',json=request_data,headers=headers).status_code==401

def test_overload_and_quota(tiny,request_data):
    app=app_for(tiny,requests_per_minute=1)
    with TestClient(app) as c:
        app.state.slots.acquire()
        try: assert c.post('/v1/systemone',json=request_data,headers=AUTH).status_code==529
        finally:app.state.slots.release()
        assert c.post('/v1/systemone',json=request_data,headers=AUTH).status_code==200
        reply=c.post('/v1/systemone',json=request_data,headers=AUTH)
        assert reply.status_code==429 and int(reply.headers['Retry-After'])>=1

def test_token_bucket_refill():
    now=[0.]
    q=Quotas(ServeSettings(api_keys=('x',),requests_per_minute=2,input_tokens_per_minute=100),clock=lambda:now[0])
    q.admit(0,100)
    with pytest.raises(HTTPException):q.admit(0,1)
    now[0]=60.;q.admit(0,100)

def test_validation_does_not_echo_content(tiny,request_data):
    request_data['state']=123
    request_data['private']='SECRET-DO-NOT-ECHO'
    with TestClient(app_for(tiny)) as c:
        response=c.post('/v1/systemone',json=request_data,headers=AUTH)
        assert response.status_code==422
        assert 'SECRET-DO-NOT-ECHO' not in response.text
        bad=c.post('/v1/systemone',content=b'{"state":1,"state":2}',headers=AUTH)
        assert bad.status_code==422

def test_explicit_context_overflow(tiny,request_data):
    request_data['state']='a'*513
    with TestClient(app_for(tiny)) as c:
        r=c.post('/v1/systemone',json=request_data,headers=AUTH)
        assert r.status_code==422 and r.json()['detail']['limit']=='state_plus_branch'

def test_body_bound_and_unknown_model(tiny,request_data):
    with TestClient(app_for(tiny,max_body_bytes=32)) as c:
        assert c.post('/v1/systemone',json=request_data,headers=AUTH).status_code==413
    request_data['model']='jev-latest'
    with TestClient(app_for(tiny)) as c:
        assert c.post('/v1/systemone',json=request_data,headers=AUTH).status_code==422

def test_retention_disabled_no_dataset_writes(tmp_path,monkeypatch,tiny,request_data):
    monkeypatch.chdir(tmp_path)
    app=app_for(tiny)
    request_data['state']='PERSONAL-STATE-NOT-TO-BE-RETAINED'
    with TestClient(app) as c:
        assert c.post('/v1/systemone',json=request_data,headers=AUTH).status_code==200
    assert list(tmp_path.iterdir())==[]
    assert 'PERSONAL-STATE' not in json.dumps(app.state.hook.values)
    assert app.state.hook.values['requests']==1
    with pytest.raises(ValueError):ServeSettings(api_keys=('x',),retention_disabled=False)

def test_untrained_refusal(tiny):
    with pytest.raises(ValueError,match='untrained'):InferenceRuntime(tiny,ByteTokenizer(),'kev-laya-test',Limits(512,8192))

def test_bounded_serving_history(tiny):
    hook=app_for(tiny).state.hook
    for _ in range(200):hook.finish(elapsed_ms=1,error=False)
    assert len(hook.values['latency_ms'])==128 and hook.values['requests']==200

def test_255_choice_and_10_score_forward_contract():
    # A small random-weight attention fixture exercises cardinality, not pretrained quality or 32k support.
    import torch
    from kev_laya.model import DecisionEngine,BackboneConfig
    torch.manual_seed(15)
    m=DecisionEngine(BackboneConfig(hidden_size=16,intermediate_size=32,num_hidden_layers=1,
                                   num_attention_heads=2,num_key_value_heads=1,pointer_dim=8,max_position_embeddings=12000))
    m.training_steps=1
    app=create_app(InferenceRuntime(m,ByteTokenizer(),'kev-laya-cardinality-fixture',Limits(12000,24000)),ServeSettings(api_keys=('local',)))
    q={'c':{'type':'choice','instructions':None,'criteria':{f'o{i}':None for i in range(255)}},
       's':{'type':'score','instructions':None,'criteria':[str(i) for i in range(10)]}}
    with TestClient(app) as c:
        response=c.post('/v1/systemone',json={'state':'x','model':'kev-laya-preview','questions':q},headers={'Authorization':'Bearer local'})
    assert response.status_code==200,response.text
    answer=response.json()['answers']
    assert len(answer['c']['probabilities'])==255
    assert sum(answer['c']['probabilities'].values())==pytest.approx(1,abs=1e-6)
    assert len(answer['s']['legend'])==10
