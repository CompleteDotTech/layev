import importlib.util
from pathlib import Path
import json
import httpx
import pytest
from fastapi.testclient import TestClient
from kev_laya.client import KevLayaClient
from kev_laya.encoding import ByteTokenizer, Limits
from kev_laya.service import InferenceRuntime, ServeSettings, create_app


def test_python_client_against_asgi_native_shapes(tiny,request_data):
    tiny.training_steps=1
    server=TestClient(create_app(InferenceRuntime(tiny,ByteTokenizer(),'kev-laya-test',Limits(512,8192)),ServeSettings(api_keys=('local-test',))))
    def relay(request):
        reply=server.request(request.method,request.url.path,content=request.content,headers=dict(request.headers))
        return httpx.Response(reply.status_code,headers=dict(reply.headers),content=reply.content)
    with KevLayaClient(api_key='local-test',transport=httpx.MockTransport(relay)) as client:
        result=client.system_one(state=request_data['state'],questions=request_data['questions'])
        assert result.model=='kev-laya-test'
        assert result.answers['truth'].type=='noul'
        assert client.models()['resolved_model']=='kev-laya-test'
    server.close()


def installer():
    path=Path(__file__).resolve().parents[1]/'integrations/overwatch/apply.py'
    spec=importlib.util.spec_from_file_location('integration_installer',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_installer_anchors_preserve_legacy_filter():
    m=installer()
    # Explicit anchor fixture, NOT a copy or successful application to the original checkout.
    snippet='''from typing import Any
class CollectorOptions:
    gcp_billing_table: str | None = None
async def collect():
    # A scoped job refresh reuses inventory and only advances that job's raw log cursor.
    if True:
        # W&B stays optional and is captured as raw hydrated run records.
        runs = [run for run in runs if run.config.get("_id_") == "TrainConfig"]

def parse_args() -> argparse.Namespace:
    parser.add_argument("--full-logs", action="store_true")
    return None
options = CollectorOptions(
        gcp_billing_table=args.gcp_billing_table,
)
'''
    result=m.transform('src/overwatch/collector.py',snippet)
    assert result.count('run.config.get("_id_") == "TrainConfig"')==1
    assert 'collect_source("kev_laya", collect_kev_laya_cache)' in result
    compile(result,'collector-anchor-fixture','exec')
    with pytest.raises(ValueError):m.transform('src/overwatch/collector.py','wrong revision')


def test_installer_refuses_dirty_targets_before_any_write(tmp_path,monkeypatch):
    m=installer()
    target='src/overwatch/constants.py';path=tmp_path/target;path.parent.mkdir(parents=True)
    original=b'# baseline\n';path.write_bytes(original)
    monkeypatch.setattr(m,'BLOBS',{target:m.blob_sha(original)})
    changes=m.prepare(tmp_path,verify_revision=False)
    assert path.read_bytes()==original
    assert b'KEV_LAYA_STALE_SECONDS' in changes[path]
    path.write_bytes(b'# unrelated work must not be lost\n')
    with pytest.raises(ValueError,match='modified or unexpected'):m.prepare(tmp_path,verify_revision=False)
    assert path.read_bytes()==b'# unrelated work must not be lost\n'


def test_installer_respects_crlf_and_existing_new_files(tmp_path,monkeypatch):
    m=installer();target='src/overwatch/constants.py';path=tmp_path/target
    path.parent.mkdir(parents=True);raw=b'# original\r\n';path.write_bytes(raw)
    monkeypatch.setattr(m,'BLOBS',{target:m.blob_sha(raw.replace(b'\r\n',b'\n'))})
    assert b'\r\n' in m.prepare(tmp_path,verify_revision=False)[path]
    collision=tmp_path/'src/overwatch/kev_laya_adapter.py';collision.write_text('unrelated code')
    with pytest.raises(ValueError,match='overwrite existing'):m.prepare(tmp_path,verify_revision=False)
    assert collision.read_text()=='unrelated code'


def test_native_typesafe_sdk_gate():
    # SDK is a separate acceptance gate, not covered by the project's own typed client.
    if importlib.util.find_spec('typesafe_sdk') is None:
        pytest.skip('official TypeSafe SDK is not installed in the offline environment')
    import os
    base=os.environ.get('KEV_LAYA_BASE_URL')
    if not base: pytest.skip('set KEV_LAYA_BASE_URL for an explicitly authorized local SDK pilot')
    from typesafe_sdk import TypeSafeClient,Choice,Noul,Score
    client=TypeSafeClient(api_key=os.environ.get('KEV_LAYA_TEST_API_KEY','local'),base_url=base,
                          model=os.environ.get('KEV_LAYA_MODEL','kev-laya-preview'))
    result=client.system_one(state='color=red; level=1;',questions={
        'color':Choice(instructions='color?',criteria={'red':None,'blue':None}),
        'flag':Noul(instructions='color=red?'),
        'level':Score(instructions='level?',criteria=['0','1','2'])})
    assert result.choices['color'].choice in {'red','blue'}
    assert 0<=result.nouls['flag'].noul<=1
    assert 0<=result.scores['level'].score<=2
