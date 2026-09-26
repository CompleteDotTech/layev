import importlib.util
import json
from pathlib import Path
import pytest


def installer():
    file=Path(__file__).resolve().parents[1]/'integrations/archive/apply_parallel.py'
    # Exact retained updater bytes; tests use temporary bundle fixtures only.
    assert file.is_file(), 'standalone source must retain the tested historical updater'
    spec=importlib.util.spec_from_file_location('parallel_installer',file)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def bundle(tmp_path,m):
    b=tmp_path/'bundle';(b/'kev-laya/src').mkdir(parents=True)
    (b/'kev-laya/src/model.py').write_bytes(b'new\n')
    (b/'kev-laya/NOTICE').write_bytes(b'license\n')
    (b/'UPDATE_MANIFEST.json').write_text(json.dumps({'files':{
        'src/model.py':{'before_sha256':m.digest(b'old\n'),'after_sha256':m.digest(b'new\n')},
        'NOTICE':{'before_sha256':None,'after_sha256':m.digest(b'license\n')}}}))
    return b


def test_installer_creates_missing_checkout_and_is_idempotent(tmp_path):
    m=installer();b=bundle(tmp_path,m);target=tmp_path/'dev'
    _,writes,_=m.plan_update(target,b);assert len(writes)==2 and not target.exists()
    result=m.apply_update(target,b);assert result['changed']==2
    assert (target/'src/model.py').read_bytes()==b'new\n'
    assert m.apply_update(target,b)['changed']==0


def test_installer_preserves_unrelated_work_and_crlf_with_backups(tmp_path):
    m=installer();b=bundle(tmp_path,m);target=tmp_path/'dev';(target/'src').mkdir(parents=True)
    (target/'src/model.py').write_bytes(b'old\r\n');(target/'personal.txt').write_text('preserve')
    result=m.apply_update(target,b)
    assert (target/'src/model.py').read_bytes()==b'new\r\n'
    assert (target/'personal.txt').read_text()=='preserve'
    assert (Path(result['backup'])/'src/model.py').read_bytes()==b'old\r\n'


def test_installer_refuses_conflict_before_any_write(tmp_path):
    m=installer();b=bundle(tmp_path,m);target=tmp_path/'dev';(target/'src').mkdir(parents=True)
    (target/'src/model.py').write_text('unrelated changes')
    with pytest.raises(ValueError,match='modified or unrelated'):m.apply_update(target,b)
    assert (target/'src/model.py').read_text()=='unrelated changes'
    assert not (target/'NOTICE').exists() and not (target/'.parallel-backups').exists()


def test_installer_refuses_tampered_delivery(tmp_path):
    m=installer();b=bundle(tmp_path,m);(b/'kev-laya/NOTICE').write_text('tampered')
    with pytest.raises(ValueError,match='checksum'):m.apply_update(tmp_path/'dev',b)
    assert not (tmp_path/'dev').exists()


def test_installer_protects_reviewed_package(tmp_path):
    m=installer();b=bundle(tmp_path,m)
    with pytest.raises(ValueError,match='reviewed package'):
        m.apply_update(tmp_path/'overwatch-model-research/reviews/review/kev-laya',b)
    with pytest.raises(ValueError,match='delivery/reviewed'):m.apply_update(b/'kev-laya',b)
