from pathlib import Path
import json
import pytest
import jsonschema
from kev_laya.checkpoint import save_checkpoint
from kev_laya.encoding import ByteTokenizer
from kev_laya.cli import main

ROOT=Path(__file__).resolve().parents[1]

def test_generated_request_schema(request_data):
    schema=json.loads((ROOT/'schemas/api-request.schema.json').read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(request_data,schema)
    request_data['questions']['color']['criteria']={str(i):None for i in range(256)}
    with pytest.raises(jsonschema.ValidationError):jsonschema.validate(request_data,schema)

def test_all_schema_documents_valid():
    for path in (ROOT/'schemas').glob('*.schema.json'):
        jsonschema.Draft202012Validator.check_schema(json.loads(path.read_text()))

def test_evaluation_and_calibration_cli_write_real_artifacts(tmp_path,tiny,suite,capsys):
    checkpoint=tmp_path/'trained-fixture.pt';tiny.training_steps=1
    save_checkpoint(checkpoint,tiny,ByteTokenizer(),provenance={'context_limits':{'branch':512,'aggregate':8192}})
    report=tmp_path/'evaluation.json'
    main(['evaluate','--checkpoint',str(checkpoint),'--suite',str(tmp_path/'suite'),'--split','test','--out',str(report)])
    telemetry=json.loads(report.with_suffix('.telemetry.json').read_text())
    assert telemetry['phase']=='completed' and telemetry['artifacts'][0]['kind']=='evaluation'
    schema=json.loads((ROOT/'schemas/telemetry.schema.json').read_text())
    jsonschema.validate(telemetry,schema)
    calibrated=tmp_path/'calibrated.pt'
    main(['calibrate','--checkpoint',str(checkpoint),'--suite',str(tmp_path/'suite'),'--out',str(calibrated)])
    telemetry=json.loads(calibrated.with_suffix('.telemetry.json').read_text())
    assert telemetry['phase']=='completed'
    assert calibrated.exists() and calibrated.with_suffix('.calibration.json').exists()
    assert not json.loads(calibrated.with_suffix('.manifest.json').read_text())['resumable']
    jsonschema.validate(telemetry,schema)
