from pathlib import Path
import sys
import pytest
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'integrations/overwatch/added/src')]
torch.set_num_threads(2)

@pytest.fixture
def request_data():
    return {'state': {'color': 'red', 'level': 1}, 'model': 'kev-laya-preview', 'questions': {
        'color': {'type': 'choice', 'instructions': 'Which color?', 'criteria': {'red': None, 'blue': None}},
        'level': {'type': 'score', 'instructions': 'What level?', 'criteria': ['low', {'description': 'medium'}, 'high']},
        'truth': {'type': 'noul', 'instructions': 'Is color red?', 'criteria': {'false': 'blue', 'true': 'red'}}}}

@pytest.fixture
def tiny():
    from kev_laya.model import DecisionEngine, BackboneConfig
    torch.manual_seed(47)
    return DecisionEngine(BackboneConfig(hidden_size=32, intermediate_size=64, pointer_dim=16))

@pytest.fixture
def suite(tmp_path):
    from kev_laya.data import freeze_smoke, load_suite
    freeze_smoke(tmp_path / 'suite')
    return load_suite(tmp_path / 'suite')
