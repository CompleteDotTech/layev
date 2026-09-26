import copy
import json
import pytest
from kev_laya.schema import Question, SystemOneRequest, strict_loads, canonical
from kev_laya.encoding import ByteTokenizer, Limits, encode_request, ContextOverflow

@pytest.mark.parametrize('entry', [None, 'text', {}, [], {'nested': [False, 0, 'é', None]}, ['a', {'b': 2}]])
def test_structured_instructions(entry):
    q = Question(type='choice', instructions=entry, criteria={'x': entry})
    assert q.instructions == entry

@pytest.mark.parametrize('size', [1, 2, 10, 255])
def test_choice_boundaries(size):
    assert len(Question(type='choice', instructions=None, criteria={str(i): None for i in range(size)}).options()) == size

@pytest.mark.parametrize('size', [0, 256])
def test_choice_rejects_invalid_cardinality(size):
    with pytest.raises(ValueError):
        Question(type='choice', instructions=None, criteria={str(i): None for i in range(size)})

@pytest.mark.parametrize('size', [2, 3, 10])
def test_score_boundaries(size):
    assert len(Question(type='score', instructions=None, criteria=[None] * size).options()) == size

@pytest.mark.parametrize('size', [0, 1, 11])
def test_score_rejects_invalid_cardinality(size):
    with pytest.raises(ValueError):
        Question(type='score', instructions=None, criteria=[None] * size)

@pytest.mark.parametrize('invalid', [1, True, 1.5])
def test_invalid_instruction_shape(invalid):
    with pytest.raises(ValueError):
        Question(type='noul', instructions=invalid)

@pytest.mark.parametrize('raw', [b'{"state":1,"state":2}', b'{"x":NaN}', b'{"x":Infinity}', b'\xff'])
def test_strict_json(raw):
    with pytest.raises((ValueError, UnicodeError)):
        strict_loads(raw)

def test_noul_false_true_order_and_empty_structures():
    q = Question(type='noul', instructions='', criteria={'true': {}, 'false': []})
    assert q.options() == [('false', []), ('true', {})]

def test_noul_rejects_unknown_criteria():
    with pytest.raises(ValueError):
        Question(type='noul', instructions='', criteria={'maybe': 'unknown'})

def test_lossless_unicode_control_text(request_data):
    request_data['state'] = 'é中文\n<|fim_prefix|>\\'
    e = encode_request(SystemOneRequest(**request_data), ByteTokenizer(), Limits(512, 8192))
    assert bytes(e.state[1:]).decode() == request_data['state']
    assert all(i < 256 for i in e.state[1:])

def test_serialized_accounting_and_id_exclusion(request_data):
    t = ByteTokenizer()
    r = SystemOneRequest(**request_data)
    e = encode_request(r, t, Limits(512, 8192))
    assert e.logical_tokens == len(e.state) + sum(len(b.ids) for b in e.branches)
    assert bytes(e.state[1:]).decode() == canonical(r.state)
    other = copy.deepcopy(request_data)
    other['questions'] = {f'unrelated-{i}-🙂': v for i, v in enumerate(other['questions'].values())}
    changed = encode_request(SystemOneRequest(**other), t, Limits(512, 8192))
    assert e.state == changed.state
    assert [b.ids for b in e.branches] == [b.ids for b in changed.branches]

def test_exact_branch_and_aggregate_boundaries(request_data):
    r = SystemOneRequest(**request_data)
    e = encode_request(r, ByteTokenizer(), Limits(512, 8192))
    longest = len(e.state) + max(len(b.ids) for b in e.branches)
    assert encode_request(r, ByteTokenizer(), Limits(longest, e.logical_tokens)).logical_tokens == e.logical_tokens
    with pytest.raises(ContextOverflow) as err:
        encode_request(r, ByteTokenizer(), Limits(longest - 1, e.logical_tokens))
    assert err.value.detail['limit'] == 'state_plus_branch'
    with pytest.raises(ContextOverflow) as err:
        encode_request(r, ByteTokenizer(), Limits(longest, e.logical_tokens - 1))
    assert err.value.detail['limit'] == 'aggregate_input'

def test_no_state_truncation():
    r = SystemOneRequest(state='x'*513, model='training', questions={'q': {'type':'noul', 'instructions':'x'}})
    with pytest.raises(ContextOverflow):
        encode_request(r, ByteTokenizer(), Limits(512, 8192))

def test_question_limit(request_data):
    with pytest.raises(ContextOverflow) as err:
        encode_request(SystemOneRequest(**request_data), ByteTokenizer(), Limits(512, 8192, 2))
    assert err.value.detail['limit'] == 'questions'

@pytest.mark.parametrize('state', [None, 3, True, float('nan')])
def test_invalid_state(state, request_data):
    request_data['state'] = state
    with pytest.raises(ValueError):
        SystemOneRequest(**request_data)

def test_deep_and_nonfinite_structures_rejected():
    x = 'x'
    for _ in range(40): x = [x]
    with pytest.raises(ValueError): Question(type='noul', instructions=x)
    with pytest.raises(ValueError): Question(type='noul', instructions={'x':float('nan')})
