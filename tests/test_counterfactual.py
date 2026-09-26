from collections import defaultdict, Counter
from kev_laya.counterfactual import freeze_counterfactual, regression_data, PROTOCOL
from kev_laya.data import load_suite
from kev_laya.io import sha256_file


def test_fixed_balanced_counterfactual_groups_and_protected_regression(tmp_path):
    freeze_counterfactual(tmp_path/'suite')
    data,manifest=load_suite(tmp_path/'suite')
    groups=defaultdict(list)
    for split,rows in data.items():
        for row in rows:
            groups[row.meta['group']].append(row)
            assert 'case=99999' not in row.request.state
    assert len(groups)==128 and sum(len(x) for x in groups.values())==768
    for group,rows in groups.items():
        assert len(rows)==6
        assert len({r.meta['language'] for r in rows})==1
        assert Counter(r.targets[0].index(1.) for r in rows)=={0:3,1:3}
        assert Counter(r.targets[1].index(1.) for r in rows)=={0:3,1:3}
        assert Counter(r.targets[2].index(1.) for r in rows)=={0:2,1:2,2:2}
        assert len({tuple(r.request.questions['color'].criteria) for r in rows})==1
        for level in (0,1,2):
            pair=[r for r in rows if r.targets[2].index(1.)==level]
            assert len(pair)==2
            assert pair[0].request.state.replace('color=red','color=blue',1)==pair[1].request.state
    freeze_counterfactual(tmp_path/'again')
    for split in data:
        assert sha256_file(tmp_path/'suite'/f'{split}.jsonl')==sha256_file(tmp_path/'again'/f'{split}.jsonl')
    assert PROTOCOL['accuracy_threshold']==.7 and PROTOCOL['prior_failure_preserved']['accuracy']==.5
    known=[r for r in regression_data() if r.request.state=='color=red; level=1; case=99999']
    assert len(known)==2
