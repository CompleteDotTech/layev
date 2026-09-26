import argparse
from pathlib import Path
from kev_laya.native_validation import run
p=argparse.ArgumentParser()
p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
p.add_argument('--device',default='cuda')
p.add_argument('--precision',choices=['fp32','bf16'],default='fp32')
a=p.parse_args()
r=run(a.checkpoint,a.out,a.device,a.precision)
print('PASS' if r['passed'] else 'FAIL: inspect report.json; no replacement certification')
raise SystemExit(0 if r['passed'] else 1)
