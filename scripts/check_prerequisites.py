"""Record native/application gates without substituting dependency stubs or fake UI.

Exit 2 means missing prerequisites. This performs no package installation, remote
publishing, token download, cloud job, or provider call.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import torch
from kev_laya.io import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--overwatch', type=Path)
    parser.add_argument('--qwen', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists(): raise FileExistsError(args.out)
    modules = {m: importlib.util.find_spec(m) is not None for m in ('tokenizers','transformers','wandb','sky','pyzstd','typesafe','playwright')}
    result = {'python':sys.version, 'platform':platform.platform(), 'torch':torch.__version__,
              'cuda_available':torch.cuda.is_available(), 'native_packages':modules,
              'overwatch_python_supported':(3,13,12)<=sys.version_info[:3]<(3,14,0),
              'overwatch_checkout':None, 'qwen_directory':None,
              'live_overwatch_integration':'unverified', 'native_context':'unverified',
              'private_dependencies':'requires the existing authorized TypeSafe package index environment'}
    if args.overwatch:
        result['overwatch_checkout'] = {'path':str(args.overwatch), 'exists':args.overwatch.is_dir()}
        if args.overwatch.is_dir():
            p=subprocess.run(['git','-C',str(args.overwatch),'rev-parse','HEAD'],text=True,capture_output=True,timeout=5)
            result['overwatch_checkout']['revision']=p.stdout.strip() if p.returncode==0 else None
    if args.qwen:
        required=('config.json','model.safetensors','tokenizer.json','tokenizer_config.json','LICENSE','source.json')
        result['qwen_directory']={'path':str(args.qwen), 'missing':[n for n in required if not (args.qwen/n).is_file()]}
    native_ready=all(modules[m] for m in ('tokenizers','transformers')) and torch.cuda.is_available() and result['qwen_directory'] is not None and not result['qwen_directory']['missing']
    app_ready=result['overwatch_python_supported'] and all(modules[m] for m in ('wandb','sky','pyzstd','typesafe')) and result['overwatch_checkout'] is not None and result['overwatch_checkout']['exists']
    result['prerequisites_present']={'native':native_ready,'overwatch_backend':app_ready}
    result['note']='Presence is not verification: native measurements, private-package compatibility and actual React UI remain separate gates.'
    atomic_json(args.out,result)
    print(json.dumps(result,indent=2))
    return 0 if native_ready and app_ready else 2
if __name__=='__main__':raise SystemExit(main())
