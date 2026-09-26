"""Stage-three guarded local installation; never edits the reviewed package or Overwatch.

--check performs no writes. Existing destinations must match every supplied
baseline file (LF/CRLF tolerated); unrelated extra files are preserved. Conflicts
are rejected before any write. Changes receive rollback backups. No Git, remote
publishing, environment installation, cloud jobs, or WSL control commands.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import uuid


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw.replace(b'\r\n',b'\n')).hexdigest()


def plan_update(target: Path, bundle: Path | None = None):
    bundle=(bundle or Path(__file__).resolve().parent).resolve()
    target=Path(target).absolute()
    if target.is_symlink():raise ValueError('symlink destination refused')
    target=target.resolve()
    if target==bundle or target.is_relative_to(bundle):raise ValueError('do not install into the delivery/reviewed bundle')
    lower=[p.casefold() for p in target.parts]
    if 'reviews' in lower or 'review' in lower:
        raise ValueError('reviewed package must remain intact; choose the development checkout')
    if target.exists() and not target.is_dir():raise ValueError('destination is not a directory')
    if (target/'.git').is_file():raise ValueError('worktree destination refused')
    manifest=json.loads((bundle/'UPDATE_MANIFEST.json').read_text())
    writes={}
    for name,entry in manifest['files'].items():
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('invalid manifest path')
        source=bundle/'kev-laya'/relative
        raw=source.read_bytes()
        if digest(raw)!=entry['after_sha256']:raise ValueError(f'delivery checksum mismatch: {name}')
        dest=target/relative
        if dest.is_symlink() or not dest.resolve().is_relative_to(target):raise ValueError(f'unsafe destination: {name}')
        if dest.exists():
            original=dest.read_bytes();current=digest(original)
            if current==entry['after_sha256']:continue
            if entry['before_sha256'] is None or current!=entry['before_sha256']:
                raise ValueError(f'modified or unrelated destination file: {name}')
            if b'\r\n' in original:raw=raw.replace(b'\r\n',b'\n').replace(b'\n',b'\r\n')
        elif target.exists() and entry['before_sha256'] is not None:
            raise ValueError(f'missing baseline file in existing checkout: {name}')
        writes[relative]=raw
    instructions=[]
    for directory in [target,*target.parents]:
        if (directory/'AGENTS.md').is_file():instructions.append(str(directory/'AGENTS.md'))
    return target,writes,instructions


def apply_update(target: Path, bundle: Path | None = None):
    target,writes,instructions=plan_update(target,bundle)
    if not writes:return {'changed':0,'target':str(target),'backup':None,'instructions':instructions}
    if not target.exists():
        target.parent.mkdir(parents=True,exist_ok=True)
        staging=Path(tempfile.mkdtemp(prefix='.kev-laya-stage3-',dir=target.parent))
        try:
            for rel,raw in writes.items():
                path=staging/rel;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
            if target.exists():raise FileExistsError('destination appeared during installation')
            staging.rename(target)
        except BaseException:
            shutil.rmtree(staging,ignore_errors=True);raise
        return {'changed':len(writes),'target':str(target),'backup':None,'instructions':instructions}
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
    backup=target/'.stage3-backups'/stamp
    # Revalidate the entire transaction before creating backups or touching files.
    _,writes,_=plan_update(target,bundle)
    backup.mkdir(parents=True)
    journal=[]
    for rel in writes:
        path=target/rel
        if path.exists():
            saved=backup/rel;saved.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,saved)
        journal.append({'path':str(rel),'existed':path.exists()})
    (backup/'rollback.json').write_text(json.dumps(journal,indent=2)+'\n')
    changed=[]
    try:
        for rel,raw in writes.items():
            path=target/rel;path.parent.mkdir(parents=True,exist_ok=True)
            with tempfile.NamedTemporaryFile('wb',dir=path.parent,delete=False) as stream:
                temporary=Path(stream.name);stream.write(raw)
            try:temporary.replace(path)
            finally:temporary.unlink(missing_ok=True)
            changed.append(rel)
    except BaseException:
        existed={item['path']:item['existed'] for item in journal}
        for rel in reversed(changed):
            if existed[str(rel)]:shutil.copy2(backup/rel,target/rel)
            else:(target/rel).unlink(missing_ok=True)
        raise
    return {'changed':len(writes),'target':str(target),'backup':str(backup),'instructions':instructions}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--target',type=Path,required=True)
    action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--check',action='store_true');action.add_argument('--apply',action='store_true')
    args=parser.parse_args()
    if args.check:
        target,writes,instructions=plan_update(args.target)
        result={'target':str(target),'would_change':len(writes),'paths':[str(p) for p in writes],
                'applicable_instructions':instructions,'writes_performed':0}
    else:result=apply_update(args.target)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
