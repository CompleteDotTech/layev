"""Measured source and resource metadata. No remote lookup or fabricated Git IDs."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from urllib.parse import urlsplit
from .encoding import preprocessing_identity
from .io import sha256_file
from .telemetry import utc_now

from . import __version__
VERSION = __version__


def capture_source(root: Path | None = None, *, archive: Path | None = None) -> dict:
    package = Path(__file__).resolve().parent
    root = Path(root).resolve() if root else package.parent.parent
    files = {}
    code = root / 'src' / 'kev_laya'
    if not code.is_dir():
        code = package
    for path in sorted(code.rglob('*.py')):
        if '__pycache__' not in path.parts and not path.is_symlink():
            files[path.relative_to(code).as_posix()] = sha256_file(path)
    tree = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    result = {'repository': None, 'commit': None, 'git_dirty': None, 'git_status': 'unavailable',
              'source_tree_sha256': tree, 'source_files': files, 'archive_sha256': None,
              'package_version': VERSION}
    if archive is not None:
        result['archive_sha256'] = sha256_file(Path(archive))
    git_root = next((p for p in (root, *root.parents) if (p / '.git').exists()), None)
    if git_root is not None:
        def git(*args):
            return subprocess.run(['git', '-C', str(git_root), *args], capture_output=True,
                                  check=True, text=True, timeout=3).stdout.strip()
        try:
            commit = git('rev-parse', 'HEAD')
            if len(commit) not in {40, 64} or any(c not in '0123456789abcdef' for c in commit):
                raise ValueError('invalid Git revision')
            result.update(commit=commit, git_dirty=bool(git('status', '--porcelain', '--untracked-files=normal')),
                          git_status='verified')
            try:
                origin = git('remote', 'get-url', 'origin')
                if origin.startswith('git@github.com:'):
                    origin = 'https://github.com/' + origin.split(':', 1)[1]
                url = urlsplit(origin)
                if url.scheme == 'https' and url.hostname and not url.username and not url.password and not url.query and not url.fragment:
                    result['repository'] = origin.removesuffix('.git')
            except subprocess.SubprocessError:
                pass  # A local Git repository may legitimately have no remote.
        except (OSError, ValueError, subprocess.SubprocessError):
            result.update(git_status='error', commit=None, git_dirty=None)
    return result


def hardware_identity(device) -> str:
    """Small measured platform label; detailed versions live in the run environment artifact."""
    import platform
    name = platform.machine()
    if getattr(device, 'type', None) == 'cuda':
        try:
            import torch
            name = torch.cuda.get_device_name(device)
        except Exception:
            name = 'CUDA device name unavailable'
    elif Path('/proc/cpuinfo').is_file():
        try:
            with open('/proc/cpuinfo', encoding='utf-8') as stream:
                for line in stream.read(8192).splitlines():
                    if line.startswith('model name'):
                        name = line.partition(':')[2].strip()
                        break
        except OSError:
            pass
    return f'{device}: {name}; {platform.system()} {platform.machine()}'[:256]


def compact_calibration(provenance: dict | None) -> dict:
    provenance = provenance or {}
    fits = provenance.get('fits', {})
    return {'status': provenance.get('status', 'unknown'), 'method': provenance.get('method'),
            'split_sha256': provenance.get('sha256'), 'per_type': {
                kind: {k: row.get(k) for k in ('temperature', 'count', 'status', 'before_nll', 'after_nll')}
                for kind, row in fits.items()}}


def extensions(tokenizer, source: dict, *, lineage=(), execution=None, resources=None, lineage_durable=None, calibration=None) -> dict:
    return {'serialization': preprocessing_identity(tokenizer),
            'source': {k: source.get(k) for k in ('source_tree_sha256', 'archive_sha256', 'git_dirty', 'git_status', 'package_version')},
            'attempt_lineage': list(lineage), 'lineage_durable': lineage_durable,
            'execution': execution, 'resources': resources, 'calibration': compact_calibration(calibration)}


class ResourceSampler:
    """At most one measured sample per configured interval; missing is unknown."""
    def __init__(self, *, interval=1.0, device=None, clock=time.monotonic):
        if not isinstance(interval, (int, float)) or not 0.01 <= interval <= 3600:
            raise ValueError('invalid resource sampling interval')
        self.interval = float(interval)
        self.device, self.clock, self.last_at, self.last = device, clock, float('-inf'), None

    def sample(self, *, force=False) -> dict:
        now = self.clock()
        if not force and self.last is not None and now - self.last_at < self.interval:
            return dict(self.last)
        rss = None
        try:
            import psutil
            rss = psutil.Process().memory_info().rss
        except Exception:  # Optional monitoring library errors are not training failures.
            try:
                with open('/proc/self/statm', 'r', encoding='ascii') as f:
                    rss = int(f.read(128).split()[1]) * os.sysconf('SC_PAGE_SIZE')
            except (OSError, ValueError, IndexError, AttributeError):
                pass
        allocated = peak = None
        if self.device is not None and getattr(self.device, 'type', None) == 'cuda':
            try:
                import torch
                allocated = torch.cuda.memory_allocated(self.device)
                peak = torch.cuda.max_memory_allocated(self.device)
            except Exception:  # CUDA measurement support may differ from execution support.
                pass
        self.last = {'sampled_at': utc_now(), 'interval_seconds': self.interval, 'rss_bytes': rss,
                     'gpu_allocated_bytes': allocated, 'gpu_peak_allocated_bytes': peak, 'units': 'bytes',
                     'unavailable': ([] if rss is not None else ['rss']) + ([] if allocated is not None else ['cuda'])}
        self.last_at = now
        return dict(self.last)
