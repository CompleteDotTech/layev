"""Verify runtime bytes/licenses in a wheel; optionally smoke the installed wheel."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--installed-smoke", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    wheels = list(args.directory.glob("kev_laya-*.whl"))
    if len(wheels) != 1:
        raise ValueError("exactly one candidate wheel required")
    matched = []
    with zipfile.ZipFile(wheels[0]) as wheel:
        for path in sorted((root / "src/kev_laya").glob("*.py")):
            member = "kev_laya/" + path.name
            if wheel.read(member) != path.read_bytes():
                raise ValueError(f"wheel/source mismatch: {member}")
            matched.append(member)
        for name in ("LICENSE", "NOTICE", "licenses/Kev-LICENSE", "licenses/Laya-LICENSE"):
            candidates = [item for item in wheel.namelist() if item.endswith(".dist-info/licenses/" + name)]
            if len(candidates) != 1 or wheel.read(candidates[0]) != (root / name).read_bytes():
                raise ValueError(f"missing or changed distributed attribution: {name}")
    if args.installed_smoke:
        with tempfile.TemporaryDirectory(prefix="layev-wheel-smoke-") as temporary:
            subprocess.run([sys.executable, "-I", "-c",
                            "import kev_laya; from pathlib import Path; "
                            "p=Path(kev_laya.__file__).resolve(); "
                            "assert 'site-packages' in p.parts, p; print(p)"],
                           cwd=temporary, check=True)
            subprocess.run([sys.executable, "-I", "-m", "kev_laya", "--help"],
                           cwd=temporary, check=True, stdout=subprocess.DEVNULL)
    print(json.dumps({"wheel": wheels[0].name,
                      "sha256": hashlib.sha256(wheels[0].read_bytes()).hexdigest(),
                      "matched_runtime_modules": len(matched), "license_files_matched": 4,
                      "installed_smoke": args.installed_smoke, "passed": True}, indent=2))


if __name__ == "__main__":
    main()
