"""Execute the dependency-free gate from this checkout, not a stale installed wheel."""
from pathlib import Path
import runpy
import sys

if __name__ == "__main__":
    sys.dont_write_bytecode = True
    runpy.run_path(str(Path(__file__).resolve().parents[1] /
                      "src/kev_laya/backbone_provenance.py"), run_name="__main__")
