"""Compatibility entry point for the regression-tested stage-three updater."""
import importlib.util
from pathlib import Path
_spec=importlib.util.spec_from_file_location("kev_laya_stage3_installer",Path(__file__).with_name("apply_stage3.py"))
_module=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
digest=_module.digest
plan_update=_module.plan_update
apply_update=_module.apply_update
if __name__=="__main__":_module.main()
