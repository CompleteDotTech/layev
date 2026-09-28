import json
import sys
import time

import pytest

from kev_laya.quality_supervisor import supervise_child, supervisor_lease, _write_receipt


def test_supervisor_kills_owned_child_at_deadline(tmp_path):
    marker = tmp_path / "late.txt"
    command = [sys.executable, "-c", "import pathlib,time,sys; time.sleep(2); pathlib.Path(sys.argv[1]).write_text('late')", str(marker)]
    status, code, elapsed = supervise_child(command, 0.1)
    assert (status, code) == ("deadline_killed", 124)
    assert elapsed < 1.5
    time.sleep(0.1)
    assert not marker.exists()


def test_supervisor_kills_owned_descendant_at_deadline(tmp_path):
    marker = tmp_path / "descendant-late.txt"
    started = tmp_path / "descendant-started.txt"
    grandchild = ("import pathlib,sys,time; pathlib.Path(sys.argv[2]).touch(); "
                  "time.sleep(2); pathlib.Path(sys.argv[1]).write_text('late')")
    child = ("import subprocess,sys,time; "
             "subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2],sys.argv[3]]); "
             "time.sleep(3)")
    status, code, _ = supervise_child([sys.executable, "-c", child, grandchild,
                                      str(marker), str(started)], 1)
    assert (status, code) == ("deadline_killed", 124)
    assert started.exists()
    time.sleep(2)
    assert not marker.exists()


def test_supervisor_refuses_competing_owner_and_exhausted_deadline(tmp_path):
    lease = tmp_path / "ledger.supervisor.lock"
    with supervisor_lease(lease):
        with pytest.raises(OSError):
            with supervisor_lease(lease):
                pass
    marker = tmp_path / "unexpected.txt"
    command = [sys.executable, "-c", "import pathlib,sys; pathlib.Path(sys.argv[1]).touch()", str(marker)]
    assert supervise_child(command, 0) == ("deadline_exhausted", 124, 0.0)
    assert not marker.exists()


def test_supervisor_success_and_exclusive_receipt(tmp_path):
    status, code, elapsed = supervise_child([sys.executable, "-c", "pass"], 5)
    assert (status, code) == ("completed", 0)
    assert elapsed >= 0
    receipt = tmp_path / "receipt.json"
    _write_receipt(receipt, {"status": status})
    assert json.loads(receipt.read_text()) == {"status": "completed"}
    with pytest.raises(FileExistsError):
        _write_receipt(receipt, {"status": "replacement"})
