"""Shipped templates must not masquerade as an approved preregistration."""
import json
from pathlib import Path

import pytest

from kev_laya.quality_protocol import ProtocolError, validate_protocol
from test_quality_protocol import protocol, review

ROOT = Path(__file__).resolve().parents[1] / "docs/quality-protocol-templates"


def test_protocol_template_cannot_validate_as_delivered():
    with pytest.raises(ProtocolError):
        validate_protocol(json.loads((ROOT / "protocol.template.json").read_text()), review())


def test_review_template_cannot_validate_as_delivered():
    with pytest.raises(ProtocolError):
        validate_protocol(protocol(), json.loads((ROOT / "data-review.template.json").read_text()))
