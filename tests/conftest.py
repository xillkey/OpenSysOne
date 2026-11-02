from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def toy_data(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("toy")
    subprocess.run(
        [sys.executable, str(ROOT / "examples/toy_tickets/make_data.py"), "--out", str(out),
         "--train", "400", "--dev", "120", "--test", "60"],
        check=True, capture_output=True,
    )
    return out


@pytest.fixture(scope="session")
def tiny_backbone(toy_data: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    from opensysone.tiny import make_tiny_backbone

    return make_tiny_backbone([toy_data / "train.jsonl"], tmp_path_factory.mktemp("bb"), vocab_size=2000,
                              hidden_size=64, num_layers=2, num_heads=2, max_position=256)
