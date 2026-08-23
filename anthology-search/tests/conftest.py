"""A tiny synthetic corpus in the real on-disk shapes, so tests never touch the 117 MB one."""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mla.index import build, connect  # noqa: E402


def _paper(key, title, year, venue, abstract, **extra):
    base = {
        "bibtex_key": key, "title": title, "year": str(year), "venue": venue,
        "venue_type": "conference", "volume": "", "pages": "", "abstract": abstract,
        "pdf_url": "", "venue_url": "", "doi": "", "openreview_url": "", "code_url": "",
        "source": "test", "source_id": key,
        "authors": [{"given": "Ada", "family": "Lovelace", "slug": "lovelace-ada"}],
    }
    return {**base, **extra}


PAPERS = [
    _paper("vaswani2017neurips-attention", "Attention is All you Need", 2017, "neurips",
           "The dominant sequence transduction models are based on recurrent networks.",
           doi="10.5555/3295222.3295349", pdf_url="https://example.invalid/a.pdf",
           authors=[{"given": "Ashish", "family": "Vaswani", "slug": "vaswani-ashish"}]),
    _paper("tyukin2024icmlw-attention",
           "Attention Is All You Need But You Don't Need All Of It At Inference",
           2024, "icmlw", "We prune attention heads at inference time.",
           venue_type="workshop"),
    _paper("he2016cvpr-deep", "Deep Residual Learning for Image Recognition", 2016, "cvpr",
           "Deeper neural networks are more difficult to train. We present a residual "
           "learning framework.", doi="10.1109/CVPR.2016.90",
           code_url="https://example.invalid/resnet",
           authors=[{"given": "Kaiming", "family": "He", "slug": "he-kaiming"}]),
    _paper("scholkopf2001neurips-kernel", "Kernel Methods and Support Vectors", 2001,
           "neurips", "A study of kernels and reproducing Hilbert spaces.",
           authors=[{"given": "Bernhard", "family": "Schölkopf",
                     "slug": "scholkopf-bernhard"}]),
]

LEGACY = [
    _paper("angluin1988colt-learning", "Learning With Hints", 1988, "colt", "",
           venue_name="Annual Conference on Computational Learning Theory"),
    _paper("valiant1984colt-theory", "A Theory of the Learnable", 1984, "colt",
           "We study learnability of concept classes from examples."),
]


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("corpus")
    (root / "papers").mkdir()
    (root / "legacy").mkdir()
    with gzip.open(root / "papers" / "mixed-2024.json.gz", "wt", encoding="utf-8") as fh:
        json.dump({"venue": "mixed", "year": "2024", "papers": PAPERS}, fh)
    with gzip.open(root / "legacy" / "colt-legacy.jsonl.gz", "wt", encoding="utf-8") as fh:
        for record in LEGACY:
            fh.write(json.dumps(record) + "\n")
    return root


@pytest.fixture(scope="session")
def index_path(tmp_path_factory, data_dir) -> Path:
    target = tmp_path_factory.mktemp("index") / "mla.sqlite3"
    build(data_dir=data_dir, index_path=target)
    return target


@pytest.fixture
def db(index_path):
    connection = connect(index_path)
    yield connection
    connection.close()
