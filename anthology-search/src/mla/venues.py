"""Canonical venue names and topical families.

The corpus stores whatever name a given proceedings volume used, so ICML 2025 is
"Proceedings of the 42nd International Conference on Machine Learning" while ICML 1988
is blank. That is right for a per-paper citation and wrong for a venue table or a
BibTeX ``booktitle``, so the stable name lives here.

FAMILIES exist because picking venues is the searcher's real decision. The skill's
METHOD-HOME axis says the paper you are missing is often phrased the way a different
community writes — ``--family theory`` sweeps COLT/ALT/UAI/ISIPTA/PGM in one call
rather than making the caller remember that ISIPTA is where imprecise-probability
people publish.
"""

from __future__ import annotations

NAMES: dict[str, str] = {
    "aaai": "AAAI Conference on Artificial Intelligence",
    "acml": "Asian Conference on Machine Learning",
    "aistats": "International Conference on Artificial Intelligence and Statistics",
    "alt": "International Conference on Algorithmic Learning Theory",
    "automl": "International Conference on Automated Machine Learning",
    "chil": "Conference on Health, Inference, and Learning",
    "clear": "Conference on Causal Learning and Reasoning",
    "collas": "Conference on Lifelong Learning Agents",
    "colt": "Annual Conference on Learning Theory",
    "corl": "Conference on Robot Learning",
    "cpal": "Conference on Parsimony and Learning",
    "cvpr": "IEEE/CVF Conference on Computer Vision and Pattern Recognition",
    "cvprw": "IEEE/CVF CVPR Workshops",
    "distill": "Distill",
    "dmlr": "Journal of Data-centric Machine Learning Research",
    "eccv": "European Conference on Computer Vision",
    "eccvw": "European Conference on Computer Vision Workshops",
    "ecml": "European Conference on Machine Learning",
    "ecmlpkdd": "European Conference on Machine Learning and Knowledge Discovery in Databases",
    "ftml": "Foundations and Trends in Machine Learning",
    "iccv": "IEEE/CVF International Conference on Computer Vision",
    "iccvw": "IEEE/CVF International Conference on Computer Vision Workshops",
    "iclr": "International Conference on Learning Representations",
    "iclrw": "International Conference on Learning Representations Workshops",
    "icml": "International Conference on Machine Learning",
    "icmlw": "International Conference on Machine Learning Workshops",
    "ijcai": "International Joint Conference on Artificial Intelligence",
    "isipta": "International Symposium on Imprecise Probability: Theories and Applications",
    "jair": "Journal of Artificial Intelligence Research",
    "jmlr": "Journal of Machine Learning Research",
    "l4dc": "Conference on Learning for Dynamics and Control",
    "log": "Learning on Graphs Conference",
    "midl": "Medical Imaging with Deep Learning",
    "mlhc": "Machine Learning for Healthcare Conference",
    "mlj": "Machine Learning (Springer)",
    "mloss": "Journal of Machine Learning Research — Open Source Software track",
    "neco": "Neural Computation",
    "neurips": "Conference on Neural Information Processing Systems",
    "neuripsw": "Conference on Neural Information Processing Systems Workshops",
    "pgm": "International Conference on Probabilistic Graphical Models",
    "tmlr": "Transactions on Machine Learning Research",
    "uai": "Conference on Uncertainty in Artificial Intelligence",
    "wacv": "IEEE/CVF Winter Conference on Applications of Computer Vision",
    "wacvw": "IEEE/CVF Winter Conference on Applications of Computer Vision Workshops",
    # --- ACL Anthology ---
    "acl": "Annual Meeting of the Association for Computational Linguistics",
    "emnlp": "Conference on Empirical Methods in Natural Language Processing",
    "naacl": "North American Chapter of the Association for Computational Linguistics",
    "eacl": "European Chapter of the Association for Computational Linguistics",
    "aacl": "Asia-Pacific Chapter of the Association for Computational Linguistics",
    "coling": "International Conference on Computational Linguistics",
    "conll": "Conference on Computational Natural Language Learning",
    "lrec": "International Conference on Language Resources and Evaluation",
    "ijcnlp": "International Joint Conference on Natural Language Processing",
    "anlp": "Conference on Applied Natural Language Processing",
    "hlt": "Human Language Technology Conference",
    "tacl": "Transactions of the Association for Computational Linguistics",
    "cl": "Computational Linguistics",
    "findings-acl": "Findings of the Association for Computational Linguistics: ACL",
    "findings-emnlp": "Findings of the Association for Computational Linguistics: EMNLP",
    "findings-naacl": "Findings of the Association for Computational Linguistics: NAACL",
    "findings-eacl": "Findings of the Association for Computational Linguistics: EACL",
    "findings-ijcnlp": "Findings of the Association for Computational Linguistics: IJCNLP",
    "findings-aacl": "Findings of the Association for Computational Linguistics: AACL",
}

FAMILIES: dict[str, tuple[str, ...]] = {
    "core-ml": (
        "neurips", "icml", "iclr", "aistats", "tmlr", "jmlr", "mlj", "neco", "dmlr",
        "acml", "ecml", "ecmlpkdd", "cpal", "collas", "automl", "distill", "mloss",
        "neuripsw", "icmlw", "iclrw",
    ),
    "theory": ("colt", "alt", "uai", "isipta", "pgm", "ftml"),
    "vision": ("cvpr", "iccv", "eccv", "wacv", "cvprw", "iccvw", "eccvw", "wacvw"),
    "ai": ("aaai", "ijcai", "jair"),
    "robotics": ("corl", "l4dc"),
    "health": ("chil", "mlhc", "midl"),
    "graphs": ("log",),
    "causal": ("clear",),
    "nlp": (
        "acl", "emnlp", "naacl", "eacl", "aacl", "coling", "conll", "lrec", "ijcnlp", "anlp", "hlt", "tacl", "cl", "findings-acl", "findings-emnlp", "findings-naacl", "findings-eacl", "findings-ijcnlp", "findings-aacl",
    ),
}

# venue -> family, for annotating a venue listing.
FAMILY_OF: dict[str, str] = {
    venue: family for family, members in FAMILIES.items() for venue in members
}


def canonical_name(venue: str, fallback: str = "") -> str:
    """The stable name for a venue slug, falling back to whatever the corpus stored."""
    return NAMES.get(venue.lower()) or fallback


def expand_families(names: tuple[str, ...]) -> tuple[list[str], list[str]]:
    """``("theory", "vision")`` -> (venue slugs, unknown family names)."""
    venues: list[str] = []
    unknown: list[str] = []
    for name in names:
        members = FAMILIES.get(name.strip().lower())
        if members is None:
            unknown.append(name)
            continue
        venues.extend(v for v in members if v not in venues)
    return venues, unknown
