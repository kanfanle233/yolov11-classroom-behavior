# -*- coding: utf-8 -*-
from __future__ import annotations

ACTION_MAP = {
    "stand": 7,
    "sit": 0,
    "hand_raise": 6,
    "reading": 8,
    "writing": 5,
    "phone": 2,
    "sleep": 3,
    "interact": 4,
    "bow_head": 1,
    "listen": 0,
}

LABEL_NORMALIZE = {
    "dx": "writing",
    "dk": "reading",
    "tt": "listen",
    "zt": "bow_head",
    "js": "hand_raise",
    "zl": "stand",
    "xt": "interact",
    "jz": "interact",
    "doze": "sleep",
    "distract": "bow_head",
}

ALL_ACTIONS = [
    "hand_raise",
    "stand",
    "sit",
    "reading",
    "writing",
    "phone",
    "sleep",
    "bow_head",
    "listen",
    "interact",
]
