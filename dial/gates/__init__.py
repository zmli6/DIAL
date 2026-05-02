"""
Adaptive gates: DIAL (the proposed method) and its no-LLM-features variant
used in ablations.
"""
from dial.gates._scg_base import SCGBase, CalibrationPoint
from dial.gates.dial_universal import PrincipledSCGGate as DIALUniversal
from dial.gates.dial import SelfEvolvingGateV2 as DIAL

__all__ = [
    "SCGBase",
    "CalibrationPoint",
    "DIAL",
    "DIALUniversal",
]
