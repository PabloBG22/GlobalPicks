import pandas as pd
import numpy as np
from math import exp
from typing import Any, Dict, List, Optional

def fair_prob_two_way(odds_yes: Optional[float], odds_no: Optional[float]) -> Optional[float]:
    """Prob. justa de 'sí' (elimina overround) en un mercado 2 vías."""
    if odds_yes is None or odds_no is None:
        return None
    try:
        oy, on = float(odds_yes), float(odds_no)
    except (TypeError, ValueError):
        return None
    if oy <= 1.0 or on <= 1.0:
        return None
    p_yes, p_no = 1.0/oy, 1.0/on
    z = p_yes + p_no
    return (p_yes / z) if z > 0 else None