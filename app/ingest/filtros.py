from typing import Dict, Any

print("Modelo en uso: Codex (GPT-5)")
FILTERS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "BTTS": {
    "abierto": {
        "min_edge": -0.10,  "min_ev": -0.10,
        "min_odds": 1.35, "max_odds": 2.10,
        "min_potential_yes": 55, "max_potential_no": 38,
        "min_pex_hit_yes": 0.45, "min_pex_ev_yes": 0.06,
        "min_pex_hit_no":  0.45, "min_pex_ev_no":  0.06,
        "sort_by":    ["pex_ev_norm_yes", "ev_real", "edge"],
        "sort_by_no": ["pex_ev_norm_no",  "ev_no",   "edge_no"],
    },
    "moderado": {
        "min_edge": 0.03, "min_ev": 0.00,
        "min_odds": 1.40, "max_odds": 2.05,
        "min_potential_yes": 60, "max_potential_no": 32,
        "min_pex_hit_yes": 0.50, "min_pex_ev_yes": 0.10,
        "min_pex_hit_no":  0.50, "min_pex_ev_no":  0.10,
        "sort_by":    ["pex_ev_norm_yes", "ev_real", "edge"],
        "sort_by_no": ["pex_ev_norm_no",  "ev_no",   "edge_no"],
    },
    "estricto": {
        "min_edge": 0.05, "min_ev": 0.00,
        "min_odds": 1.45, "max_odds": 2.00,
        "min_potential_yes": 65, "max_potential_no": 28,
        "min_pex_hit_yes": 0.55, "min_pex_ev_yes": 0.15,
        "min_pex_hit_no":  0.55, "min_pex_ev_no":  0.15,
        "sort_by":    ["pex_ev_norm_yes", "ev_real", "edge"],
        "sort_by_no": ["pex_ev_norm_no",  "ev_no",   "edge_no"],
    },
},
"OVER25": {
    "abierto":  {
        "min_edge": -0.10,  "min_ev": -0.10,
        "min_odds": 1.35, "max_odds": 2.10,
        "min_potential_over": 55,
        "min_pex_hit": 0.47, "min_pex_ev_norm": 0.06,
    },
    "moderado": {
        "min_edge": 0.03, "min_ev": 0.03,
        "min_odds": 1.40, "max_odds": 2.05,
        "min_potential_over": 60,
        "min_pex_hit": 0.52, "min_pex_ev_norm": 0.10,
    },
    "estricto": {
        "min_edge": 0.05, "min_ev": 0.05,
        "min_odds": 1.45, "max_odds": 2.00,
        "min_potential_over": 65,
        "min_pex_hit": 0.57, "min_pex_ev_norm": 0.15,
    },
},
"OVER15": {
    "abierto":  {
        "min_edge": -0.10,  "min_ev": -0.10,
        "min_odds": 1.15, "max_odds": 1.80,
        "min_potential_over": 60,
        "min_pex_hit": 0.60, "min_pex_ev_norm": 0.02,
    },
    "moderado": {
        "min_edge": 0.01, "min_ev": 0.00,
        "min_odds": 1.18, "max_odds": 1.85,
        "min_potential_over": 65,
        "min_pex_hit": 0.65, "min_pex_ev_norm": 0.04,
    },
    "estricto": {
        "min_edge": 0.02, "min_ev": 0.01,
        "min_odds": 1.20, "max_odds": 1.90,
        "min_potential_over": 70,
        "min_pex_hit": 0.70, "min_pex_ev_norm": 0.06,
    },
},

    # >>> Estándar Gol HT con tope de cuota 1.35 <<<
    "O05HT": {
        "abierto": {
        "min_edge": -0.10,  "min_ev": -0.10,
        "min_odds": 1.02,   "max_odds": 1.65,   # más partidos aunque cuota base sea baja
        "min_p_model_ht": 0.66,
        "min_potential_ht": 70,
        "min_total_xg": 2.0,
        "min_pex_hit": 0.55,
        "min_pex_ev_norm": 0.03,
        "racha_min_goal_rate": 0.6,
        "sort_by": ["pex_ev_norm_ht", "ev_ht", "edge_ht"],
    },
    "moderado": {
        "min_edge": -0.02,  "min_ev": -0.01,
        "min_odds": 1.02,   "max_odds": 1.50,   # equilibrio con odds bajas permitidas
        "min_p_model_ht": 0.70,
        "min_potential_ht": 70,
        "min_total_xg": 2.6,
        "min_pex_hit": 0.60,
        "min_pex_ev_norm": 0.05,
        "racha_min_goal_rate": 0.65,
        "sort_by": ["pex_ev_norm_ht", "ev_ht", "edge_ht"],
    },
    "estricto": {
        "min_edge": 0.00,   "min_ev": 0.01,
        "min_odds": 1.02,   "max_odds": 1.40,   # mantiene exigencia pero acepta cuotas iniciales bajas
        "min_p_model_ht": 0.73,
        "min_potential_ht": 70,
        "min_total_xg": 2.8,
        "min_pex_hit": 0.65,
        "min_pex_ev_norm": 0.08,
        "racha_min_goal_rate": 0.7,
        "sort_by": ["pex_ev_norm_ht", "ev_ht", "edge_ht"],
    },
    },
"CORNERS": {
        "abierto": {
            "min_corners_ht": 2,
            "min_corners_ft": 8,
            "min_potential_ft": 8.0,
            "min_o85_potential": 60.0,
            "min_odds_over85": 1.10,
            "max_odds_over85": 3.50,
        },
        "moderado": {
            "min_corners_ht": 3,
            "min_corners_ft": 8,
            "min_potential_ft": 9.0,
            "min_o85_potential": 65.0,
            "min_odds_over85": 1.20,
            "max_odds_over85": 3.10,
        },
        "estricto": {
            "min_corners_ht": 4,
            "min_corners_ft": 9,
            "min_potential_ft": 10.0,
            "min_o85_potential": 70.0,
            "min_odds_over85": 1.25,
            "max_odds_over85": 2.75,
        },
    },
    "CORNERS_U105": {
        "abierto": {
            "max_potential_ft": 11.0,
            "max_o105_potential": 55.0,
            "min_odds_under105": 1.30,
            "max_odds_under105": 1.80,
        },
        "moderado": {
            "max_potential_ft": 10.0,
            "max_o105_potential": 50.0,
            "min_odds_under105": 1.30,
            "max_odds_under105": 1.80,
        },
        "estricto": {
            "max_potential_ft": 9.5,
            "max_o105_potential": 45.0,
            "min_odds_under105": 1.30,
            "max_odds_under105": 1.80,
        },
    },
    "CORNERS_U95": {
        "abierto": {
            "max_potential_ft": 10.5,
            "max_o95_potential": 60.0,
            "min_odds_under95": 1.30,
            "max_odds_under95": 1.80,
        },
        "moderado": {
            "max_potential_ft": 10.0,
            "max_o95_potential": 55.0,
            "min_odds_under95": 1.30,
            "max_odds_under95": 1.80,
        },
        "estricto": {
            "max_potential_ft": 9.0,
            "max_o95_potential": 50.0,
            "min_odds_under95": 1.30,
            "max_odds_under95": 1.80,
        },
    },
}

def filters(market: str, n_matches: int) -> Dict[str, Any]:
    """
    Devuelve el preset de filtros adaptado al volumen diario de partidos.
    - <10 partidos: sin filtros (se analiza todo)
    - 10-25 partidos: filtros abiertos (más accesibles)
    - 26-50 partidos: filtros moderados
    - >50 partidos: filtros estrictos
    """
    presets = FILTERS.get(market)
    if presets is None:
        raise ValueError(f"Mercado no soportado: {market}")

    if n_matches < 10:
        cfg = dict(presets["abierto"])
        cfg["aplicar_filtros"] = False
        return cfg
    if n_matches <= 25:
        tier = "abierto"
    elif n_matches <= 50:
        tier = "moderado"
    else:
        tier = "estricto"

    # Se devuelve una copia para evitar mutaciones accidentales del preset base
    return dict(presets[tier])
