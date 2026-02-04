from typing import Dict, Any
import argparse
import pandas as pd
import numpy as np

from app.ingest.normalizacion.normalizacion_over15 import over15_to_df
from app.ingest.listado_ligas import enrich_liga_cols
from app.ingest.filtros import filters
from app.ingest.historico.over_contexto import enriquecer_over
from app.ingest.resultados.estado import anotar_estado


def _format_decimal(value, decimals):
    if pd.isna(value):
        return ""
    s = f"{float(value):.{decimals}f}"
    s = s.rstrip("0").rstrip(".")
    return s or "0"


def _format_common(df: pd.DataFrame) -> pd.DataFrame:
    if "ODDS" in df.columns:
        df["ODDS"] = pd.to_numeric(df["ODDS"], errors="coerce").apply(lambda x: _format_decimal(x, 2))
    if "Tarjetas" in df.columns:
        df["Tarjetas"] = pd.to_numeric(df["Tarjetas"], errors="coerce").apply(lambda x: _format_decimal(x, 1))
    if "ROI_estimado" in df.columns:
        df["ROI_estimado"] = pd.to_numeric(df["ROI_estimado"], errors="coerce").apply(lambda x: _format_decimal(x, 1))
    if "Ventaja_modelo" in df.columns:
        df["Ventaja_modelo"] = pd.to_numeric(df["Ventaja_modelo"], errors="coerce").apply(lambda x: _format_decimal(x, 1))
    return df


def _apply_over_aliases(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    alias = {
        "p_model_over15": "p_model_over",
        "p_fair_over15":  "p_fair_over",
        "edge_over15":    "edge_over",
        "ev_over15":      "ev_over",
        "cuota_real_over15":  "cuota_real_over",
        "cuota_model_over15": "cuota_model_over",
        "cuota_justa_over15": "cuota_justa_over",
        "value_pct_model_over15": "value_pct_model_over",
        "value_pct_justa_over15": "value_pct_justa_over",
        "odds_o15": "odds_over15",
        "pex_hit_over15": "pex_hit_over",
        "pex_ev_norm_over15": "pex_ev_norm_over",
    }
    for src, dst in alias.items():
        if src in d.columns and dst not in d.columns:
            d = d.rename(columns={src: dst})
    return d


def _ensure_over_cols(d: pd.DataFrame) -> pd.DataFrame:
    d = _apply_over_aliases(d).copy()

    if "cuota_real_over" not in d.columns and "odds_over15" in d.columns:
        d["cuota_real_over"] = pd.to_numeric(d["odds_over15"], errors="coerce")

    if "edge_over" not in d.columns and {"p_model_over", "p_fair_over"} <= set(d.columns):
        d["edge_over"] = d["p_model_over"] - d["p_fair_over"]

    if "ev_over" not in d.columns and {"p_model_over", "cuota_real_over"} <= set(d.columns):
        d["ev_over"] = d["p_model_over"] * d["cuota_real_over"] - 1

    if "cuota_model_over" not in d.columns and "p_model_over" in d.columns:
        d["cuota_model_over"] = d["p_model_over"].apply(lambda x: (1/x) if pd.notna(x) and x > 0 else np.nan)

    if "cuota_justa_over" not in d.columns and "p_fair_over" in d.columns:
        d["cuota_justa_over"] = d["p_fair_over"].apply(lambda x: (1/x) if pd.notna(x) and x > 0 else np.nan)

    if "ODDS" not in d.columns and "odds_over15" in d.columns:
        d["ODDS"] = pd.to_numeric(d["odds_over15"], errors="coerce")

    return d


def _apply_optional_threshold(mask: pd.Series, series: pd.Series, thr: Any, op: str = ">=") -> pd.Series:
    if thr is None:
        return mask
    if op == ">=":
        return mask & (series.fillna(-1) >= thr)
    if op == "<=":
        return mask & (series.fillna(1e9) <= thr)
    return mask


def _final_over15_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    d = _ensure_over_cols(raw_df).copy()
    d["OVER"] = "OVER 1.5"

    if "ev_over" in d.columns:
        d["ROI_estimado"] = d["ev_over"]
    if "edge_over" in d.columns:
        d["Ventaja_modelo"] = d["edge_over"]

    d = enrich_liga_cols(d, competition_id_col="competition_id", season_id_col="season_id")

    d = d.rename(columns={"kickoff_cdmx": "hora"})

    cols_out = [
        "match_id", "season_id","season_label","competition_id","home_id","away_id", "hora", "Pais", "Liga", "home", "away",
        "ODDS", "OVER",
        "pex_hit_over", "pex_ev_norm_over",
        "o15_potential", "cards_potential", "ROI_estimado", "Ventaja_modelo",
        "Potencial_final", "Prob_modelo"
    ]
    d = d[[c for c in cols_out if c in d.columns]].sort_values("hora").reset_index(drop=True)

    for c in ("ODDS", "ROI_estimado", "Ventaja_modelo", "pex_hit_over", "pex_ev_norm_over"):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce").round(3)
    if "o15_potential" in d.columns:
        d["o15_potential"] = pd.to_numeric(d["o15_potential"], errors="coerce").round(0)
    if "p_model_over" in d.columns:
        prob_series = pd.to_numeric(d["p_model_over"], errors="coerce")
        d["Potencial_final"] = (prob_series * 100).round(1)
        d["Prob_modelo"] = (prob_series * 100).round(0).astype("Int64")

    return d


def build_over15_picks_df(cfg: Dict[str, Any], today_match, aplicar_filtros: bool = True) -> pd.DataFrame:
    base = over15_to_df(today_match).copy()
    base = _ensure_over_cols(base)
    base = enriquecer_over(base)

    for c in ["edge_over", "ev_over", "ODDS", "o15_potential", "pex_hit_over", "pex_ev_norm_over"]:
        if c in base.columns:
            base[c] = pd.to_numeric(base[c], errors="coerce")

    if not aplicar_filtros:
        base["Rescate"] = False
        return _final_over15_df(base)

    edge = base["edge_over"] if "edge_over" in base.columns else pd.Series(np.nan, index=base.index)
    ev = base["ev_over"] if "ev_over" in base.columns else pd.Series(np.nan, index=base.index)
    odds = base["ODDS"] if "ODDS" in base.columns else pd.Series(np.nan, index=base.index)

    mask = (edge.fillna(-1) >= cfg["min_edge"]) & (ev.fillna(-1) >= cfg["min_ev"])
    mask &= odds.between(cfg["min_odds"], cfg["max_odds"], inclusive="both")

    if "o15_potential" in base.columns and "min_potential_over" in cfg:
        mask &= base["o15_potential"].fillna(0) >= cfg["min_potential_over"]

    if "min_pex_hit" in cfg and "pex_hit_over" in base.columns and cfg["min_pex_hit"] is not None:
        mask = _apply_optional_threshold(mask, base["pex_hit_over"], cfg["min_pex_hit"], ">=")
    if "min_pex_ev_norm" in cfg and "pex_ev_norm_over" in base.columns and cfg["min_pex_ev_norm"] is not None:
        mask = _apply_optional_threshold(mask, base["pex_ev_norm_over"], cfg["min_pex_ev_norm"], ">=")

    filtro_base = base.loc[mask].copy()
    fallback_mask = (~mask) & (
        base.get("local_flag_romper_over15", pd.Series(False, index=base.index)).fillna(False) |
        base.get("visita_flag_romper_over15", pd.Series(False, index=base.index)).fillna(False)
    )
    rescate = base.loc[fallback_mask].copy()

    filtro_base["Rescate"] = False
    rescate["Rescate"] = True

    candidatos = pd.concat([filtro_base, rescate], ignore_index=True)
    sort_keys = [k for k in ["pex_ev_norm_over", "ev_over", "edge_over"] if k in candidatos.columns]
    if sort_keys:
        candidatos = candidatos.sort_values(sort_keys, ascending=False)

    return _final_over15_df(candidatos)


def build_arg_parser(default_cfg: Dict[str, Any]) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Filtra picks Over 1.5")
    p.add_argument("--min-edge", type=float, default=default_cfg.get("min_edge"))
    p.add_argument("--min-ev", type=float, default=default_cfg.get("min_ev"))
    p.add_argument("--min-odds", type=float, default=default_cfg.get("min_odds"))
    p.add_argument("--max-odds", type=float, default=default_cfg.get("max_odds"))
    p.add_argument("--min-potential-over", type=float, default=default_cfg.get("min_potential_over"))
    p.add_argument("--min-pex-hit", type=float, default=default_cfg.get("min_pex_hit"))
    p.add_argument("--min-pex-ev-norm", type=float, default=default_cfg.get("min_pex_ev_norm"))
    p.add_argument("--csv-out", type=str, default=None, help="Ruta CSV para exportar picks")
    return p


def main_over15(today_match):
    market = "OVER15"

    base = over15_to_df(today_match).copy()
    base = _ensure_over_cols(base)

    preset_cfg: Dict[str, Any] = filters(market, len(base))
    aplicar_filtros = preset_cfg.get("aplicar_filtros", True)

    args = build_arg_parser(preset_cfg).parse_args()

    cfg = {
        "min_edge": args.min_edge,
        "min_ev": args.min_ev,
        "min_odds": args.min_odds,
        "max_odds": args.max_odds,
        "min_potential_over": args.min_potential_over,
        "min_pex_hit": args.min_pex_hit,
        "min_pex_ev_norm": args.min_pex_ev_norm,
    }

    df_over_final = build_over15_picks_df(cfg, today_match, aplicar_filtros=aplicar_filtros)
    return format_over15_output(df_over_final)


def format_over15_output(df_over_final: pd.DataFrame) -> pd.DataFrame:
    df_over_final = df_over_final.copy()
    df_over_final = df_over_final.rename(columns={
        "season_id": "Season_id",
        "hora": "Hora",
        "home": "Local",
        "away": "Visitante",
        "match_id": "Partido",
        "o15_potential": "Potencial",
        "cards_potential": "Tarjetas",
        "OVER": "Mercado",
        "pex_hit_over": "PEX_HIT",
        "pex_ev_norm_over": "PEX_NORM",
        "local_over15_rate": "Local_over15_rate",
        "visita_over15_rate": "Visita_over15_rate",
        "local_under15_streak": "Local_under15_streak",
        "visita_under15_streak": "Visita_under15_streak",
        "local_flag_romper_over15": "Flag_local_romper",
        "visita_flag_romper_over15": "Flag_visita_romper",
    })

    if "PEX_HIT" in df_over_final.columns:
        df_over_final["PROB_Historico"] = (
            pd.to_numeric(df_over_final["PEX_HIT"], errors="coerce") * 100
        ).round(0).astype("Int64")
    if "Prob_modelo" not in df_over_final.columns:
        prob_src = df_over_final.get("Potencial_final", df_over_final.get("Potencial"))
        df_over_final["Prob_modelo"] = (
            pd.to_numeric(prob_src, errors="coerce")
        ).round(0).astype("Int64")

    if "home_id" in df_over_final.columns and "homeID" not in df_over_final.columns:
        df_over_final["homeID"] = df_over_final["home_id"]
    if "away_id" in df_over_final.columns and "awayID" not in df_over_final.columns:
        df_over_final["awayID"] = df_over_final["away_id"]

    df_over_final["Season_label"] = df_over_final.get("season_label")
    if "competition_id" in df_over_final.columns:
        df_over_final["Season_id"] = df_over_final["Season_id"].where(
            df_over_final["Season_id"].notna(),
            df_over_final["competition_id"]
        )
        df_over_final = df_over_final.drop(columns=["competition_id"])

    cols_base = [
        "Partido", "Season_id","Season_label","home_id","away_id","homeID","awayID", "Hora", "Pais", "Liga", "Local", "Visitante",
        "ODDS", "Mercado", "Potencial", "Potencial_final", "Prob_modelo", "Tarjetas",
        "ROI_estimado", "Ventaja_modelo",
        "PROB_Historico",
        "Local_over15_rate", "Visita_over15_rate",
        "Local_under15_streak", "Visita_under15_streak",
        "Flag_local_romper", "Flag_visita_romper",
        "Rescate"
    ]
    cols_base = [c for c in cols_base if c in df_over_final.columns]
    if "Hora" in cols_base:
        df_over_final = df_over_final[cols_base].sort_values("Hora")
    else:
        df_over_final = df_over_final[cols_base]
    df_over_final = anotar_estado(df_over_final, "OVER")
    desired_cols = [
        "Partido", "Season_id","Season_label","home_id","away_id","homeID","awayID", "Hora", "Pais", "Liga", "Local", "Visitante",
        "ODDS", "Mercado", "Marcador", "Potencial", "Potencial_final", "Prob_modelo", "Tarjetas", "Estado",
        "ROI_estimado", "Ventaja_modelo",
        "PROB_Historico",
        "Local_over15_rate", "Visita_over15_rate",
        "Local_under15_streak", "Visita_under15_streak",
        "Flag_local_romper", "Flag_visita_romper",
        "Rescate"
    ]
    df_over_final = df_over_final[[c for c in desired_cols if c in df_over_final.columns]]
    if "PEX_NORM" in df_over_final.columns:
        df_over_final = df_over_final.drop(columns=["PEX_NORM"])
    if "ROI_estimado" in df_over_final.columns:
        df_over_final["ROI_estimado"] = (
            pd.to_numeric(df_over_final["ROI_estimado"], errors="coerce") * 100
        )
    if "Ventaja_modelo" in df_over_final.columns:
        df_over_final["Ventaja_modelo"] = (
            pd.to_numeric(df_over_final["Ventaja_modelo"], errors="coerce") * 100
        )
    df_over_final = _format_common(df_over_final)
    return df_over_final
