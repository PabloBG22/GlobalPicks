# app/ingest/normalizacion/picks_o05ht.py
from typing import Dict, Any, List
import argparse
import pandas as pd
import numpy as np

# OJO: asegúrate que la ruta del módulo coincide con tu archivo real
from app.ingest.normalizacion.normalizacion_05ht import o05ht_to_df
from app.ingest.listado_ligas import enrich_liga_cols
from app.ingest.filtros import filters  # presets abierto/moderado/estricto
from app.ingest.resultados.estado import anotar_estado
from app.ingest.historico.golht_contexto import enriquecer_golht, obtener_partidos_golht


# ---------- helpers de alias y columnas ----------
def _apply_ht_aliases(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    alias = {
        # métricas over 0.5 HT
        "p_model_o05HT": "p_model_ht",
        "p_fair_o05HT":  "p_fair_ht",
        "edge_o05HT":    "edge_ht",
        "ev_o05HT":      "ev_ht",
        "cuota_real_o05HT":  "cuota_real_ht",
        "cuota_model_o05HT": "cuota_model_ht",
        "cuota_justa_o05HT": "cuota_justa_ht",
        "value_pct_model_o05HT": "value_pct_model_ht",
        "value_pct_justa_o05HT": "value_pct_justa_ht",
        "odds_1st_half_over05":  "odds_o05HT",
        "odds_over05_ht":        "odds_o05HT",
        # nuevas PEX
        "pex_hit_o05HT":     "pex_hit_ht",
        "pex_ev_norm_o05HT": "pex_ev_norm_ht",
    }
    for src, dst in alias.items():
        if src in d.columns and dst not in d.columns:
            d = d.rename(columns={src: dst})
    return d


def _to_num_safe(s):
    # Garantiza 1D list-like para to_numeric
    try:
        return pd.to_numeric(pd.Series(s), errors="coerce")
    except Exception:
        # fallback robusto
        return pd.to_numeric(np.asarray(s), errors="coerce")


def _ensure_ht_cols(d: pd.DataFrame) -> pd.DataFrame:
    d = _apply_ht_aliases(d).copy()

    # cuota real desde odds si hace falta
    if "cuota_real_ht" not in d.columns and "odds_o05HT" in d.columns:
        d["cuota_real_ht"] = pd.to_numeric(d["odds_o05HT"], errors="coerce")

    # edge / ev si faltan
    if "edge_ht" not in d.columns and {"p_model_ht","p_fair_ht"} <= set(d.columns):
        d["edge_ht"] = d["p_model_ht"] - d["p_fair_ht"]

    if "ev_ht" not in d.columns and {"p_model_ht","cuota_real_ht"} <= set(d.columns):
        d["ev_ht"] = d["p_model_ht"] * d["cuota_real_ht"] - 1

    # cuotas modelo/justa
    if "cuota_model_ht" not in d.columns and "p_model_ht" in d.columns:
        d["cuota_model_ht"] = d["p_model_ht"].apply(lambda x: (1/x) if pd.notna(x) and x > 0 else np.nan)

    if "cuota_justa_ht" not in d.columns and "p_fair_ht" in d.columns:
        d["cuota_justa_ht"] = d["p_fair_ht"].apply(lambda x: (1/x) if pd.notna(x) and x > 0 else np.nan)

    # ODDS para imprimir
    if "ODDS" not in d.columns and "odds_o05HT" in d.columns:
        d["ODDS"] = pd.to_numeric(d["odds_o05HT"], errors="coerce")

    return d


def _apply_optional_threshold(mask: pd.Series, series: pd.Series, thr: Any, op: str = ">=") -> pd.Series:
    """Aplica un umbral opcional si 'thr' no es None."""
    if thr is None:
        return mask
    if op == ">=":
        return mask & (series.fillna(-1) >= thr)
    elif op == "<=":
        return mask & (series.fillna(1e9) <= thr)
    return mask


# ---------- formato final (salida) ----------
def _final_o05ht_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    d = _ensure_ht_cols(raw_df).copy()
    d["MERCADO"] = "GOL HT"

    if "ev_ht" in d.columns:   d["ROI_estimado"]   = d["ev_ht"]
    if "edge_ht" in d.columns: d["Ventaja_modelo"] = d["edge_ht"]

    d = enrich_liga_cols(d, competition_id_col="competition_id", season_id_col="season_id")

    d = d.rename(columns={"kickoff_cdmx": "hora"})

    cols_out = [
        "match_id","season_id","hora","Pais","Liga","home_id","away_id","home","away",
        "ODDS","MERCADO",
        # Scores
        "pex_ht","pex_band",
        "pex_hit_ht","pex_ev_norm_ht",
        # Extras
        "o05HT_potential","cards_potential","ROI_estimado","Ventaja_modelo",
        "Potencial_final","Prob_modelo",
        "xg_total_prematch","xg_ht_aprox","ht_share",
        "local_matches_ht","local_ht_goal_rate","local_ht_scored_rate",
        "local_ht_conceded_rate","local_ht_score_streak","local_ht_cs_streak",
        "visita_matches_ht","visita_ht_goal_rate","visita_ht_scored_rate",
        "visita_ht_conceded_rate","visita_ht_score_streak","visita_ht_cs_streak"
    ]
    d = d[[c for c in cols_out if c in d.columns]].sort_values("hora").reset_index(drop=True)

    for c in ("ODDS","ROI_estimado","Ventaja_modelo",
              "xg_total_prematch","xg_ht_aprox","ht_share",
              "pex_ht","pex_hit_ht","pex_ev_norm_ht"):
        if c in d.columns:
            s = d[c]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            d[c] = pd.to_numeric(s, errors="coerce").round(3)
    if "o05HT_potential" in d.columns:
        d["o05HT_potential"] = pd.to_numeric(d["o05HT_potential"], errors="coerce").round(0)
    if "p_model_ht" in d.columns:
        prob_series = pd.to_numeric(d["p_model_ht"], errors="coerce")
        d["Potencial_final"] = (prob_series * 100).round(1)
        d["Prob_modelo"] = (prob_series * 100).round(0).astype("Int64")

    return d


# ---------- builder ----------
def build_o05ht_picks_df(cfg: Dict[str, Any], today_match, aplicar_filtros: bool = True) -> pd.DataFrame:
    base = o05ht_to_df(today_match).copy()
    base = _ensure_ht_cols(base)
    base = enriquecer_golht(base)

    # ---- NORMALIZA DTYPES (evita FutureWarning y comparaciones raras) ----
    numeric_cols = [
        "edge_ht","ev_ht","ODDS","o05HT_potential","xg_total_prematch",
        "o15HT_potential","btts_fhg_potential","p_model_ht","p_fair_ht",
        "cuota_real_ht","pex_ht","pex_hit_ht","pex_ev_norm_ht"
    ]
    for c in numeric_cols:
        if c in base.columns:
            base[c] = pd.to_numeric(base[c], errors="coerce")

    if not aplicar_filtros:
        base["Rescate"] = False
        return _final_o05ht_df(base)

    edge  = base["edge_ht"] if "edge_ht" in base.columns else pd.Series(np.nan, index=base.index)
    ev    = base["ev_ht"]   if "ev_ht"   in base.columns else pd.Series(np.nan, index=base.index)
    odds  = base["ODDS"]    if "ODDS"    in base.columns else pd.Series(np.nan, index=base.index)

    # ---- MÁSCARA DE FILTROS PRINCIPALES ----
    mask = (edge.fillna(-1.0) >= cfg["min_edge"]) & (ev.fillna(-1.0) >= cfg["min_ev"])
    mask &= odds.between(cfg["min_odds"], cfg["max_odds"], inclusive="both")

    if "o05HT_potential" in base.columns and "min_potential_ht" in cfg:
        mask &= base["o05HT_potential"].fillna(0) >= cfg["min_potential_ht"]

    # ---- FILTROS OPCIONALES (PEX y señales) ----
    if "min_total_xg" in cfg and "xg_total_prematch" in base.columns:
        mask &= base["xg_total_prematch"].fillna(0) >= cfg["min_total_xg"]
    if "min_pex" in cfg and "pex_ht" in base.columns and cfg["min_pex"] is not None:
        mask &= base["pex_ht"].fillna(0) >= float(cfg["min_pex"])

    # nuevos: PEX hit y EV norm
    if "min_pex_hit" in cfg and "pex_hit_ht" in base.columns and cfg["min_pex_hit"] is not None:
        mask = _apply_optional_threshold(mask, base["pex_hit_ht"], cfg["min_pex_hit"], ">=")
    if "min_pex_ev_norm" in cfg and "pex_ev_norm_ht" in base.columns and cfg["min_pex_ev_norm"] is not None:
        mask = _apply_optional_threshold(mask, base["pex_ev_norm_ht"], cfg["min_pex_ev_norm"], ">=")

    # Señales opcionales (si las usas)
    # if "min_o15ht" in cfg and "o15HT_potential" in base.columns:
    #     mask &= base["o15HT_potential"].fillna(0) >= cfg["min_o15ht"]
    # if "min_btts_fhg" in cfg and "btts_fhg_potential" in base.columns:
    #     mask &= base["btts_fhg_potential"].fillna(0) >= cfg["min_btts_fhg"]

    rescue_df = pd.DataFrame()
    racha_goal_rate = cfg.get("racha_min_goal_rate", 0.65)
    rescue_enabled = False
    if {"local_ht_goal_rate","local_ht_score_streak","visita_ht_goal_rate","visita_ht_score_streak"} <= set(base.columns):
        home_signal = (
            base["local_ht_goal_rate"].fillna(0) >= racha_goal_rate
        ) & (base["local_ht_score_streak"].fillna(0) == 0)
        away_signal = (
            base["visita_ht_goal_rate"].fillna(0) >= racha_goal_rate
        ) & (base["visita_ht_score_streak"].fillna(0) == 0)
        rescue_mask = (~mask) & (home_signal | away_signal)
        rescue_df = base.loc[rescue_mask].copy()
        if not rescue_df.empty:
            rescue_df["Rescate"] = True
            rescue_enabled = True

    filtered_strict = base.loc[mask].copy()
    filtered_strict["Rescate"] = False

    filtered = pd.concat([filtered_strict, rescue_df], ignore_index=True) if rescue_enabled else filtered_strict
    filtered = enriquecer_golht(filtered)

    # orden: PEX_NORM -> EV -> EDGE
    sort_keys = [k for k in ["pex_ev_norm_ht","ev_ht","edge_ht"] if k in filtered.columns]
    if sort_keys:
        filtered = filtered.sort_values(sort_keys, ascending=False)

    return _final_o05ht_df(filtered)


def format_o05ht_output(df_ht: pd.DataFrame, meta_source: pd.DataFrame | None = None) -> pd.DataFrame:
    df_ht_final = df_ht.rename(columns={
        "season_id":"Season_id","hora": "Hora",
        "home": "Local", "away": "Visitante",
        "match_id": "Partido","o05HT_potential": "Potencial",
        "cards_potential": "Tarjetas",
        "MERCADO":"Mercado","pex_hit_ht":"PEX_HIT", "pex_ev_norm_ht":"PEX_NORM",
        "local_matches_ht":"Local_ht_partidos",
        "local_ht_goal_rate":"Local_ht_goal_rate",
        "local_ht_scored_rate":"Local_ht_scored_rate",
        "local_ht_conceded_rate":"Local_ht_conceded_rate",
        "local_ht_score_streak":"Local_ht_score_streak",
        "local_ht_cs_streak":"Local_ht_cs_streak",
        "visita_matches_ht":"Visita_ht_partidos",
        "visita_ht_goal_rate":"Visita_ht_goal_rate",
        "visita_ht_scored_rate":"Visita_ht_scored_rate",
        "visita_ht_conceded_rate":"Visita_ht_conceded_rate",
        "visita_ht_score_streak":"Visita_ht_score_streak",
        "visita_ht_cs_streak":"Visita_ht_cs_streak"
    }).copy()

    if "PEX_HIT" in df_ht_final.columns:
        df_ht_final["PROB_Historico"] = (
            pd.to_numeric(df_ht_final["PEX_HIT"], errors="coerce") * 100
        ).round(0).astype("Int64")

    meta_df = meta_source if meta_source is not None else df_ht
    if isinstance(meta_df, pd.DataFrame):
        if "season_label" in meta_df.columns and "Season_label" not in df_ht_final.columns:
            df_ht_final["Season_label"] = meta_df["season_label"]
        if "competition_id" in meta_df.columns:
            df_ht_final["competition_id"] = meta_df["competition_id"]
        home_series = meta_df["home_id"] if "home_id" in meta_df.columns else meta_df.get("homeID")
        away_series = meta_df["away_id"] if "away_id" in meta_df.columns else meta_df.get("awayID")
        season_series = meta_df["season_id"] if "season_id" in meta_df.columns else meta_df.get("seasonId")
        meta_cols = {
            "_competition_id": meta_df.get("competition_id"),
            "_season_label": meta_df.get("season_label"),
            "_home_id": home_series,
            "_away_id": away_series,
            "season_id": season_series,
            "home_id": home_series,
            "away_id": away_series,
            "homeID": meta_df["homeID"] if "homeID" in meta_df.columns else home_series,
            "awayID": meta_df["awayID"] if "awayID" in meta_df.columns else away_series,
        }
        for col, series in meta_cols.items():
            if isinstance(series, pd.Series):
                aligned = series.reindex(df_ht_final.index)
                df_ht_final[col] = aligned.values

    if "Rescate" not in df_ht_final.columns:
        df_ht_final["Rescate"] = False
    if "Prob_modelo" not in df_ht_final.columns:
        prob_src = df_ht_final.get("Potencial_final", df_ht_final.get("Potencial"))
        df_ht_final["Prob_modelo"] = (
            pd.to_numeric(prob_src, errors="coerce")
        ).round(0).astype("Int64")

    cols_base = [
        "Partido","Season_id","Season_label","competition_id","season_id","home_id","away_id","homeID","awayID",
        "Hora", "Pais", "Liga", "Local", "Visitante",
        "ODDS", "Mercado", "Potencial","Potencial_final","Prob_modelo","Tarjetas",
        "ROI_estimado", "Ventaja_modelo",
        "Local_ht_partidos","Local_ht_goal_rate","Local_ht_scored_rate",
        "Local_ht_conceded_rate","Local_ht_score_streak","Local_ht_cs_streak",
        "Visita_ht_partidos","Visita_ht_goal_rate","Visita_ht_scored_rate",
        "Visita_ht_conceded_rate","Visita_ht_score_streak","Visita_ht_cs_streak",
        "PROB_Historico","Rescate"
    ]
    cols_base = [c for c in cols_base if c in df_ht_final.columns]
    if "Hora" in cols_base:
        df_ht_final = df_ht_final[cols_base].sort_values("Hora")
    else:
        df_ht_final = df_ht_final[cols_base]
    df_ht_final = anotar_estado(df_ht_final, "GOLHT")
    desired_cols = [
        "Partido","Season_id","Season_label","competition_id","season_id","home_id","away_id","homeID","awayID",
        "Hora", "Pais", "Liga", "Local", "Visitante",
        "ODDS", "Mercado", "Marcador_HT","Potencial","Potencial_final","Prob_modelo","Tarjetas",
        "Local_ht_partidos","Local_ht_goal_rate","Local_ht_scored_rate",
        "Local_ht_conceded_rate","Local_ht_score_streak","Local_ht_cs_streak",
        "Visita_ht_partidos","Visita_ht_goal_rate","Visita_ht_scored_rate",
        "Visita_ht_conceded_rate","Visita_ht_score_streak","Visita_ht_cs_streak",
        "Estado","ROI_estimado", "Ventaja_modelo",
        "PROB_Historico","Rescate"
    ]
    df_ht_final = df_ht_final[[c for c in desired_cols if c in df_ht_final.columns]]
    if "PEX_NORM" in df_ht_final.columns:
        df_ht_final = df_ht_final.drop(columns=["PEX_NORM"])
    if "ROI_estimado" in df_ht_final.columns:
        df_ht_final["ROI_estimado"] = (
            pd.to_numeric(df_ht_final["ROI_estimado"], errors="coerce") * 100
        )
    if "Ventaja_modelo" in df_ht_final.columns:
        df_ht_final["Ventaja_modelo"] = (
            pd.to_numeric(df_ht_final["Ventaja_modelo"], errors="coerce") * 100
        )
    df_ht_final = _format_common(df_ht_final)
    return df_ht_final


# ---------- argparse ----------
def build_arg_parser_ht(default_cfg: Dict[str, Any]) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Filtra picks Gol HT (Over 0.5)")
    p.add_argument("--nivel", choices=["abierto","moderado","estricto"], default=None)
    p.add_argument("--debug", action="store_true")

    p.add_argument("--min-edge", type=float, default=default_cfg.get("min_edge"))
    p.add_argument("--min-ev", type=float, default=default_cfg.get("min_ev"))
    p.add_argument("--min-odds", type=float, default=default_cfg.get("min_odds"))
    p.add_argument("--max-odds", type=float, default=default_cfg.get("max_odds"))

    p.add_argument("--min-potential-ht", type=float, default=default_cfg.get("min_potential_ht"))
    p.add_argument("--min-total-xg", type=float, default=default_cfg.get("min_total_xg"))
    p.add_argument("--min-o15ht", type=float, default=default_cfg.get("min_o15ht"))
    p.add_argument("--min-btts-fhg", type=float, default=default_cfg.get("min_btts_fhg"))

    # PEX compuesto original
    p.add_argument("--min-pex", type=float, default=default_cfg.get("min_pex"),
                   help="Umbral mínimo de PEX HT (score compuesto) en [0,1]")

    # NUEVOS: umbrales por PEX_HIT y PEX_EV_NORM
    p.add_argument("--min-pex-hit", type=float, default=default_cfg.get("min_pex_hit"),
                   help="Umbral mínimo para pex_hit_ht (probabilidad modelo) en [0,1]")
    p.add_argument("--min-pex-ev-norm", type=float, default=default_cfg.get("min_pex_ev_norm"),
                   help="Umbral mínimo para pex_ev_norm_ht (EV normalizado) en [0,1]")

    # Banda PEX
    p.add_argument("--pex-band", choices=["abierto","moderado","estricto"],
                   default=default_cfg.get("pex_band"),
                   help="Filtra por banda de PEX calculada")

    return p


def debug_o05ht(df: pd.DataFrame, cfg: Dict[str, Any]):
    d = df.copy()
    for c in ["edge_ht","ev_ht","ODDS","o05HT_potential","xg_total_prematch",
              "o15HT_potential","btts_fhg_potential","p_model_ht","pex_ht",
              "pex_hit_ht","pex_ev_norm_ht"]:
        if c in d.columns: d[c] = pd.to_numeric(d[c], errors="coerce")

    with np.errstate(divide="ignore", invalid="ignore"):
        d["odds_needed_min_ev"] = (1 + cfg.get("min_ev", 0)) / d["p_model_ht"]

    checks = {
        "edge_ok": d["edge_ht"] >= cfg.get("min_edge", 0),
        "ev_ok":   d["ev_ht"]   >= cfg.get("min_ev", 0),
        "odds_ok": d["ODDS"].between(cfg.get("min_odds", 0), cfg.get("max_odds", 999), inclusive="both"),
        "pot_ok":  d["o05HT_potential"].fillna(0) >= cfg.get("min_potential_ht", 0),
    }
    if "min_total_xg" in cfg and "xg_total_prematch" in d: checks["xg_ok"] = d["xg_total_prematch"].fillna(0) >= cfg["min_total_xg"]
    if "min_o15ht" in cfg and "o15HT_potential" in d:     checks["o15_ok"] = d["o15HT_potential"].fillna(0) >= cfg["min_o15ht"]
    if "min_btts_fhg" in cfg and "btts_fhg_potential" in d: checks["fhg_ok"] = d["btts_fhg_potential"].fillna(0) >= cfg["min_btts_fhg"]
    if "min_pex" in cfg and "pex_ht" in d.columns: checks["pex_ok"] = d["pex_ht"].fillna(0) >= float(cfg["min_pex"])
    if "min_pex_hit" in cfg and "pex_hit_ht" in d.columns: checks["pex_hit_ok"] = d["pex_hit_ht"].fillna(0) >= float(cfg["min_pex_hit"])
    if "min_pex_ev_norm" in cfg and "pex_ev_norm_ht" in d.columns: checks["pex_norm_ok"] = d["pex_ev_norm_ht"].fillna(0) >= float(cfg["min_pex_ev_norm"])

    cols = [c for c in [
        "home","away","ODDS","p_model_ht","p_fair_ht","edge_ht","ev_ht",
        "pex_ht","pex_band","pex_hit_ht","pex_ev_norm_ht",
        "o05HT_potential","o15HT_potential","btts_fhg_potential",
        "xg_total_prematch","odds_needed_min_ev"
    ] if c in d.columns]
    # print(d.sort_values(["pex_ev_norm_ht","ev_ht","edge_ht"], ascending=False)[cols].head(8).to_string(index=False))


# ---------- main ----------
def main_o05ht(today_match):
    market = "O05HT"

    # Base
    base = o05ht_to_df(today_match).copy()
    base = _ensure_ht_cols(base)

    # --- Fase 0: parseo ligero para capturar --nivel y --debug primero ---
    p0 = argparse.ArgumentParser(add_help=False)
    p0.add_argument("--nivel", choices=["abierto","moderado","estricto"])
    p0.add_argument("--debug", action="store_true")
    args0, rem = p0.parse_known_args()

    # Determina preset (auto por volumen o forzado por --nivel)
    from app.ingest.filtros import FILTERS
    preset_auto = filters(market, len(base))
    preset_cfg = FILTERS[market][args0.nivel].copy() if args0.nivel else preset_auto
    aplicar_filtros = preset_cfg.get("aplicar_filtros", True)

    # --- Fase 1: parser completo con defaults del preset elegido ---
    args = build_arg_parser_ht(preset_cfg).parse_args(rem)

    # Config efectiva (si el usuario pasa flags, pisa los defaults)
    cfg = {
        "min_edge": args.min_edge,
        "min_ev": args.min_ev,
        "min_odds": args.min_odds,
        "max_odds": args.max_odds,
        "min_potential_ht": args.min_potential_ht,
        "min_total_xg": args.min_total_xg,
        "min_o15ht": args.min_o15ht,
        "min_btts_fhg": args.min_btts_fhg,
        "min_pex": args.min_pex,
        "pex_band": args.pex_band,
        # nuevos umbrales PEX
        "min_pex_hit": args.min_pex_hit,
        "min_pex_ev_norm": args.min_pex_ev_norm,
    }

    # Debug opcional
    if args0.debug or getattr(args, "debug", False):
        # print(f"[O05HT] Nivel={'forzado:'+args0.nivel if args0.nivel else 'auto'}  CFG={cfg}")
        debug_o05ht(base, cfg)

    # Construye picks
    df_ht = build_o05ht_picks_df(cfg, today_match, aplicar_filtros=aplicar_filtros)
    return format_o05ht_output(df_ht, meta_source=df_ht)
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


def _format_ht_score(gf, ga):
    if pd.isna(gf) and pd.isna(ga):
        return "-"
    try:
        gf_val = int(gf) if not pd.isna(gf) else 0
        ga_val = int(ga) if not pd.isna(ga) else 0
        return f"{gf_val}-{ga_val}"
    except Exception:
        return "-"


def _historial_por_equipo(df_matches: pd.DataFrame, team_id: int, max_rows: int | None = None) -> pd.DataFrame:
    if df_matches.empty or pd.isna(team_id):
        return pd.DataFrame()
    subset = df_matches.loc[
        (df_matches["_home_id"] == team_id) | (df_matches["_away_id"] == team_id)
    ].copy()
    if subset.empty:
        return pd.DataFrame()
    subset = subset.sort_values("match_ts", ascending=False)
    subset["Condicion"] = np.where(subset["_home_id"] == team_id, "Local", "Visitante")
    subset["Rival"] = np.where(subset["Condicion"] == "Local", subset["_away_name"], subset["_home_name"])
    subset["_ht_for"] = np.where(subset["Condicion"] == "Local", subset["_ht_home"], subset["_ht_away"])
    subset["_ht_against"] = np.where(subset["Condicion"] == "Local", subset["_ht_away"], subset["_ht_home"])
    subset["Marcador_HT"] = subset.apply(lambda r: _format_ht_score(r["_ht_for"], r["_ht_against"]), axis=1)
    subset["Fecha"] = subset["match_ts"].dt.strftime("%Y-%m-%d")
    subset["Liga"] = subset["_league_name"]
    subset["o05HT_potential"] = pd.to_numeric(subset["_o05ht_potential"], errors="coerce")
    cols = [
        "Fecha", "Liga", "Condicion", "Rival",
        "ht_goals_team_a", "ht_goals_team_b", "HTGoalCount",
        "o05HT_potential", "Marcador_HT"
    ]
    if max_rows is not None:
        subset = subset.head(max_rows)
    return subset[cols].reset_index(drop=True)


def build_golht_detalle(df_picks: pd.DataFrame, max_historial: int | None = None) -> List[Dict[str, Any]]:
    if df_picks is None or df_picks.empty:
        return []
    required_cols = {"Season_id", "_competition_id", "_home_id", "_away_id"}
    if not required_cols <= set(df_picks.columns):
        return []

    detalles = []
    cache: dict[tuple, pd.DataFrame] = {}
    df_iter = df_picks.reset_index(drop=True)
    for idx, row in df_iter.iterrows():
        comp_id = row.get("_competition_id")
        season_id = row.get("Season_id")
        season_label = row.get("_season_label")
        home_id = row.get("_home_id")
        away_id = row.get("_away_id")
        if pd.isna(comp_id) or pd.isna(home_id) or pd.isna(away_id):
            continue
        key = (
            int(comp_id),
            int(season_id) if pd.notna(season_id) else None,
            str(season_label) if season_label not in (None, "") and not pd.isna(season_label) else None,
        )
        if key not in cache:
            cache[key] = obtener_partidos_golht(*key)
        matches = cache.get(key, pd.DataFrame())
        detalle = {
            "numero": idx + 1,
            "local": row.get("Local"),
            "visitante": row.get("Visitante"),
            "local_historial": _historial_por_equipo(matches, int(home_id) if pd.notna(home_id) else None, max_historial),
            "visitante_historial": _historial_por_equipo(matches, int(away_id) if pd.notna(away_id) else None, max_historial),
        }
        detalles.append(detalle)
    return detalles
