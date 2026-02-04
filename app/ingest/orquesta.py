
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import numpy as np
import calendar
import requests
import json
import math
import time
from datetime import datetime,date, timezone,timedelta
from pathlib import Path
from uuid import uuid4
from tabulate import tabulate
from typing import Any, Iterable, Optional
from functools import lru_cache

from pandas.tseries.offsets import Day

from app.common.config import settings
from app.common.db import get_engine
from app.ingest.today_match import get_todays_matches_normalized
from app.ingest.listado_ligas import enrich_liga_cols
from app.ingest.historico.golht_contexto import obtener_partidos_golht
from app.ingest.mercados.btts_yes_no import main_btts, build_btts_picks_df
from app.ingest.mercados.over15 import main_over15, build_over15_picks_df, format_over15_output
from app.ingest.mercados.golht import main_o05ht, build_o05ht_picks_df, format_o05ht_output
from app.ingest.filtros import FILTERS, filters
from app.ingest.maestro_schema import reform_maestro, ensure_maestro_aliases
# from app.ingest.mercados.corners import main_corners  # sigue en pausa
# from app.ingest.mercados.corners_winner import build_corners_winner_df  # Winner Córners deshabilitado
from app.ingest.resultados.estado import (
    anotar_estado,
    _find_match_league,
    _find_match_data,
    _status_finalizado,
    _status_en_juego,
)
from app.common.telegram_client import get_default_client, Prueba
from scripts.validacion_gpt import resultados_gpt

MASTER_DIR = Path("df_maestro")
MASTER_FILE = MASTER_DIR / "maestro_picks_v2.pkl"
MASTER_FILE_RAW = MASTER_DIR / "maestro_picks.pkl"
MASTER_TABLE = "maestro_picks_v2"
MASTER_TABLE_RAW = "maestro_picks"
MASTER_KEYS = ["Fecha_ejecucion", "market_group", "Match_id", "Mercado"]
EXEC_AVG_ODDS = 1.80  # Piso mínimo para métricas _EXE
TELEGRAM_STATUS_COL = "Telegram_estado"
TELEGRAM_MESSAGE_ID_COL = "Telegram_message_id"


def print_markdown_table(df, title):

    if df.empty:
        print(f"\n⚠️ No hay picks disponibles para {title}")
        return
    df_to_print = df.copy()
    if "Partido" in df_to_print.columns:
        df_to_print.insert(0, "#", range(1, len(df_to_print) + 1))
        df_to_print = df_to_print.drop(columns=["Partido"])
    prioridad = ["Season_id", "Season_label", "season_id", "season_label", "home_id", "away_id", "homeID", "awayID"]
    cols_prioritarios = [c for c in prioridad if c in df_to_print.columns]
    cols_restantes = [c for c in df_to_print.columns if c not in cols_prioritarios]
    if cols_prioritarios:
        df_to_print = df_to_print[cols_prioritarios + cols_restantes]
    print(f"\n### {title}")
    print(tabulate(df_to_print, headers='keys', tablefmt='rounded_outline', showindex=False, floatfmt=".3f"))


def _print_resumen_dual(df: pd.DataFrame, title: str):
    """
    Imprime dos tablas: pre-partido (campos normales) y ejecutadas (campos *_EXE sin sufijo).
    """
    if df is None or df.empty:
        print(f"\n⚠️ No hay datos para {title}")
        return
    pre_cols = [c for c in ["Mercado", "Aciertos", "Fallos", "Nulos", "Total", "PEX", "ROI", "AVG_ODDS", "Unidades"] if c in df.columns]
    if pre_cols:
        print_markdown_table(df[pre_cols], f"{title} - Cuotas Prepartido")
    exe_map = {
        "PEX_EXE": "PEX",
        "ROI_EXE": "ROI",
        "AVG_ODDS_EXE": "AVG_ODDS",
        "Unidades_EXE": "Unidades",
    }
    exe_cols_present = [c for c in exe_map if c in df.columns]
    if exe_cols_present:
        base_cols = [c for c in ["Mercado", "Aciertos", "Fallos", "Nulos", "Total"] if c in df.columns]
        exe_cols = base_cols + exe_cols_present
        exe_df = df[exe_cols].rename(columns=exe_map)
        print_markdown_table(exe_df, f"{title} - Cuotas Ejecutadas")


def _ajustar_nulos_por_id(fecha: str):
    """
    Permite marcar picks como NULO por ID consecutivo y ajusta ODDS_metricas:
    - IDs indicados -> Estado NULO
    - ODDS_metricas = max(ODDS, 1.80) cuando existe ODDS
    """
    df = _maestro_por_fecha(fecha)
    if df.empty:
        print(f"\n⚠️ No hay picks en maestro para {fecha}.")
        return

    # Normaliza columnas clave
    if "Match_id" not in df.columns and "Partido" in df.columns:
        df["Match_id"] = df["Partido"]
    if "Partido" not in df.columns and "Match_id" in df.columns:
        df["Partido"] = df["Match_id"]
    if "Local" not in df.columns and "Home" in df.columns:
        df["Local"] = df["Home"]
    if "Visitante" not in df.columns and "Visita" in df.columns:
        df["Visitante"] = df["Visita"]
    df = _normalize_estado_column(df)

    listado = df.copy()
    listado.insert(0, "ID", range(1, len(listado) + 1))
    cols_display = [c for c in ["ID", "Match_id", "Hora", "Pais", "Liga", "Local", "Visitante", "Mercado", "Estado"] if c in listado.columns]
    print_markdown_table(listado[cols_display], f"Picks {fecha} (selección NULO)")

    entrada = input("IDs o Match_id de picks NULO (separados por coma/espacio, vacío para omitir): ").strip()
    if not entrada:
        print("No se realizaron ajustes de NULO.")
        return
    import re
    tokens = [x for x in re.split(r"[,\s;]+", entrada) if x]
    ids_nulos = set()
    match_ids_nulos = set()
    for tok in tokens:
        if tok.isdigit():
            ids_nulos.add(int(tok))
            match_ids_nulos.add(tok)  # también interpreta números como Match_id
        else:
            match_ids_nulos.add(tok)
    if not ids_nulos and not match_ids_nulos:
        print("No se realizaron ajustes de NULO.")
        return

    def _calc_odds_metricas(row):
        odd = pd.to_numeric(row.get("ODDS"), errors="coerce")
        if pd.notna(odd):
            return EXEC_AVG_ODDS if odd < EXEC_AVG_ODDS else odd
        return np.nan

    # Expand IDs a Match_id para anular todas las filas del mismo partido
    match_ids_from_ids = set(listado.loc[listado["ID"].isin(ids_nulos), "Match_id"].astype(str).tolist())
    match_ids_nulos |= match_ids_from_ids

    mask_nulo = listado["ID"].isin(ids_nulos)
    if match_ids_nulos:
        mask_nulo = mask_nulo | listado["Match_id"].astype(str).isin(match_ids_nulos)

    df.loc[mask_nulo, "Estado"] = "NULO"
    df.loc[mask_nulo, "Estado_EXE"] = "NULO"
    # Recalcula ODDS_metricas para todos (piso 1.80 a odds menores)
    df["ODDS_metricas"] = df.apply(_calc_odds_metricas, axis=1)
    df = _normalize_estado_column(df)
    df = _annotate_unidades(df)
    _upsert_maestro(df)
    nulos_count = int(mask_nulo.sum())
    print(f"✅ Ajustados {nulos_count} picks a NULO y recalculados con piso 1.80 en ODDS.")
    if nulos_count:
        print(df[df["Estado"] == "NULO"][["Match_id", "Mercado", "Estado"]])

def _calc_pex_value(aciertos: int, total: int) -> str:
    if total and total > 0:
        return str(int(round((aciertos / total) * 100)))
    return "0"

def _odds_metricas(df: pd.DataFrame, mercado: str | None = None) -> pd.Series | None:
    """
    Devuelve la serie de odds a usar en métricas/unidades.
    - Aplica piso de 1.80 para odds < 1.80.
    """
    if not isinstance(df, pd.DataFrame) or "ODDS" not in df.columns:
        return None
    odds = pd.to_numeric(df["ODDS"], errors="coerce")
    return odds.apply(lambda x: EXEC_AVG_ODDS if pd.notna(x) and x < EXEC_AVG_ODDS else x)


def _merge_odds_metricas(df: pd.DataFrame, mercado: str | None = None) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df
    calc = _odds_metricas(df, mercado)
    if calc is None:
        return df
    if "ODDS_metricas" not in df.columns:
        df["ODDS_metricas"] = calc
        return df
    existing = pd.to_numeric(df["ODDS_metricas"], errors="coerce")
    df["ODDS_metricas"] = existing.where(calc.isna(), calc)
    return df


def _calc_metricas_resumen(estados: pd.Series, odds: pd.Series | None = None, odds_override: float | None = None) -> dict:
    estados = estados.astype(str).str.upper()
    aciertos = int((estados == "VERDE").sum())
    fallos = int((estados == "ROJO").sum())
    nulos = int((estados == "NULO").sum())
    total_eff = aciertos + fallos  # excluye nulos

    odds_avg = None
    if odds is not None:
        mask_valid = estados.isin(["VERDE", "ROJO"])
        odds_clean = pd.to_numeric(odds.where(mask_valid), errors="coerce").dropna()
        if not odds_clean.empty:
            odds_avg = round(float(odds_clean.mean()), 3)

    odds_use = odds_override if odds_override is not None else odds_avg
    # Si no hay aciertos/fallos, evita promedios ficticios
    if total_eff == 0:
        odds_use = None
    unidades = roi = None
    if odds_use is not None and total_eff:
        unidades = round((aciertos * (odds_use - 1)) - fallos, 3)
        roi = round((unidades / total_eff) * 100, 3)

    pex = _calc_pex_value(aciertos, total_eff)
    try:
        pex_int = int(pex)
    except Exception:
        pex_int = pex

    return {
        "Aciertos": aciertos,
        "Fallos": fallos,
        "Nulos": nulos,
        "Total": total_eff,
        "PEX": pex_int,
        "ROI": roi,
        "AVG_ODDS": odds_use if odds_override is None else odds_override if total_eff else None,
        "Unidades": unidades,
    }

def _mapear_topic_telegram(mercado: str) -> str | None:
    """
    Normaliza el nombre de mercado a la clave usada en Prueba.
    """
    m = (mercado or "").lower()
    if "team over" in m or "over 1.5" in m:
        return "Goles"  # Enviar Team Over 1.5 al hilo de Over/Goles
    if "corner" in m:
        return "Corners"
    if "btts" in m:
        return "Btts"
    if "gol ht" in m or "ht" in m:
        return "Gol ht"
    if "over" in m:
        return "Goles"
    return None

def _pick_column(df: pd.DataFrame) -> str | None:
    """Devuelve el nombre de la columna que identifica el partido."""
    for col in ("Partido", "Match_id", "match_id"):
        if col in df.columns:
            return col
    return None

def _ensure_roi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Completa ROI_estimado cuando falte, usando EV_modelo o Prob_modelo/ODDS.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df
    d = df.copy()
    if "ROI_estimado" not in d.columns:
        d["ROI_estimado"] = np.nan
    if d["ROI_estimado"].isna().any():
        if "EV_modelo" in d.columns:
            ev = pd.to_numeric(d["EV_modelo"], errors="coerce")
            d["ROI_estimado"] = d["ROI_estimado"].fillna(ev * 100)
        prob = pd.to_numeric(d.get("Prob_modelo"), errors="coerce")
        odds = pd.to_numeric(d.get("ODDS"), errors="coerce")
        calc = (prob / 100.0) * odds - 1
        d["ROI_estimado"] = d["ROI_estimado"].fillna(calc * 100)
    return d

def _annotate_unidades(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula unidades ganadas/perdidas por pick: 1 * ODDS si VERDE/Acierto, -1 si ROJO/Fallo.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df
    d = df.copy()
    d = _normalize_estado_column(d)
    if "Estado_EXE" not in d.columns:
        d["Estado_EXE"] = d["Estado"]
    odds = pd.to_numeric(d.get("ODDS_metricas", d.get("ODDS")), errors="coerce")
    estados = d.get("Estado_EXE", d.get("Estado", pd.Series("", index=d.index))).astype(str).str.upper()

    verde_mask = estados == "VERDE"
    rojo_mask = estados == "ROJO"
    nulo_mask = estados == "NULO"

    unidades = np.where(verde_mask, odds - 1.0, 0.0)
    unidades = np.where(rojo_mask, -1.0, unidades)
    unidades = np.where(nulo_mask, 0.0, unidades)
    d["Unidades"] = unidades
    return d


def _normalize_estado_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Homologa Estado tomando Acierto como respaldo y dejando solo la columna Estado
    con valores PENDIENTE/LIVE/VERDE/ROJO.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df
    d = df.copy()
    estado = d.get("Estado", pd.Series("", index=d.index))
    acierto = d.get("Acierto")
    estado_norm = estado.fillna("").astype(str).str.strip()
    if acierto is not None:
        acierto_norm = acierto.fillna("").astype(str).str.strip().str.lower()
        mapped = acierto_norm.map({"acierto": "VERDE", "fallo": "ROJO"})
        estado_norm = estado_norm.where(estado_norm != "", mapped)
    estado_norm = estado_norm.str.upper()
    estado_norm = estado_norm.replace({
        "": "PENDIENTE",
        "PENDING": "PENDIENTE",
        "PENDIENTE": "PENDIENTE",
        "LIVE": "LIVE",
        "EN VIVO": "LIVE",
        "VERDE": "VERDE",
        "ROJO": "ROJO",
        "NULO": "NULO",
        "VOID": "NULO",
        "CANCELADO": "NULO",
        "PUSH": "NULO",
    })
    d["Estado"] = estado_norm
    if "Acierto" in d.columns:
        d = d.drop(columns=["Acierto"])
    return d


def _sync_estado_exe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sincroniza Estado_EXE con Estado cuando el ejecutado está vacío o pendiente.
    Preserva NULO y estados finales ya seteados.
    """
    if not isinstance(df, pd.DataFrame) or df.empty or "Estado" not in df.columns:
        return df
    d = df.copy()
    estado = d["Estado"].fillna("").astype(str).str.strip().str.upper()
    if "Estado_EXE" not in d.columns:
        d["Estado_EXE"] = estado
        return d
    estado_exe = d["Estado_EXE"].fillna("").astype(str).str.strip().str.upper()
    update_mask = estado_exe.isin(["", "PENDIENTE", "PENDING", "LIVE", "EN VIVO", "NAN", "NONE"])
    update_mask |= estado.eq("NULO")
    d.loc[update_mask, "Estado_EXE"] = estado
    d["Estado_EXE"] = d["Estado_EXE"].fillna("").astype(str).str.strip().str.upper()
    return d


def _load_maestro() -> pd.DataFrame:
    if settings.database_url:
        engine = get_engine()
        try:
            df = pd.read_sql_table(MASTER_TABLE, engine)
            return ensure_maestro_aliases(df)
        except Exception:
            try:
                df = pd.read_sql_table(MASTER_TABLE_RAW, engine)
                return ensure_maestro_aliases(df)
            except Exception:
                return pd.DataFrame()
    for path in (MASTER_FILE, MASTER_FILE_RAW):
        if path.exists():
            try:
                df = pd.read_pickle(path)
                return ensure_maestro_aliases(df)
            except Exception:
                return pd.DataFrame()
    return pd.DataFrame()


def _save_maestro(df: pd.DataFrame) -> None:
    df_clean = reform_maestro(df)
    if settings.database_url:
        engine = get_engine()
        df_clean.to_sql(MASTER_TABLE, engine, index=False, if_exists="replace", method="multi", chunksize=1000)
        return
    MASTER_DIR.mkdir(exist_ok=True)
    df_clean.to_pickle(MASTER_FILE)


def _normalizar_claves_maestro(df: pd.DataFrame, key_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """
    Normaliza claves para deduplicar aunque vengan con tipos/case distintos.
    Devuelve el DF normalizado y la lista de columnas a usar en drop_duplicates.
    """
    if df is None or df.empty or not key_cols:
        return df, key_cols
    d = df.copy()
    dedup_cols: list[str] = []
    for key in key_cols:
        if key == "Match_id" and key in d.columns:
            d["_Match_id_norm"] = d[key].astype("string").str.strip()
            dedup_cols.append("_Match_id_norm")
        elif key == "Mercado" and key in d.columns:
            d["_Mercado_norm"] = d[key].astype("string").str.strip().str.upper()
            dedup_cols.append("_Mercado_norm")
        elif key == "market_group" and key in d.columns:
            d["_market_group_norm"] = d[key].astype("string").str.strip().str.upper()
            dedup_cols.append("_market_group_norm")
        elif key == "Fecha_ejecucion" and key in d.columns:
            d["_Fecha_ejecucion_norm"] = d[key].astype("string").str.strip()
            dedup_cols.append("_Fecha_ejecucion_norm")
        else:
            dedup_cols.append(key)
    return d, dedup_cols


def _maestro_por_fecha(fecha: str) -> pd.DataFrame:
    maestro = _load_maestro()
    if maestro is None or maestro.empty:
        return pd.DataFrame()
    fecha_col = "Fecha_ejecucion" if "Fecha_ejecucion" in maestro.columns else None
    if fecha_col is None and "Fecha" in maestro.columns:
        fecha_col = "Fecha"
    if fecha_col is None:
        return pd.DataFrame()
    fecha_key = str(fecha).strip()
    fechas = maestro[fecha_col].astype("string").str.strip().str.slice(0, 10)
    mask = fechas == fecha_key[:10]
    return maestro.loc[mask].copy()


def _hay_pendientes_telegram(fecha: str) -> bool:
    df = _maestro_por_fecha(fecha)
    if df.empty or TELEGRAM_STATUS_COL not in df.columns:
        return False
    estados = df[TELEGRAM_STATUS_COL].astype(str).str.upper()
    return estados.eq("PENDIENTE").any()


def _listado_partidos(today_match: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Devuelve un dataframe compacto con Match_id, hora, liga y equipos para facilitar
    la selección manual de partidos.
    """
    df = pd.DataFrame(today_match)
    if df.empty:
        return df
    df = enrich_liga_cols(df, competition_id_col="competition_id", season_id_col="season_id")
    df = df.rename(columns={
        "match_id": "Match_id",
        "kickoff_local_cdmx": "Hora",
        "home": "Local",
        "away": "Visitante",
    })
    cols = [
        "Match_id", "Hora", "Pais", "Liga", "Local", "Visitante",
        "competition_id", "season_id", "season_label"
    ]
    cols_present = [c for c in cols if c in df.columns]
    listado = df[cols_present].sort_values("Hora").reset_index(drop=True)
    listado.insert(0, "ID", range(1, len(listado) + 1))
    return listado


def _mostrar_listado_partidos(listado: pd.DataFrame) -> None:
    if listado.empty:
        print("\n⚠️ No hay partidos para picks manuales.")
        return
    cols_display = [c for c in ["ID", "Match_id", "Hora", "Liga", "Local", "Visitante"] if c in listado.columns]
    print_markdown_table(listado[cols_display], "Partidos del día (selección manual)")


def _normalizar_manual_mercado(nombre: str | None) -> str | None:
    m = (nombre or "").lower()
    if m in {"1", "gol ht", "golht"}:
        return "Gol HT"
    if m in {"2", "over", "over 1.5", "over1.5"}:
        return "OVER 1.5"
    if m in {"3", "btts"}:
        return "BTTS"
    if m in {"4", "team over", "team over 1.5", "teamover", "teamover15"}:
        return "Team Over 1.5"
    if "btts" in m:
        return "BTTS"
    if "gol" in m and "ht" in m:
        return "Gol HT"
    if "0.5" in m and "ht" in m:
        return "Gol HT"
    if "team" in m and ("1.5" in m or "over" in m):
        return "Team Over 1.5"
    if "over" in m and "1.5" in m:
        return "OVER 1.5"
    return None


def _solicitar_picks_manual(today_match: list[dict[str, Any]]) -> list[dict[str, str]]:
    listado = _listado_partidos(today_match)
    _mostrar_listado_partidos(listado)
    selecciones: list[dict[str, str]] = []
    print("\nMercados: 1) Gol HT  2) Over Goles 1.5  3) BTTS  4) Team Over 1.5 (agrega 3er número: 1 Local / 2 Visitante)")
    prompt = "ID listado o MatchID, mercado_num[, equipo_num] (ej: 2,4,1) [ENTER para terminar]: "
    while True:
        entrada = input(prompt).strip()
        if not entrada:
            break
        partes = [p.strip() for p in entrada.split(",") if p.strip()]
        if len(partes) < 2:
            print("⚠️ Formato no reconocido. Usa '2,3' o '2,4,1'.")
            continue
        match_token, mercado_raw = partes[0], partes[1]
        mercado_norm = _normalizar_manual_mercado(mercado_raw)
        if not mercado_norm:
            print("⚠️ Mercado no soportado. Usa códigos 1-4.")
            continue
        match_id_val = None
        # Permite seleccionar por ID secuencial de la lista
        if match_token.isdigit() and "ID" in listado.columns:
            idx = int(match_token)
            fila = listado.loc[listado["ID"] == idx]
            if fila.empty:
                print(f"⚠️ ID {idx} no encontrado en la lista.")
                continue
            match_id_val = str(fila["Match_id"].iloc[0])
        else:
            match_id_val = match_token
        seleccion = {"match_id": match_id_val, "mercado": mercado_norm}
        if mercado_norm == "Team Over 1.5":
            if len(partes) < 3:
                print("⚠️ Team Over 1.5 requiere tercer número: 1 Local, 2 Visitante.")
                continue
            lado = partes[2].strip()
            if lado not in {"1", "2"}:
                print("⚠️ Usa 1 para Local, 2 para Visitante.")
                continue
            seleccion["oteam"] = "Local" if lado == "1" else "Visitante"
        selecciones.append(seleccion)
    return selecciones


def _extraer_picks_por_match(df: pd.DataFrame, selecciones: list[dict[str, str]], mercado_objetivo: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    pick_col = _pick_column(df)
    if pick_col is None:
        return pd.DataFrame()
    ids = {str(s["match_id"]) for s in selecciones if s.get("mercado") == mercado_objetivo}
    if not ids:
        return pd.DataFrame()
    d = df.copy()
    d[pick_col] = d[pick_col].astype(str)
    subset = d[d[pick_col].isin(ids)].copy()
    if mercado_objetivo == "Team Over 1.5" and "OTeam" in subset.columns:
        seleccion_map = {str(s["match_id"]): s.get("oteam") for s in selecciones if s.get("mercado") == mercado_objetivo}
        subset["OTeam"] = subset["OTeam"].astype(str)
        def _match_side(row):
            target = seleccion_map.get(str(row[pick_col]))
            if not target:
                return True
            return row["OTeam"].strip().lower() == target.lower()
        subset = subset[subset.apply(_match_side, axis=1)]
    if subset.empty:
        return subset
    subset["Manual"] = True
    return subset


def _filtrar_partidos_manual(today_match: list[dict[str, Any]], selecciones: list[dict[str, str]]) -> list[dict[str, Any]]:
    ids = {str(s.get("match_id")).strip() for s in selecciones if s.get("match_id")}
    if not ids:
        return []
    filtrados = []
    for match in today_match:
        match_id = match.get("match_id") or match.get("Match_id") or match.get("id") or match.get("fixture_id")
        if match_id is None:
            continue
        if str(match_id) in ids:
            filtrados.append(match)
    return filtrados


def _construir_picks_manual(selecciones: list[dict[str, str]], today_match: list[dict[str, Any]]) -> dict[str, pd.DataFrame]:
    if not selecciones:
        return {}

    mercados_solicitados = {s["mercado"] for s in selecciones if s.get("mercado")}
    resultado_manual: dict[str, pd.DataFrame] = {}

    if "BTTS" in mercados_solicitados:
        cfg_btts = filters("BTTS", len(today_match))
        df_btts = build_btts_picks_df(cfg_btts, today_match, aplicar_filtros=False)
        resultado_manual["BTTS"] = _extraer_picks_por_match(df_btts, selecciones, "BTTS")

    if "OVER 1.5" in mercados_solicitados:
        cfg_over = FILTERS["OVER15"]["estricto"].copy()
        df_over = build_over15_picks_df(cfg_over, today_match, aplicar_filtros=False)
        df_over_fmt = format_over15_output(df_over)
        resultado_manual["OVER 1.5"] = _extraer_picks_por_match(df_over_fmt, selecciones, "OVER 1.5")

    if "Gol HT" in mercados_solicitados:
        cfg_ht = filters("O05HT", len(today_match))
        df_ht = build_o05ht_picks_df(cfg_ht, today_match, aplicar_filtros=False)
        df_ht_fmt = format_o05ht_output(df_ht, meta_source=df_ht)
        resultado_manual["Gol HT"] = _extraer_picks_por_match(df_ht_fmt, selecciones, "Gol HT")

    if "Team Over 1.5" in mercados_solicitados:
        df_team = _tabla_team_over15(today_match, prob_min=None)
        df_team = _anotar_estado_team_over15(df_team)
        resultado_manual["Team Over 1.5"] = _extraer_picks_por_match(df_team, selecciones, "Team Over 1.5")

    return {k: v for k, v in resultado_manual.items() if isinstance(v, pd.DataFrame) and not v.empty}


def _upsert_maestro(df_new: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega/actualiza registros en el maestro usando claves MASTER_KEYS para evitar duplicados.
    """
    if df_new is None or df_new.empty:
        return _load_maestro()
    df_new = _normalize_estado_column(df_new)
    maestro = _load_maestro()
    all_cols = sorted(set(maestro.columns) | set(df_new.columns))
    maestro = maestro.reindex(columns=all_cols)
    df_new = df_new.reindex(columns=all_cols)

    def _norm_telegram_estado(series: pd.Series) -> pd.Series:
        s = series.astype("string").str.strip().str.upper()
        return s.replace({"": pd.NA, "NAN": pd.NA, "NONE": pd.NA})

    if TELEGRAM_STATUS_COL in df_new.columns:
        estado_tel = _norm_telegram_estado(df_new[TELEGRAM_STATUS_COL])
        df_new[TELEGRAM_STATUS_COL] = estado_tel.fillna("PENDIENTE")

    key_cols = [k for k in MASTER_KEYS if k in all_cols]
    combinado = pd.concat([maestro, df_new], ignore_index=True)
    if key_cols:
        combinado_norm, dedup_cols = _normalizar_claves_maestro(combinado, key_cols)
        if TELEGRAM_STATUS_COL in combinado.columns:
            estado_tel = _norm_telegram_estado(combinado[TELEGRAM_STATUS_COL])
            combinado[TELEGRAM_STATUS_COL] = estado_tel
            enviado_mask = estado_tel.eq("ENVIADO")
            if dedup_cols:
                group_keys = [combinado_norm[c] for c in dedup_cols]
                any_enviado = enviado_mask.groupby(group_keys, dropna=False).transform("any")
                combinado.loc[any_enviado, TELEGRAM_STATUS_COL] = "ENVIADO"
        dup_mask = combinado_norm.duplicated(subset=dedup_cols, keep="last")
        combinado = combinado.loc[~dup_mask].copy()
    _save_maestro(combinado)
    return combinado


def _actualizar_pendientes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recalcula Estado solo para filas PENDIENTE/LIVE agrupando por market_group.
    """
    if df is None or df.empty or "market_group" not in df.columns:
        return df
    df = _normalize_estado_column(df)
    grupos_actualizados = []
    for group, subset in df.groupby("market_group"):
        estados = subset.get("Estado", pd.Series("", index=subset.index)).astype(str).str.upper()
        estados_exe = subset.get("Estado_EXE", pd.Series("", index=subset.index)).astype(str).str.upper()
        marcador_series = subset.get("Marcador", pd.Series("", index=subset.index)).astype(str).str.strip()
        mask_pend = estados.isin(["PENDIENTE", "LIVE"])
        marcador_miss = marcador_series.isin(["", "nan"])
        marcador_placeholder = marcador_series.isin(["0-0", "0"])
        mask_nulo = estados.eq("NULO") | estados_exe.eq("NULO")
        mask_target = (mask_pend | marcador_miss | marcador_placeholder) & ~mask_nulo
        if not mask_target.any():
            grupos_actualizados.append(subset)
            continue
        group_key = str(group).upper()
        mercado_param = {
            "GOLHT": "GOLHT",
            "BTTS": "BTTS",
            "OVER": "OVER",
            "TEAM_OVER": "TEAM_OVER",
            "CORNERS": "CORNERS",
        }.get(group_key)
        if mercado_param is None:
            grupos_actualizados.append(subset)
            continue
        recalculado = anotar_estado(subset.loc[mask_target], mercado_param, forzar_busqueda=True)
        actualizado = subset.copy()
        for col in recalculado.columns:
            if col not in actualizado.columns:
                actualizado[col] = pd.NA
            # Evita problemas de dtype (asigna como object para marcadores/cadenas)
            if actualizado[col].dtype != object:
                actualizado[col] = actualizado[col].astype(object)
            actualizado.loc[mask_target, col] = recalculado[col].values
        grupos_actualizados.append(actualizado)
    return pd.concat(grupos_actualizados, ignore_index=True)

def _filtrar_consistencia_prob(df: pd.DataFrame, umbral: float = 12) -> pd.DataFrame:
    """
    Filtra filas donde la diferencia entre Prob_modelo y PROB_Historico excede el umbral dado.
    Si faltan columnas o el DF está vacío, se devuelve sin cambios.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df
    if "Prob_modelo" not in df.columns or "PROB_Historico" not in df.columns:
        return df
    d = df.copy()
    modelo = pd.to_numeric(d["Prob_modelo"], errors="coerce")
    historico = pd.to_numeric(d["PROB_Historico"], errors="coerce")
    diff = (modelo - historico).abs()
    return d.loc[diff <= umbral].reset_index(drop=True)


def _poisson_p_ge2(lam: float | None) -> float | None:
    """P(X >= 2) para X ~ Poisson(lam)."""
    if lam is None:
        return None
    try:
        lmbd = float(lam)
    except (TypeError, ValueError):
        return None
    if lmbd <= 0:
        return None
    return 1.0 - math.exp(-lmbd) * (1.0 + lmbd)


def _parse_goals_field(value: Any) -> int | None:
    """
    Convierte un campo de goles del JSON (puede venir como lista, string "[]", número)
    en el total de goles FT. Devuelve None si no se puede interpretar.
    """
    if value is None:
        return None


def _coalesce_match_id(df: pd.DataFrame) -> pd.Series:
    """Combina posibles columnas de match_id/fixture_id en una sola serie de texto."""
    candidates = [c for c in ("match_id", "_match_id", "id", "fixture_id") if c in df.columns]
    if not candidates:
        return pd.Series(pd.NA, index=df.index)
    result = df[candidates[0]].astype(str)
    for col in candidates[1:]:
        result = result.fillna(df[col].astype(str))
    result = result.replace({"nan": pd.NA, "None": pd.NA})
    return result


def _parse_fecha(fecha_str: str | None) -> Optional[pd.Timestamp]:
    if not fecha_str:
        return None
    try:
        return pd.to_datetime(fecha_str, errors="coerce")
    except Exception:
        return None
    # lista de eventos -> su longitud
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
        try:
            return len(list(value))
        except Exception:
            pass
    # string con formato de lista
    if isinstance(value, str):
        txt = value.strip()
        if txt.startswith("[") and txt.endswith("]"):
            try:
                # intenta contar elementos separados por coma
                inner = txt[1:-1].strip()
                if not inner:
                    return 0
                return len([p for p in inner.split(",") if p.strip() != ""])
            except Exception:
                pass
        try:
            return int(float(txt))
        except Exception:
            pass
    # numérico
    try:
        v = int(float(value))
        return v if v >= 0 else None
    except Exception:
        return None


def _ft_goals_columns(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Devuelve (goles_local, goles_visitante) en FT con fallback de columnas."""
    def _first(colnames):
        for c in colnames:
            if c in df.columns:
                return pd.to_numeric(df[c], errors="coerce")
        return pd.Series(dtype=float, index=df.index)
    home = _first(["homeGoalCount", "homeGoals", "home_goals_ft", "home_goals"])
    away = _first(["awayGoalCount", "awayGoals", "away_goals_ft", "away_goals"])
    return home, away


def _historico_over15_rate(competition_id: int, season_id: int, season_label: str,
                           team_ids: list[int]) -> dict[int, float | None]:
    """Calcula tasa histórica de anotar 2+ goles FT por equipo."""
    if not team_ids:
        return {}
    df = obtener_partidos_golht(competition_id, season_id, season_label)
    if df.empty:
        return {tid: None for tid in team_ids}
    home_ft, away_ft = _ft_goals_columns(df)
    df = df.copy()
    df["homeID"] = pd.to_numeric(df.get("homeID"), errors="coerce")
    df["awayID"] = pd.to_numeric(df.get("awayID"), errors="coerce")
    df["home_ft"] = home_ft
    df["away_ft"] = away_ft

    tasas = {}
    for tid in team_ids:
        mask = (df["homeID"] == tid) | (df["awayID"] == tid)
        subset = df.loc[mask]
        if subset.empty:
            tasas[tid] = None
            continue
        goles = subset.apply(lambda r: r["home_ft"] if r["homeID"] == tid else r["away_ft"], axis=1)
        goles = pd.to_numeric(goles, errors="coerce")
        validos = goles.notna()
        total = validos.sum()
        if not total:
            tasas[tid] = None
            continue
        over = (goles.fillna(0) >= 2).sum()
        tasas[tid] = over / total
    return tasas


def _tabla_team_over15(matches: list[dict], prob_min: float | None = 60.0) -> pd.DataFrame:
    """
    Construye tabla: equipo con mayor probabilidad de anotar 2+ goles (Over 1.5 equipo)
    para cada partido, filtrando por probabilidad mínima (en %). Usa modelo (xG) y
    tasa histórica como respaldo: toma el máximo disponible.
    """
    if not matches:
        return pd.DataFrame()

    registros = []
    for m in matches:
        p_home_model = _poisson_p_ge2(m.get("team_a_xg_prematch"))
        p_away_model = _poisson_p_ge2(m.get("team_b_xg_prematch"))

        home_ft_today = _parse_goals_field(
            m.get("homeGoalCount") or m.get("homeGoals") or m.get("home_goals")
        )
        away_ft_today = _parse_goals_field(
            m.get("awayGoalCount") or m.get("awayGoals") or m.get("away_goals")
        )

        # Históricos por equipo (si hay datos de liga/temporada)
        comp_id = m.get("competition_id")
        season_id_raw = m.get("season_id")
        season_label = m.get("season_label")
        # Si no viene season_id, usa competition_id como respaldo para mantener un identificador
        season_id = season_id_raw
        if season_id in (None, "") and comp_id not in (None, ""):
            season_id = comp_id
        home_id = m.get("home_id") or m.get("homeID")
        away_id = m.get("away_id") or m.get("awayID")
        p_home_hist = p_away_hist = None
        if comp_id not in (None, "") and home_id not in (None, "") and away_id not in (None, ""):
            try:
                tasas = _historico_over15_rate(
                    int(comp_id),
                    int(season_id) if season_id not in (None, "") else None,
                    season_label,
                    [int(home_id), int(away_id)],
                )
                p_home_hist = tasas.get(int(home_id))
                p_away_hist = tasas.get(int(away_id))
            except Exception:
                pass

        p_home = max([p for p in [p_home_model, p_home_hist] if p is not None], default=None)
        p_away = max([p for p in [p_away_model, p_away_hist] if p is not None], default=None)
        opciones = {"Local": p_home, "Visitante": p_away}
        opciones = {k: v for k, v in opciones.items() if v is not None}
        if not opciones:
            continue

        oteam, prob = max(opciones.items(), key=lambda kv: kv[1])
        prob_pct = prob * 100
        if prob_min is not None and prob_pct < prob_min:
            continue

        registros.append({
            "Partido": m.get("match_id"),
            "ID_partido": m.get("match_id"),
            "competition_id": m.get("competition_id"),
            "season_id": season_id,
            "season_label": season_label,
            "home_id": home_id,
            "away_id": away_id,
            "homeGoals_today": home_ft_today,
            "awayGoals_today": away_ft_today,
            "Hora": m.get("kickoff_local_cdmx"),
            "Pais": "",
            "Liga": "",
            "Local": m.get("home"),
            "Visitante": m.get("away"),
            "Mercado": "Team Over 1.5",
            "OTeam": oteam,
            "Probabilidad": round(prob_pct, 1),
            "Cuota_Justa": round((1.0 / prob), 2) if prob and prob > 0 else None,
        })

    if not registros:
        return pd.DataFrame()
    df_out = pd.DataFrame(registros)
    df_out = enrich_liga_cols(df_out, competition_id_col="competition_id", season_id_col="season_id")
    return df_out.sort_values("Probabilidad", ascending=False).reset_index(drop=True)


def _resolver_marcador(
    comp_id: int,
    season_id: int | None,
    season_label: str | None,
    home_id: int,
    away_id: int,
    match_id: Any = None,
    fecha_str: str | None = None,
) -> tuple[float | None, float | None]:
    """Busca el marcador FT combinando match_id, equipos y cercanía de fecha."""
    df = obtener_partidos_golht(comp_id, season_id, season_label)
    if df.empty:
        return None, None
    df = df.copy()
    df["homeID"] = pd.to_numeric(df.get("homeID"), errors="coerce")
    df["awayID"] = pd.to_numeric(df.get("awayID"), errors="coerce")
    home_ft, away_ft = _ft_goals_columns(df)
    df["home_ft"] = home_ft
    df["away_ft"] = away_ft
    df["match_std"] = _coalesce_match_id(df)

    if "date_unix" in df.columns:
        df["match_ts"] = pd.to_datetime(df["date_unix"], unit="s", errors="coerce")
    elif "date" in df.columns:
        df["match_ts"] = pd.to_datetime(df["date"], errors="coerce")
    target_ts = _parse_fecha(fecha_str)

    mask = (df["homeID"] == int(home_id)) & (df["awayID"] == int(away_id))
    # Filtro por match_id si existe
    subset = df.loc[mask]
    if match_id not in (None, ""):
        subset_id = subset.loc[subset["match_std"].astype(str) == str(match_id)]
        if subset_id.empty:
            subset_id = df.loc[df["match_std"].astype(str) == str(match_id)]
        if not subset_id.empty:
            subset = subset_id

    # Si queda vacío, intenta cualquier fila de los equipos
    if subset.empty:
        subset = df.loc[(df["homeID"] == int(home_id)) & (df["awayID"] == int(away_id))]
    if subset.empty:
        return None, None

    # Prioriza por cercanía de fecha si hay target_ts
    if target_ts is not None and "match_ts" in subset.columns:
        subset = subset.assign(delta_ts=(subset["match_ts"] - target_ts).abs())
        subset = subset.sort_values(["delta_ts"])
    elif "date_unix" in subset.columns:
        subset = subset.sort_values("date_unix", ascending=False)

    row = subset.iloc[0]
    return (
        row["home_ft"] if pd.notna(row.get("home_ft")) else None,
        row["away_ft"] if pd.notna(row.get("away_ft")) else None,
    )


def _anotar_estado_team_over15(tabla: pd.DataFrame) -> pd.DataFrame:
    """Marca Acierto/Fallo/Pendiente usando league-matches (season_id + match_id)."""
    if tabla is None or tabla.empty:
        return tabla
    d = tabla.copy()
    estados = []
    goles_ot = []
    marcadores = []
    for _, row in d.iterrows():
        season_id = row.get("season_id") or row.get("Season_id") or row.get("competition_id")
        match_id = row.get("Partido") or row.get("ID_partido")
        home_id = row.get("home_id")
        away_id = row.get("away_id")
        hora = row.get("Hora")
        oteam = str(row.get("OTeam", "")).lower()

        # primero intenta con goles ya presentes (today-matches)
        home_ft = row.get("homeGoals_today")
        away_ft = row.get("awayGoals_today")
        marcador = None
        if home_ft is not None and away_ft is not None:
            marcador = f"{home_ft}-{away_ft}"
        else:
            match_data = {}
            try:
                match_data = _find_match_data(season_id, match_id, home_id, away_id, hora)
            except Exception:
                match_data = {}
            status = str(match_data.get("status") or "")
            if status and not _status_finalizado(status):
                estados.append("LIVE" if _status_en_juego(status) else "PENDIENTE")
                goles_ot.append(None)
                marcadores.append(marcador)
                continue
            home_ft = match_data.get("homeGoalCount") if home_ft is None else home_ft
            away_ft = match_data.get("awayGoalCount") if away_ft is None else away_ft
            if home_ft is None or away_ft is None:
                home_ft = match_data.get("homeGoals")
                away_ft = match_data.get("awayGoals")
            if home_ft is not None and away_ft is not None:
                marcador = f"{home_ft}-{away_ft}"
            else:
                marcador = match_data.get("ft_score") or match_data.get("full_time_score") or match_data.get("score")
                if marcador and isinstance(marcador, str) and "-" in marcador:
                    try:
                        home_ft = int(marcador.split("-")[0].strip())
                        away_ft = int(marcador.split("-")[1].strip())
                    except Exception:
                        pass

        if home_ft is None or away_ft is None:
            estados.append("Pendiente")
            goles_ot.append(None)
            marcadores.append(marcador)
            continue

        goles_eq = home_ft if oteam == "local" else away_ft
        goles_ot.append(goles_eq)
        marcadores.append(marcador or f"{home_ft}-{away_ft}")
        estados.append("VERDE" if goles_eq >= 2 else "ROJO")

    d["Marcador"] = marcadores
    d["Goles_OTeam"] = goles_ot
    d["Estado"] = [e.upper() if isinstance(e, str) else e for e in estados]
    return d


def _fetch_historial_partidos(
    competition_id: int,
    season_id: int,
    season_label: str,
    home_id: int,
    away_id: int,
    home_name: str = "",
    away_name: str = "",
) -> pd.DataFrame:
    if competition_id in (None, "") and season_id in (None, ""):
        return pd.DataFrame()
    df = pd.DataFrame()
    try:
        df = obtener_partidos_golht(
            competition_id,
            int(season_id) if season_id not in (None, "") else None,
            season_label
        )
    except Exception:
        df = pd.DataFrame()
    if df.empty:
        fallback_id = season_id if season_id not in (None, "") else competition_id
        try:
            params = {
                "key": settings.footystats_key,
                "season_id": int(fallback_id),
                "max_per_page": 1000,
            }
            resp = requests.get(f"{settings.footystats_url}/league-matches", params=params, timeout=20)
            resp.raise_for_status()
            df = pd.DataFrame(resp.json().get("data", resp.json()))
        except Exception:
            return pd.DataFrame()
    if df.empty:
        return df

    cols = [
        "date_unix", "homeID", "awayID", "status", "_home_name", "_away_name",
        "ht_goals_team_a", "ht_goals_team_b",
        "HTGoalCount", "o05HT_potential",
        "homeGoalCount", "awayGoalCount", "totalGoalCount",
        "team_a_corners", "team_b_corners",
        "team_a_cards_num", "team_b_cards_num",
    ]
    cols_present = [c for c in cols if c in df.columns]
    if not cols_present:
        return pd.DataFrame()

    subset = df.copy()
    if "status" in subset.columns:
        subset = subset[subset["status"] == "complete"]
    subset["Fecha"] = pd.to_datetime(subset["date_unix"], unit="s", errors="coerce")
    mask = pd.Series(True, index=subset.index)
    if "homeID" in subset.columns and "awayID" in subset.columns:
        mask = (
            (subset["homeID"].astype("Int64") == int(home_id)) |
            (subset["awayID"].astype("Int64") == int(home_id)) |
            (subset["homeID"].astype("Int64") == int(away_id)) |
            (subset["awayID"].astype("Int64") == int(away_id))
        )
    subset = subset.loc[mask]
    if subset.empty and {"_home_name", "_away_name"} <= set(subset.columns):
        home_name = (home_name or "").strip().lower()
        away_name = (away_name or "").strip().lower()
        if home_name or away_name:
            mask_names = (
                subset["_home_name"].astype(str).str.lower().eq(home_name) |
                subset["_away_name"].astype(str).str.lower().eq(home_name) |
                subset["_home_name"].astype(str).str.lower().eq(away_name) |
                subset["_away_name"].astype(str).str.lower().eq(away_name)
            )
            subset = subset.loc[mask_names]
    subset["Equipo_base"] = f"{home_id}-{away_id}"
    return subset[[c for c in cols_present] + ["Fecha", "Equipo_base"]].reset_index(drop=True)


def _build_resultados_historicos(df_picks: pd.DataFrame) -> pd.DataFrame:
    if df_picks is None or df_picks.empty:
        return pd.DataFrame()
    registros = []
    temp_dir = Path("tmp_resultados")
    temp_dir.mkdir(exist_ok=True)
    for _, pick in df_picks.iterrows():
        season_id = pick.get("Season_id") or pick.get("season_id")
        competition_id = pick.get("competition_id")
        season_label = pick.get("Season_label") or pick.get("season_label")
        home_id = pick.get("homeID") or pick.get("home_id")
        away_id = pick.get("awayID") or pick.get("away_id")
        if any(pd.isna(val) for val in (home_id, away_id)):
            continue
        if competition_id in (None, "") and (season_id in (None, "") or pd.isna(season_id)):
            continue
        historial = _fetch_historial_partidos(
            competition_id,
            season_id,
            season_label,
            int(home_id),
            int(away_id),
            str(pick.get("Local") or ""),
            str(pick.get("Visitante") or pick.get("Visita") or "")
        )
        if historial.empty:
            continue
        historial = historial.copy()
        pick_id = pick.get("Partido") or pick.get("Match_id")
        home_name = pick.get("Local")
        away_name = pick.get("Visitante") or pick.get("Visita")
        historial.insert(0, "Pick", pick_id)
        historial.insert(1, "Local", home_name)
        historial.insert(2, "Visitante", away_name)
        pick_payload = {k: (v if not isinstance(v, (float, int)) or not pd.isna(v) else None)
                        for k, v in pick.to_dict().items()}
        hist_payload = historial.to_dict(orient="records")
        uid = uuid4().hex
        pick_json = temp_dir / f"pick_{pick_id}_{uid}.json"
        hist_json = temp_dir / f"hist_{pick_id}_{uid}.json"
        with pick_json.open("w", encoding="utf-8") as f:
            json.dump(pick_payload, f, ensure_ascii=False, indent=2, default=str)
        with hist_json.open("w", encoding="utf-8") as f:
            json.dump(hist_payload, f, ensure_ascii=False, indent=2, default=str)
        historial["json_pick"] = str(pick_json)
        historial["json_hist"] = str(hist_json)
        registros.append(historial)
    if not registros:
        return pd.DataFrame()
    return pd.concat(registros, ignore_index=True)


@lru_cache(maxsize=1024)
def _fetch_match_trends(match_id: str | int | None) -> dict:
    if match_id in (None, ""):
        return {}
    try:
        if pd.isna(match_id):
            return {}
    except Exception:
        pass
    try:
        url = f"{settings.footystats_url}/match"
        params = {"key": settings.footystats_key, "match_id": match_id}
        resp = requests.get(url, params=params, timeout=12)
        resp.raise_for_status()
        payload = resp.json()
        if not isinstance(payload, dict):
            return {}
        data = payload.get("data", {})
        if isinstance(data, list) and data:
            data = data[0]
        if not isinstance(data, dict):
            return {}
        trends = data.get("trends")
        return trends if isinstance(trends, dict) else {}
    except Exception:
        return {}


def _format_trends_text(trends: dict, local: str = "", visitante: str = "") -> str:
    if not trends:
        return ""

    def _entries(entries: Any, label: str) -> str:
        if not entries:
            return ""
        lines: list[str] = []
        for entry in entries:
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                kind = str(entry[0]).strip()
                text = str(entry[1]).strip()
                if text:
                    prefix = f"{kind}: " if kind else ""
                    lines.append(f"- {prefix}{text}")
            elif isinstance(entry, str):
                if entry.strip():
                    lines.append(f"- {entry.strip()}")
            else:
                text = str(entry).strip()
                if text:
                    lines.append(f"- {text}")
        if not lines:
            return ""
        return f"{label}:\n" + "\n".join(lines)

    home_label = f"Local ({local})" if local else "Local"
    away_label = f"Visitante ({visitante})" if visitante else "Visitante"
    home_entries = trends.get("home") or trends.get("team_a") or []
    away_entries = trends.get("away") or trends.get("team_b") or []
    sections = [
        _entries(home_entries, home_label),
        _entries(away_entries, away_label),
    ]
    return "\n".join([s for s in sections if s])


def orquesta(
    fecha,
    export_bucket=None,
    manual_picks: list[dict[str, str]] | None = None,
    solicitar_manual: bool = False,
    solo_manual: bool = False,
):

    today_match = get_todays_matches_normalized(fecha)

    if not today_match:
        print(f"\n📭 Hoy no hay partidos ({fecha})")
        return pd.DataFrame()

    selecciones_manual = list(manual_picks or [])
    if solicitar_manual:
        selecciones_manual.extend(_solicitar_picks_manual(today_match))
    today_match_manual = today_match
    if solo_manual:
        today_match_manual = _filtrar_partidos_manual(today_match, selecciones_manual)
    picks_manual = _construir_picks_manual(selecciones_manual, today_match_manual)

    resultados_hist_global = []
    gpt_evaluaciones = []

    if solo_manual:
        if not picks_manual:
            print("\n⚠️ No hay picks manuales para analizar.")
            return pd.DataFrame()
        resultados = picks_manual
        total_manual = sum(len(df) for df in picks_manual.values() if isinstance(df, pd.DataFrame))
        if total_manual:
            print(f"\n✏️ Picks manuales añadidos: {total_manual}")
    else:
        # Mercados tradicionales
        df_gol_ht = main_o05ht(today_match)
        df_btts = main_btts(today_match)
        df_over15 = main_over15(today_match)
        df_team_over15 = _tabla_team_over15(today_match, prob_min=60.0)
        df_team_over15 = _anotar_estado_team_over15(df_team_over15)

        resultados = {
            "Gol HT": df_gol_ht,
            "BTTS": df_btts,
            "OVER 1.5": df_over15,
            "Team Over 1.5": df_team_over15,
        }

        if picks_manual:
            for mercado, df_manual in picks_manual.items():
                existente = resultados.get(mercado)
                if isinstance(existente, pd.DataFrame) and not existente.empty:
                    resultados[mercado] = pd.concat([existente, df_manual], ignore_index=True, sort=False)
                else:
                    resultados[mercado] = df_manual
            total_manual = sum(len(df) for df in picks_manual.values() if isinstance(df, pd.DataFrame))
            if total_manual:
                print(f"\n✏️ Picks manuales añadidos: {total_manual}")

    if export_bucket is not None:
        for mercado, df in resultados.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                copia = df.copy()
                copia.insert(0, "Fecha", fecha)
                export_bucket.setdefault("resultados", {}).setdefault(mercado, []).append(copia)

    # Reconstruye históricos por mercado para alimentar GPT
    for mercado, df in resultados.items():
        resultados_hist_global.append((mercado, _build_resultados_historicos(df)))

    # Validación GPT por pick
    gpt_map = {}
    gpt_evaluaciones = []
    mercados_con_gpt = set(resultados.keys())

    def _gpt_key(pid: Any) -> str:
        try:
            return str(pid)
        except Exception:
            return ""

    for mercado, df in resultados.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        pick_col = _pick_column(df)
        if pick_col is None:
            continue
        if mercado in mercados_con_gpt:
            n_partidos = df[pick_col].nunique(dropna=True) if pick_col else len(df)
            print(f"\n🧪 Evaluando mercado {mercado} (GPT) {n_partidos} partidos")

        hist_df = next((h for m, h in resultados_hist_global if m == mercado), pd.DataFrame())
        hist_lookup = {}
        if isinstance(hist_df, pd.DataFrame) and not hist_df.empty and "Pick" in hist_df.columns:
            hist_lookup = hist_df.drop_duplicates(subset=["Pick"]).set_index("Pick")[["json_pick", "json_hist"]].to_dict("index")

        for _, row in df.iterrows():
            if bool(row.get("Manual")):
                continue  # los picks manuales no pasan por GPT
            pick_id = row.get(pick_col)
            if pd.isna(pick_id):
                continue
            pick_key = _gpt_key(pick_id)
            hist_paths = hist_lookup.get(pick_id, {})
            local_name = row.get("Local") or row.get("Visita") or ""
            visitante_name = row.get("Visitante") or row.get("Visita") or ""
            match_id_val = row.get("Match_id") or row.get("Partido") or row.get("ID_partido") or pick_id
            trends = _fetch_match_trends(match_id_val)
            trends_text = _format_trends_text(trends, str(local_name), str(visitante_name))
            resp = resultados_gpt(
                mercado,
                local_name,
                visitante_name,
                fecha,
                row.get("Hora") or "",
                hist_paths.get("json_pick"),
                hist_paths.get("json_hist"),
                trends_text,
            )
            if resp:
                gpt_map[(mercado, pick_key)] = resp
                gpt_evaluaciones.append({
                    "Mercado": mercado,
                    "Pick": pick_id,
                    "Partido": f"{row.get('Local') or row.get('Visita') or ''} vs {row.get('Visitante') or row.get('Visita') or ''}",
                    "Hora": row.get("Hora"),
                    "valido": resp.get("valido", False),
                    "explicacion": resp.get("explicacion", ""),
                })

    columnas_por_mercado = {
        "Gol HT": [
            "Hora", "Pais", "Liga", "Local", "Visitante", "ODDS", "Mercado", "Marcador_HT",
            "Potencial", "Prob_modelo", "Tarjetas",
            "Estado", "ROI_estimado", "Ventaja_modelo", "PROB_Historico"
        ],
        "BTTS": [
            "Hora", "Pais", "Liga", "Local", "Visita", "ODDS", "Mercado", "Marcador",
            "Potencial", "Prob_modelo", "Tarjetas", "Estado", "ROI_estimado", "Ventaja_modelo", "PROB_Historico"
        ],
        "OVER 1.5": [
            "Hora", "Pais", "Liga", "Local", "Visitante", "ODDS", "Mercado", "Marcador",
            "Potencial", "Potencial_final", "Prob_modelo", "Tarjetas", "Estado",
            "ROI_estimado", "Ventaja_modelo", "PROB_Historico"
        ],
        "Corners 8.5": [
            "Hora", "Pais", "Liga", "Local", "Visitante", "Mercado", "ODDS",
            "Corners_local", "Corners_visitante", "Corners_total",
            "Potencial_total", "Potencial_o85", "Prob_modelo", "EV_modelo",
            "Estado", "PROB_Historico"
        ],
        # Team Over 1.5 solo muestra campos clave para no saturar la tabla
        "Team Over 1.5": [
            "Hora", "Pais", "Liga", "Local", "Visitante", "Mercado",
            "OTeam", "Probabilidad", "Cuota_Justa", "Marcador",
            "Goles_OTeam", "Estado", "Fecha_ejecucion"
        ],
    }

    if gpt_evaluaciones:
        df_validos = pd.DataFrame([r for r in gpt_evaluaciones if r["valido"]])
        if not df_validos.empty:
            print_markdown_table(
                df_validos[["Mercado", "Pick", "Partido", "Hora", "explicacion"]],
                f"Picks validados GPT {fecha}"
            )

    picks_registro = []

    for mercado, df in resultados.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        pick_col = _pick_column(df)
        if pick_col is None:
            continue
        df = _normalize_estado_column(df)
        if mercado in mercados_con_gpt:
            if gpt_map:
                df["GPT_valido"] = df[pick_col].map(
                    lambda pid, m=mercado: gpt_map.get((m, _gpt_key(pid)), {}).get("valido", True)
                )
                df["GPT_explicacion"] = df[pick_col].map(
                    lambda pid, m=mercado: gpt_map.get((m, _gpt_key(pid)), {}).get("explicacion", "")
                )
            else:
                df["GPT_valido"] = True
                df["GPT_explicacion"] = ""
        else:
            df["GPT_valido"] = True
            df["GPT_explicacion"] = ""
        if "Manual" in df.columns:
            df.loc[df["Manual"] == True, "GPT_valido"] = True
            df.loc[df["Manual"] == True, "GPT_explicacion"] = ""
        df = _ensure_roi(df)
        df = _annotate_unidades(df)
        # Imprime todos los picks evaluados (incluye GPT_valido) para visibilidad completa.
        columnas_config = columnas_por_mercado.get(mercado, df.columns.tolist())
        columnas_presentes = [c for c in columnas_config if c in df.columns]
        tabla_evaluada = df[columnas_presentes].copy() if columnas_presentes else df.copy()
        print_markdown_table(tabla_evaluada, f"Picks evaluados {mercado} {fecha}")
        df_validos = df[df["GPT_valido"]]
        if not df_validos.empty:
            columnas_config = columnas_por_mercado.get(mercado, df_validos.columns.tolist())
            columnas_presentes = [c for c in columnas_config if c in df_validos.columns]
            tabla_validada = df_validos[columnas_presentes].copy()
            print_markdown_table(tabla_validada, f"Picks seleccionados {mercado} {fecha}")
        # Guarda todos los picks analizados (GPT_valido True/False) para poder enviarlos por fecha.
        df_export = df.copy()
        if "Match_id" not in df_export.columns and pick_col in df_export.columns:
            df_export["Match_id"] = df_export[pick_col]
        if "Partido" not in df_export.columns and pick_col in df_export.columns:
            df_export["Partido"] = df_export[pick_col]
        # Normaliza Season_id para no perder IDs al persistir en maestro.
        if "Season_id" not in df_export.columns:
            if "season_id" in df_export.columns:
                df_export["Season_id"] = df_export["season_id"]
            elif "competition_id" in df_export.columns:
                df_export["Season_id"] = df_export["competition_id"]
        if "season_id" not in df_export.columns and "Season_id" in df_export.columns:
            df_export["season_id"] = df_export["Season_id"]
        if "competition_id" not in df_export.columns and "Season_id" in df_export.columns:
            df_export["competition_id"] = df_export["Season_id"]
        df_export["market_group"] = {
            "Gol HT": "GOLHT",
            "BTTS": "BTTS",
            "OVER 1.5": "OVER",
            "Corners 8.5": "CORNERS",
            "Team Over 1.5": "TEAM_OVER",
        }.get(mercado, mercado.upper())
        df_export["Fecha_ejecucion"] = fecha
        picks_registro.append(df_export)

    resumen_detalle = []
    global_odds: list[float] = []
    global_odds_exe: list[float] = []
    for nombre, df in resultados.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            df_util = df
            if "GPT_valido" in df.columns:
                df_util = df[df["GPT_valido"]]
            df_util = _normalize_estado_column(df_util)
            df_util = _sync_estado_exe(df_util)
            df_util = _merge_odds_metricas(df_util, nombre)
            df_util = _annotate_unidades(df_util)
            estados_pre = df_util.get("Estado", pd.Series("", index=df_util.index)).astype(str).str.upper()
            estados_exe = df_util.get("Estado_EXE", estados_pre).astype(str).str.upper()
            odds_pre = pd.to_numeric(df_util.get("ODDS"), errors="coerce")
            if not isinstance(odds_pre, pd.Series):
                odds_pre = pd.Series(np.nan, index=df_util.index, dtype="float")
            odds_exe = pd.to_numeric(df_util.get("ODDS_metricas", odds_pre), errors="coerce")
            metricas = _calc_metricas_resumen(estados_pre, odds_pre)
            metricas_exe = _calc_metricas_resumen(estados_exe, odds_exe)
            valid_mask = estados_pre.isin(["VERDE", "ROJO"])
            odds_clean = pd.to_numeric(odds_pre.where(valid_mask), errors="coerce").dropna()
            if not odds_clean.empty:
                global_odds.extend(odds_clean.tolist())
            valid_mask_exe = estados_exe.isin(["VERDE", "ROJO"])
            odds_clean_exe = pd.to_numeric(odds_exe.where(valid_mask_exe), errors="coerce").dropna()
            if not odds_clean_exe.empty:
                global_odds_exe.extend(odds_clean_exe.tolist())
        else:
            metricas = _calc_metricas_resumen(pd.Series(dtype=str), pd.Series(dtype=float))
            metricas_exe = _calc_metricas_resumen(pd.Series(dtype=str), pd.Series(dtype=float))

        resumen_detalle.append({
            "Mercado": nombre,
            "Aciertos": metricas["Aciertos"],
            "Fallos": metricas["Fallos"],
            "Nulos": metricas["Nulos"],
            "Total": metricas["Total"],
            "PEX": metricas["PEX"] if metricas["PEX"] is not None else 0,
            "ROI": metricas["ROI"],
            "AVG_ODDS": metricas["AVG_ODDS"],
            "Unidades": metricas["Unidades"],
            "Nulos_EXE": metricas_exe["Nulos"],
            "PEX_EXE": metricas_exe["PEX"] if metricas_exe["PEX"] is not None else 0,
            "ROI_EXE": metricas_exe["ROI"],
            "AVG_ODDS_EXE": metricas_exe["AVG_ODDS"],
            "Unidades_EXE": metricas_exe["Unidades"],
        })

    resumen = pd.DataFrame()
    if resumen_detalle:
        resumen = pd.DataFrame(resumen_detalle)
        total_aciertos = resumen["Aciertos"].sum()
        total_fallos = resumen["Fallos"].sum()
        total_nulos = resumen["Nulos"].sum() if "Nulos" in resumen.columns else 0
        total_juegos = total_aciertos + total_fallos
        total_odds = pd.to_numeric(pd.Series(global_odds), errors="coerce")
        avg_total_odds = round(total_odds.mean(), 3) if not total_odds.empty else None
        total_odds_exe = pd.to_numeric(pd.Series(global_odds_exe), errors="coerce")
        avg_total_odds_exe = round(total_odds_exe.mean(), 3) if not total_odds_exe.empty else None
        roi_total = unidades_total = None
        if avg_total_odds is not None and total_juegos:
            unidades_total = round((total_aciertos * (avg_total_odds - 1) - total_fallos), 3)
            roi_total = round((unidades_total / total_juegos) * 100, 3)
        unidades_exe = None
        roi_exe = None
        if avg_total_odds_exe is not None and total_juegos:
            unidades_exe = round((total_aciertos * (avg_total_odds_exe - 1) - total_fallos), 3)
            roi_exe = round((unidades_exe / total_juegos) * 100, 3)
        total_row = {
            "Mercado": "TOTAL",
            "Aciertos": total_aciertos,
            "Fallos": total_fallos,
            "Nulos": total_nulos,
            "Total": total_juegos,
            "PEX": int(round((total_aciertos / total_juegos) * 100)) if total_juegos else 0,
            "ROI": roi_total,
            "AVG_ODDS": avg_total_odds,
            "Unidades": unidades_total,
            "PEX_EXE": int(round((total_aciertos / total_juegos) * 100)) if total_juegos else 0,
            "ROI_EXE": roi_exe,
            "AVG_ODDS_EXE": avg_total_odds_exe if total_juegos else None,
            "Unidades_EXE": unidades_exe,
        }
        resumen = pd.concat([resumen, pd.DataFrame([total_row])], ignore_index=True, sort=False)
        if "PEX" in resumen.columns:
            resumen["PEX"] = resumen["PEX"].astype(int)
        _print_resumen_dual(resumen, f"Resumen PEX {fecha}")

    if export_bucket is not None and not resumen.empty:
        resumen_export = resumen.copy()
        resumen_export.insert(0, "Fecha", fecha)
        export_bucket.setdefault("resumen", []).append(resumen_export)

    if picks_registro:
        picks_df = pd.concat(picks_registro, ignore_index=True)
        _upsert_maestro(picks_df)
        print(f"✅ Picks guardados en maestro ({len(picks_df)} filas)")

    tmp_dir = Path("tmp_resultados")
    if tmp_dir.exists() and tmp_dir.is_dir():
        for json_tmp in tmp_dir.glob("*.json"):
            try:
                json_tmp.unlink()
            except OSError:
                pass

    if not resumen.empty:
        return resumen
    return pd.DataFrame()


def mostrar_resultados_guardados(fecha: str, return_detalle: bool = False):
    df = _maestro_por_fecha(fecha)
    if df.empty:
        print(f"\n📄 No hay picks en maestro para {fecha}.")
        return
    # Asegura IDs fundamentales
    if "Season_id" not in df.columns:
        if "season_id" in df.columns:
            df["Season_id"] = df["season_id"]
        elif "competition_id" in df.columns:
            df["Season_id"] = df["competition_id"]
    if "season_id" not in df.columns and "Season_id" in df.columns:
        df["season_id"] = df["Season_id"]
    if "Match_id" not in df.columns and "Partido" in df.columns:
        df["Match_id"] = df["Partido"]
    if "Partido" not in df.columns and "Match_id" in df.columns:
        df["Partido"] = df["Match_id"]
    if "Odds_O85" in df.columns:
        if "ODDS" in df.columns:
            df["ODDS"] = df["ODDS"].fillna(df["Odds_O85"])
            df = df.drop(columns=["Odds_O85"])
        else:
            df = df.rename(columns={"Odds_O85": "ODDS"})
    df = _normalize_estado_column(df)
    df = _actualizar_pendientes(df)
    df = _sync_estado_exe(df)
    df = _annotate_unidades(_ensure_roi(df))
    # Refleja cambios en el maestro
    _upsert_maestro(df)
    if df.empty:
        print(f"\n📄 El archivo de {fecha} no contiene picks almacenados.")
        return

    columnas_por_grupo = {
        "GOLHT": [
            "Hora", "Pais", "Liga", "Local", "Visitante", "ODDS", "Mercado", "Marcador_HT",
            "Potencial", "Prob_modelo", "Tarjetas",
            "Estado"
        ],
        "BTTS": [
            "Hora", "Pais", "Liga", "Local", "Visita", "ODDS", "Mercado", "Marcador",
            "Potencial", "Prob_modelo", "Tarjetas", "Estado"
        ],
        "OVER": [
            "Hora", "Pais", "Liga", "Local", "Visitante", "ODDS", "Mercado", "Marcador",
            "Potencial", "Prob_modelo", "Tarjetas", "Estado"
        ],
        "CORNERS": [
            "Hora", "Pais", "Liga", "Local", "Visitante", "Mercado", "ODDS",
            "Corners_local", "Corners_visitante", "Corners_total",
            "Potencial_total", "Potencial_o85", "Estado"
        ],
        # TEAM_OVER usa la versión reducida de columnas para facilitar la lectura
        "TEAM_OVER": [
            "Hora", "Pais", "Liga", "Local", "Visitante", "Mercado",
            "OTeam", "Probabilidad", "Cuota_Justa", "Marcador",
            "Goles_OTeam", "Estado", "Fecha_ejecucion"
        ],
    }
    nombres_display = {
        "GOLHT": "Gol HT",
        "BTTS": "BTTS",
        "OVER": "OVER 1.5",
        # "CORNERS": "Corners 8.5",  # deshabilitado temporalmente
        "TEAM_OVER": "Team Over 1.5",
    }

    resumen_resultados = []
    resumen_odds_global: list[float] = []
    resumen_odds_exe_global: list[float] = []
    resumen_unidades_global: list[float] = []
    detalles: list[pd.DataFrame] = []

    for group, df_group in df.groupby("market_group"):
        group_key = group
        # Normaliza alias con espacios para Team Over 1.5
        if str(group).upper().replace(" ", "_") in ("TEAM_OVER_1.5", "TEAM_OVER1.5"):
            group_key = "TEAM_OVER"
        # Saltar córners 8.5 mientras esté deshabilitado
        if group_key == "CORNERS":
            continue
        nombre = nombres_display.get(group_key, nombres_display.get(group, group))
        df_local = df_group.copy()
        if "Match_id" not in df_local.columns and "Partido" in df_local.columns:
            df_local["Match_id"] = df_local["Partido"]
        if "ID_partido" not in df_local.columns and "Match_id" in df_local.columns:
            df_local["ID_partido"] = df_local["Match_id"]
        if "Match_id" not in df_local.columns and "ID_partido" in df_local.columns:
            df_local["Match_id"] = df_local["ID_partido"]
        if "Season_id" not in df_local.columns:
            if "season_id" in df_local.columns:
                df_local["Season_id"] = df_local["season_id"]
            elif "competition_id" in df_local.columns:
                df_local["Season_id"] = df_local["competition_id"]
        if "season_id" not in df_local.columns and "Season_id" in df_local.columns:
            df_local["season_id"] = df_local["Season_id"]
        nulo_mask_orig = df_local["Estado"].astype(str).str.upper() == "NULO"
        df_local = anotar_estado(df_local, group_key)
        if nulo_mask_orig.any():
            df_local.loc[nulo_mask_orig, "Estado"] = "NULO"
        df_local = _sync_estado_exe(df_local)
        df_local = _merge_odds_metricas(df_local, nombre)
        df_local = _annotate_unidades(df_local)
        columnas = columnas_por_grupo.get(group_key, columnas_por_grupo.get(group, df_local.columns.tolist()))
        columnas_presentes = [c for c in columnas if c in df_local.columns]
        print_markdown_table(df_local[columnas_presentes], f"Resultados {nombre} {fecha}")
        if return_detalle:
            df_export = df_local.copy()
            df_export["Fecha_consulta"] = fecha
            detalles.append(df_export)

        if {"Estado"} <= set(df_local.columns):
            estados = df_local["Estado"].astype(str).str.upper()
            estados_exe = df_local.get("Estado_EXE", estados).astype(str).str.upper()
            odds_pre = pd.to_numeric(df_local.get("ODDS"), errors="coerce")
            if not isinstance(odds_pre, pd.Series):
                odds_pre = pd.Series(np.nan, index=df_local.index, dtype="float")
            odds_exe = pd.to_numeric(df_local.get("ODDS_metricas", odds_pre), errors="coerce")
            metricas = _calc_metricas_resumen(estados, odds_pre)
            metricas_exe = _calc_metricas_resumen(estados_exe, odds_exe)
            valid_mask = estados.isin(["VERDE", "ROJO"])
            odds_clean = pd.to_numeric(odds_pre.where(valid_mask), errors="coerce").dropna()
            if not odds_clean.empty:
                resumen_odds_global.extend(odds_clean.tolist())
            valid_mask_exe = estados_exe.isin(["VERDE", "ROJO"])
            odds_clean_exe = pd.to_numeric(odds_exe.where(valid_mask_exe), errors="coerce").dropna()
            if not odds_clean_exe.empty:
                resumen_odds_exe_global.extend(odds_clean_exe.tolist())
            resumen_resultados.append({
                "Mercado": nombre,
                "Aciertos": metricas["Aciertos"],
                "Fallos": metricas["Fallos"],
                "Nulos": metricas["Nulos"],
                "Total": metricas["Total"],
                "PEX": metricas["PEX"] if metricas["PEX"] is not None else 0,
                "ROI": metricas["ROI"],
                "AVG_ODDS": metricas["AVG_ODDS"],
                "Unidades": metricas["Unidades"],
                "Nulos_EXE": metricas_exe["Nulos"],
                "PEX_EXE": metricas_exe["PEX"] if metricas_exe["PEX"] is not None else 0,
                "ROI_EXE": metricas_exe["ROI"],
                "AVG_ODDS_EXE": metricas_exe["AVG_ODDS"],
                "Unidades_EXE": metricas_exe["Unidades"],
            })

    resumen_df = pd.DataFrame()
    if resumen_resultados:
        resumen_df = pd.DataFrame(resumen_resultados)
        total_aciertos = resumen_df["Aciertos"].sum()
        total_fallos = resumen_df["Fallos"].sum()
        total_nulos = resumen_df["Nulos"].sum() if "Nulos" in resumen_df.columns else 0
        total_juegos = total_aciertos + total_fallos
        total_odds = pd.to_numeric(pd.Series(resumen_odds_global), errors="coerce")
        avg_total_odds = round(total_odds.mean(), 3) if not total_odds.empty else None
        total_odds_exe = pd.to_numeric(pd.Series(resumen_odds_exe_global), errors="coerce")
        avg_total_odds_exe = round(total_odds_exe.mean(), 3) if not total_odds_exe.empty else None
        roi_total = unidades_total = None
        if avg_total_odds is not None and total_juegos:
            unidades_total = round((total_aciertos * (avg_total_odds - 1)) - total_fallos, 3)
            roi_total = round((unidades_total / total_juegos) * 100, 3)
        unidades_exe = None
        roi_exe = None
        if avg_total_odds_exe is not None and total_juegos:
            unidades_exe = round((total_aciertos * (avg_total_odds_exe - 1)) - total_fallos, 3)
            roi_exe = round((unidades_exe / total_juegos) * 100, 3) if total_juegos else None
        resumen_df = pd.concat([resumen_df, pd.DataFrame([{
            "Mercado": "TOTAL",
            "Aciertos": total_aciertos,
            "Fallos": total_fallos,
            "Nulos": total_nulos,
            "Total": total_juegos,
            "PEX": int(round((total_aciertos / total_juegos) * 100)) if total_juegos else 0,
            "ROI": roi_total,
            "AVG_ODDS": avg_total_odds,
            "Unidades": unidades_total,
            "PEX_EXE": int(round((total_aciertos / total_juegos) * 100)) if total_juegos else 0,
            "ROI_EXE": roi_exe,
            "AVG_ODDS_EXE": avg_total_odds_exe if total_juegos else None,
            "Unidades_EXE": unidades_exe,
        }])], ignore_index=True)
        _print_resumen_dual(resumen_df, f"Resumen resultados {fecha}")

    detalle_df = pd.concat(detalles, ignore_index=True) if detalles else pd.DataFrame()
    if return_detalle:
        return resumen_df, detalle_df
    return resumen_df

def dias(
    offset: int = 0,
    solicitar_manual: bool = False,
    manual_picks: list[dict[str, str]] | None = None,
    solo_manual: bool = False,
):
    """
    Ejecuta la función 'orquesta' para un único día desplazado respecto a HOY.

    offset = 0  → hoy
    offset > 0  → días futuros (1=mañana, 2=pasado mañana, etc.)
    offset < 0  → días pasados (-1=ayer, -2=anteayer, etc.)
    solo_manual = True → analiza únicamente los picks manuales seleccionados.
    """
    base = date.today()
    day = (base + timedelta(days=offset)).isoformat()
    print(f"\n▶️ Ejecutando orquesta para {day}")
    orquesta(day, manual_picks=manual_picks, solicitar_manual=solicitar_manual, solo_manual=solo_manual)


def _combinar_resumenes(resumenes: list[pd.DataFrame]) -> pd.DataFrame:
    if not resumenes:
        return pd.DataFrame()
    partes = []
    for df in resumenes:
        partes.append(df[df["Mercado"] != "TOTAL"])
    if not partes:
        return pd.DataFrame()
    merged = pd.concat(partes, ignore_index=True)
    for col in ["Aciertos", "Fallos", "Nulos", "Total", "Unidades", "AVG_ODDS", "Unidades_EXE", "AVG_ODDS_EXE"]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")
    agg_dict = {"Aciertos": "sum", "Fallos": "sum", "Nulos": "sum", "Total": "sum"}
    if "Unidades" in merged.columns:
        agg_dict["Unidades"] = "sum"
    if "AVG_ODDS" in merged.columns:
        agg_dict["AVG_ODDS"] = "mean"
    if "Unidades_EXE" in merged.columns:
        agg_dict["Unidades_EXE"] = "sum"
    if "AVG_ODDS_EXE" in merged.columns:
        agg_dict["AVG_ODDS_EXE"] = "mean"
    agrupado = merged.groupby("Mercado").agg(agg_dict).reset_index()
    agrupado = agrupado[agrupado["Total"] > 0]
    agrupado["PEX"] = agrupado.apply(lambda row: _calc_pex_value(row["Aciertos"], row["Total"]), axis=1)
    if {"Aciertos", "Fallos", "Total", "AVG_ODDS"} <= set(agrupado.columns):
        agrupado["ROI"] = agrupado.apply(
            lambda row: round(((row["Aciertos"] * (row["AVG_ODDS"] - 1)) - row["Fallos"]) / row["Total"] * 100, 3)
            if row["Total"] and not pd.isna(row["AVG_ODDS"]) else None,
            axis=1
        )
        agrupado["Unidades"] = agrupado.apply(
            lambda row: round((row["Aciertos"] * (row["AVG_ODDS"] - 1)) - row["Fallos"], 3)
            if not pd.isna(row["AVG_ODDS"]) else None,
            axis=1
        )
    if {"Aciertos", "Fallos", "Total"} <= set(agrupado.columns):
        agrupado["PEX_EXE"] = agrupado.apply(
            lambda row: _calc_pex_value(row["Aciertos"], row["Total"]), axis=1
        )
    if {"Aciertos", "Fallos", "Total", "AVG_ODDS_EXE"} <= set(agrupado.columns):
        agrupado["ROI_EXE"] = agrupado.apply(
            lambda row: round(((row["Aciertos"] * (row.get("AVG_ODDS_EXE", EXEC_AVG_ODDS) - 1)) - row["Fallos"]) / row["Total"] * 100, 3)
            if row["Total"] and not pd.isna(row.get("AVG_ODDS_EXE", EXEC_AVG_ODDS)) else None,
            axis=1
        )
        agrupado["Unidades_EXE"] = agrupado.apply(
            lambda row: round((row["Aciertos"] * (row.get("AVG_ODDS_EXE", EXEC_AVG_ODDS) - 1)) - row["Fallos"], 3)
            if not pd.isna(row.get("AVG_ODDS_EXE", EXEC_AVG_ODDS)) else None,
            axis=1
        )
    total_aciertos = agrupado["Aciertos"].sum()
    total_fallos = agrupado["Fallos"].sum()
    total_nulos = agrupado["Nulos"].sum() if "Nulos" in agrupado.columns else 0
    total_juegos = total_aciertos + total_fallos
    total_odds = agrupado["AVG_ODDS"] if "AVG_ODDS" in agrupado.columns else pd.Series(dtype=float)
    avg_total_odds = round(total_odds.mean(), 3) if not total_odds.empty else None
    total_odds_exe = agrupado["AVG_ODDS_EXE"] if "AVG_ODDS_EXE" in agrupado.columns else pd.Series(dtype=float)
    avg_total_odds_exe = round(total_odds_exe.mean(), 3) if not total_odds_exe.empty else None
    total_unidades = agrupado["Unidades"] if "Unidades" in agrupado.columns else pd.Series(dtype=float)
    total_unidades_sum = round(total_unidades.sum(), 3) if not total_unidades.empty else None
    total_unidades_exe = agrupado["Unidades_EXE"] if "Unidades_EXE" in agrupado.columns else pd.Series(dtype=float)
    total_unidades_exe_sum = round(total_unidades_exe.sum(), 3) if not total_unidades_exe.empty else None
    total_row = {
        "Mercado": "TOTAL",
        "Aciertos": total_aciertos,
        "Fallos": total_fallos,
        "Nulos": total_nulos,
        "Total": total_juegos,
        "PEX": str(int(round((total_aciertos / total_juegos) * 100))) if total_juegos else "0",
        "ROI": round(((total_aciertos * (avg_total_odds - 1)) - total_fallos) / total_juegos * 100, 3)
               if total_juegos and avg_total_odds is not None else None,
        "AVG_ODDS": avg_total_odds,
        "Unidades": total_unidades_sum,
        "PEX_EXE": str(int(round((total_aciertos / total_juegos) * 100))) if total_juegos else "0",
        "ROI_EXE": round(((total_aciertos * (avg_total_odds_exe - 1)) - total_fallos) / total_juegos * 100, 3)
                   if total_juegos and avg_total_odds_exe is not None else None,
        "AVG_ODDS_EXE": avg_total_odds_exe if total_juegos else None,
        "Unidades_EXE": total_unidades_exe_sum if total_unidades_exe_sum is not None else None,
    }
    resumen = pd.concat([agrupado, pd.DataFrame([total_row])], ignore_index=True)
    col_order = ["Mercado", "Aciertos", "Fallos", "Nulos", "Total", "PEX", "ROI", "AVG_ODDS", "Unidades",
                 "PEX_EXE", "ROI_EXE", "AVG_ODDS_EXE", "Unidades_EXE"]
    resumen = resumen[[c for c in col_order if c in resumen.columns]]
    return resumen


def _sanitize_sheet_name(name: str) -> str:
    clean = "".join(ch if ch.isalnum() or ch in (" ", "-", "_") else "_" for ch in str(name))
    return clean.strip()[:31] or "Sheet"


def _es_periodo_mensual(descripcion: str) -> bool:
    return "mes" in (descripcion or "").lower()


def _tag_mes_anio(fechas: list[str]) -> str | None:
    for fecha in fechas:
        try:
            dt = datetime.fromisoformat(fecha)
            return f"{dt.month:02d}{dt.year}"
        except Exception:
            continue
    return None


def _formatear_maestro(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reorganiza y renombra columnas para el archivo maestro mensual.
    """
    objetivo = [
        "Hora",
        "Liga",
        "Local",
        "Visitante",
        "ODDS",
        "Mercado",
        "Marcador_HT",
        "Marcador_FT",
        "Corners_total",
        "Estado",
        "Prob_Modelo",
        "Prob_Historico",
    ]
    d = df.copy()
    # Completar Visitante usando Visita cuando falte.
    if "Visitante" not in d.columns:
        d["Visitante"] = d.get("Visita", "")
    elif "Visita" in d.columns:
        mask = d["Visitante"].isna() | (d["Visitante"].astype(str).str.strip() == "")
        d.loc[mask, "Visitante"] = d.loc[mask, "Visita"]
    # Marcadores: conservar si existen y completar con alternativas.
    d["Marcador_HT"] = d.get("Marcador_HT", d.get("Marcador_ht", ""))
    if "Marcador_ht" in d.columns:
        d["Marcador_HT"] = d["Marcador_HT"].fillna(d["Marcador_ht"])
    d["Marcador_FT"] = d.get("Marcador_FT", "")
    if "Marcador" in d.columns:
        d["Marcador_FT"] = d["Marcador_FT"].fillna(d["Marcador"])
        vacios_ft = d["Marcador_FT"].astype(str).str.strip() == ""
        d.loc[vacios_ft, "Marcador_FT"] = d.loc[vacios_ft, "Marcador"]
    # Corners_total: intenta completar usando corners totales calculados.
    if "Corners_total" not in d.columns:
        total = pd.to_numeric(d.get("Corners_local"), errors="coerce").fillna(0) + pd.to_numeric(
            d.get("Corners_visitante"), errors="coerce"
        ).fillna(0)
        d["Corners_total"] = total.replace(0, pd.NA)
    if "Corners_FT" in d.columns:
        d["Corners_total"] = d["Corners_total"].fillna(d["Corners_FT"])
    # Probabilidades: renombrar a formato solicitado.
    if "Prob_modelo" in d.columns:
        d["Prob_Modelo"] = d["Prob_modelo"]
    if "PROB_Historico" in d.columns:
        d["Prob_Historico"] = d["PROB_Historico"]
    # Asegurar columnas faltantes vacías.
    for col in objetivo:
        if col not in d.columns:
            d[col] = ""
    return d[objetivo]


def _exportar_excel(resultados: dict, resumen: pd.DataFrame):
    activos = {
        mercado: df
        for mercado, df in resultados.items()
        if isinstance(df, pd.DataFrame) and not df.empty
    }
    include_resumen = isinstance(resumen, pd.DataFrame) and not resumen.empty
    if not activos and not include_resumen:
        return

    carpeta = Path("reportes")
    carpeta.mkdir(exist_ok=True)
    archivo = carpeta / "orquesta.xlsx"

    existentes = {}
    if archivo.exists():
        try:
            existentes = pd.read_excel(archivo, sheet_name=None)
        except Exception:
            existentes = {}

    hojas = dict(existentes)
    for mercado, df in activos.items():
        sheet = _sanitize_sheet_name(mercado)
        previo = hojas.get(sheet)
        combinado = pd.concat([previo, df], ignore_index=True) if isinstance(previo, pd.DataFrame) else df
        hojas[sheet] = combinado

    if include_resumen:
        previo = hojas.get("Resumen")
        combinado = pd.concat([previo, resumen], ignore_index=True) if isinstance(previo, pd.DataFrame) else resumen
        hojas["Resumen"] = combinado

    with pd.ExcelWriter(archivo) as writer:
        for sheet, df in hojas.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                nombre = _sanitize_sheet_name(sheet)
                df.to_excel(writer, sheet_name=nombre, index=False)
    print(f"\n📁 Reporte acumulado: {archivo}")

def _ejecutar_periodo(fechas: list[str], descripcion: str, exportar: bool = False):
    resumenes = []
    export_bucket = {"resultados": {}, "resumen": []} if exportar else None
    for fecha in fechas:
        print(f"\n▶️ Ejecutando orquesta para {fecha}")
        resumen = orquesta(fecha, export_bucket=export_bucket)
        if isinstance(resumen, pd.DataFrame) and not resumen.empty:
            resumenes.append(resumen)
    resumen_global = _combinar_resumenes(resumenes)
    if not resumen_global.empty:
        _print_resumen_dual(resumen_global, f"Resumen PEX {descripcion}")
    if exportar and export_bucket:
        resultados_excel = {
            mercado: pd.concat(partes, ignore_index=True)
            for mercado, partes in export_bucket.get("resultados", {}).items()
            if partes
        }
        resumen_excel = (
            pd.concat(export_bucket.get("resumen", []), ignore_index=True)
            if export_bucket.get("resumen")
            else pd.DataFrame()
        )
        _exportar_excel(resultados_excel, resumen_excel)


def _fechas_finde() -> list[str]:
    hoy = date.today()
    # buscamos el viernes pasado (puede ser hoy si ya es viernes)
    offset_viernes = (hoy.weekday() - 4) % 7
    inicio = hoy - timedelta(days=offset_viernes or 7)
    return [(inicio + timedelta(days=i)).isoformat() for i in range(3)]


def _fechas_semana_pasada() -> list[str]:
    hoy = date.today()
    inicio_semana = hoy - timedelta(days=hoy.weekday() + 7)
    return [(inicio_semana + timedelta(days=i)).isoformat() for i in range(7)]


def _fechas_semana_actual() -> list[str]:
    hoy = date.today()
    inicio_semana = hoy - timedelta(days=hoy.weekday())
    dias_transcurridos = (hoy - inicio_semana).days + 1
    return [(inicio_semana + timedelta(days=i)).isoformat() for i in range(dias_transcurridos)]


def _fechas_mes_actual() -> list[str]:
    hoy = date.today()
    inicio = hoy.replace(day=1)
    dias_transcurridos = (hoy - inicio).days + 1
    return [(inicio + timedelta(days=i)).isoformat() for i in range(dias_transcurridos)]


def _fechas_mes_anterior() -> list[str]:
    hoy = date.today()
    inicio_mes_actual = hoy.replace(day=1)
    ultimo_dia_mes_anterior = inicio_mes_actual - timedelta(days=1)
    inicio_mes_anterior = ultimo_dia_mes_anterior.replace(day=1)
    dias_mes_anterior = ultimo_dia_mes_anterior.day
    return [
        (inicio_mes_anterior + timedelta(days=i)).isoformat()
        for i in range(dias_mes_anterior)
    ]

def _enviar_telegram_desde_maestro(
    fecha: str,
    destinos: dict[str, dict[str, Any]] | None = None,
    usar_pendientes: bool = False,
):
    """
    Envía a Telegram los picks guardados en maestro_picks agrupados por mercado.
    """
    destinos = destinos or Prueba

    def _pick(row: pd.Series, *keys: str) -> str:
        """
        Devuelve el primer valor no vacío/no-NaN encontrado en las keys indicadas.
        Se usa para limpiar nombres de equipos (evita mostrar 'nan' en Telegram).
        """
        for key in keys:
            val = row.get(key)
            if pd.notna(val):
                val_str = str(val).strip()
                if val_str:
                    return val_str
        return ""

    df = _maestro_por_fecha(fecha)
    if df.empty:
        print(f"\n⚠️ No hay picks en maestro para {fecha}, no se envía nada.")
        return
    if usar_pendientes:
        if TELEGRAM_STATUS_COL not in df.columns:
            print(f"\n⚠️ No hay pendientes registrados en maestro para {fecha}.")
            return
        pendientes_mask = df[TELEGRAM_STATUS_COL].astype(str).str.upper() == "PENDIENTE"
        if not pendientes_mask.any():
            print(f"\n⚠️ No hay pendientes registrados en maestro para {fecha}.")
            return
        df = df.loc[pendientes_mask].copy()
    enviados_idx: list[Any] = []
    pendientes_idx: list[Any] = []
    enviados_msg: dict[Any, Any] = {}
    client = get_default_client()
    for idx, row in df.iterrows():
        topic = _mapear_topic_telegram(str(row.get("Mercado", "")))
        datos = destinos.get(topic) if topic else None
        if not datos:
            continue
        fecha_msg = row.get("Fecha_ejecucion") or row.get("Fecha") or fecha
        if pd.isna(fecha_msg):
            fecha_msg = fecha
        pais = row.get("Pais") or row.get("Country") or ""
        liga = row.get("Liga") or ""
        oteam = row.get("OTeam")
        oteam_str = "" if pd.isna(oteam) else str(oteam)
        oteam_label = ""
        if oteam_str:
            oteam_lower = oteam_str.lower()
            if "visit" in oteam_lower:
                oteam_label = "Visitante"
            elif "loc" in oteam_lower:
                oteam_label = "Local"
            else:
                oteam_label = oteam_str
        local = _pick(row, "Local", "Home", "home")
        visitante = _pick(row, "Visitante", "Visita", "Away", "away")
        mercado = row.get("Mercado") or ""
        odds = row.get("ODDS")
        if pd.isna(odds):
            odds = ""
        hora_mx = row.get("Hora") or ""

        lineas = [f"📌 Partido: {local} vs {visitante}"]
        if not pd.isna(liga) and not pd.isna(pais) and liga and pais:
            lineas.append(f"🏆 Liga: {liga} | {pais}")
        if not pd.isna(hora_mx) and hora_mx:
            lineas.append(f"⏰ Hora: {hora_mx} 🇲🇽")

        mercado_lower = str(mercado).lower()
        es_team_over = bool(oteam_label and ("team over" in mercado_lower or "over 1.5" in mercado_lower))
        es_over_general = ("over 1.5" in mercado_lower) and not es_team_over
        mercado_line = f"📊 Mercado: {mercado}"
        if es_team_over:
            mercado_line = f"📊 Mercado: {mercado} ({oteam_label})"
        elif es_over_general:
            mercado_line = "📊 Mercado: Over 1,5 FT"
        lineas.append(mercado_line)

        if mercado_lower.startswith("btts"):
            lineas.append(f"💵 Cuota: {odds}")
        elif es_team_over:
            lineas.append("💵 Cuota LIVE: +1.70")  # solo cuota live para over 1.5 por equipo
        else:
            lineas.append(f"💵 Cuota Pre-Partido: {odds}")
            lineas.append("💵 Cuota LIVE: +1.70")
        lineas.append("🎯 Stake: 1% BANK")

        texto = "\n".join(lineas)
        try:
            resp = client.send_message(
                chat_id=datos["chat_id"],
                text=texto,
                thread_id=datos["thread"],
                disable_web_page_preview=True,
            )
            enviados_idx.append(idx)
            message_id = (resp or {}).get("result", {}).get("message_id")
            if message_id is not None:
                enviados_msg[idx] = message_id
        except requests.exceptions.HTTPError as exc:
            status = getattr(exc.response, "status_code", None)
            if status == 429:
                # Respeta retry_after si viene en el payload; sino espera 3s.
                retry_after = 3
                try:
                    payload = exc.response.json()
                    retry_after = int(payload.get("parameters", {}).get("retry_after", retry_after))
                except Exception:
                    pass
                print(f"⏳ 429 Too Many Requests. Esperando {retry_after}s antes de reintentar '{mercado}' ({local} vs {visitante})")
                time.sleep(retry_after + 1)
                try:
                    resp = client.send_message(
                        chat_id=datos["chat_id"],
                        text=texto,
                        thread_id=datos["thread"],
                        disable_web_page_preview=True,
                    )
                    enviados_idx.append(idx)
                    message_id = (resp or {}).get("result", {}).get("message_id")
                    if message_id is not None:
                        enviados_msg[idx] = message_id
                except Exception as exc2:
                    print(f"❌ Error (reintento) pick '{mercado}' ({local} vs {visitante}): {exc2}")
                    pendientes_idx.append(idx)
            else:
                print(f"❌ Error enviando pick '{mercado}' ({local} vs {visitante}): {exc}")
                pendientes_idx.append(idx)
        except Exception as exc:
            print(f"❌ Error enviando pick '{mercado}' ({local} vs {visitante}): {exc}")
            pendientes_idx.append(idx)
        # Pausa base para no saturar el endpoint de Telegram
        time.sleep(1.2)

    if enviados_idx:
        print(f"✅ Enviados {len(enviados_idx)} picks a Telegram para {fecha}")
    else:
        print(f"\n⚠️ No hay picks mapeados a hilos de Telegram para {fecha}.")
    if pendientes_idx:
        print(f"⚠️ Quedaron {len(pendientes_idx)} picks pendientes en maestro.")

    actualizados = list(dict.fromkeys(enviados_idx + pendientes_idx))
    if actualizados:
        df.loc[enviados_idx, TELEGRAM_STATUS_COL] = "ENVIADO"
        df.loc[pendientes_idx, TELEGRAM_STATUS_COL] = "PENDIENTE"
        if enviados_msg:
            msg_series = pd.Series(enviados_msg)
            df.loc[msg_series.index, TELEGRAM_MESSAGE_ID_COL] = msg_series.values
        _upsert_maestro(df.loc[actualizados].copy())

def _flujo_picks(force_manual: bool = False, solo_manual: bool = False):
    entrada = input("Inserta el número de días (ej: 3 o -1, o 'F'/'M'/'S'): ").strip()
    modo = entrada.lower()
    if modo == "f":
        _ejecutar_periodo(_fechas_finde(), "FIN DE SEMANA")
    elif modo == "m":
        seleccion_mes = input("¿Ejecutar mes en curso (C) o mes anterior (A)? ").strip().lower()
        if seleccion_mes == "a":
            _ejecutar_periodo(_fechas_mes_anterior(), "MES ANTERIOR", exportar=True)
        else:
            _ejecutar_periodo(_fechas_mes_actual(), "MES COMPLETO", exportar=True)
    elif modo == "s":
        _ejecutar_periodo(_fechas_semana_pasada(), "SEMANA ANTERIOR")
    else:
        try:
            offset = int(entrada)
        except ValueError:
            print("Entrada no válida. Usa un número o F/M/S.")
            return
        solicitar_manual = True if force_manual else False
        dias(offset, solicitar_manual=solicitar_manual, solo_manual=solo_manual)


def _mostrar_resultados_periodo(fechas: list[str], descripcion: str):
    resumenes = []
    detalles_periodo = []
    generar_maestro = _es_periodo_mensual(descripcion)
    for fecha in fechas:
        print(f"\n📅 Mostrando resultados para {fecha} ({descripcion})")
        resultado = mostrar_resultados_guardados(fecha, return_detalle=generar_maestro)
        if resultado is None:
            continue
        if generar_maestro and isinstance(resultado, tuple):
            resumen, detalle = resultado
        else:
            resumen = resultado
            detalle = None
        if isinstance(resumen, pd.DataFrame) and not resumen.empty:
            resumenes.append(resumen)
        if generar_maestro and isinstance(detalle, pd.DataFrame) and not detalle.empty:
            detalles_periodo.append(detalle)
    if resumenes:
        resumen_periodo = _combinar_resumenes(resumenes)
        if not resumen_periodo.empty:
            _print_resumen_dual(resumen_periodo, f"Resumen acumulado {descripcion}")
    if generar_maestro and detalles_periodo:
        maestro_df = pd.concat(detalles_periodo, ignore_index=True)
        maestro_df = _formatear_maestro(maestro_df)
        tag = _tag_mes_anio(fechas)
        if tag:
            carpeta = Path("reportes")
            carpeta.mkdir(exist_ok=True)
            archivo = carpeta / f"Resumen{tag}.xlsx"
            maestro_df.to_excel(archivo, index=False)
            print(f"\n📁 Resumen mensual generado: {archivo}")


def _flujo_resultados():
    entrada = input("Inserta el número de días (ej: 3 o -1, o 'F'/'M'/'S'): ").strip()
    modo = entrada.lower()
    if modo == "f":
        _mostrar_resultados_periodo(_fechas_finde(), "Fin de semana")
    elif modo == "m":
        seleccion_mes = input("¿Consultar mes en curso (C) o mes anterior (A)? ").strip().lower()
        if seleccion_mes == "a":
            _mostrar_resultados_periodo(_fechas_mes_anterior(), "Mes anterior")
        else:
            _mostrar_resultados_periodo(_fechas_mes_actual(), "Mes en curso")
    elif modo == "s":
        seleccion_semana = input("¿Consultar semana en curso (C) o semana anterior (A)? ").strip().lower()
        if seleccion_semana == "a":
            _mostrar_resultados_periodo(_fechas_semana_pasada(), "Semana anterior")
        else:
            _mostrar_resultados_periodo(_fechas_semana_actual(), "Semana en curso")
    else:
        try:
            offset = int(entrada)
        except ValueError:
            print("Entrada no válida. Usa un número o F/M/S.")
            return
        fecha = (date.today() + timedelta(days=offset)).isoformat()
        ajustar_nulos = input("¿Deseas ajustar NULOs por ID antes de ver resultados? (s/N): ").strip().lower() == "s"
        if ajustar_nulos:
            _ajustar_nulos_por_id(fecha)
        mostrar_resultados_guardados(fecha)

def _flujo_telegram():
    destino_sel = input("Selecciona destino Telegram (P: Prueba, G: GlobalPicks): ").strip().lower()
    destinos = Prueba
    if destino_sel == "g":
        try:
            from app.common.telegram_client import GlobalPicks
            destinos = GlobalPicks  # type: ignore
        except Exception:
            print("⚠️ No se pudo cargar GlobalPicks, se usará Prueba.")
    entrada = input("Fecha a enviar (YYYY-MM-DD) o número de días (0=hoy, -1=ayer): ").strip()
    if not entrada:
        fecha = date.today().isoformat()
    else:
        try:
            offset = int(entrada)
            fecha = (date.today() + timedelta(days=offset)).isoformat()
        except ValueError:
            fecha = entrada
    usar_pend = False
    if _hay_pendientes_telegram(fecha):
        resp = input(f"Hay pendientes en maestro para {fecha}. ¿Enviar solo esos? (s/N): ").strip().lower()
        usar_pend = resp == "s"
    _enviar_telegram_desde_maestro(fecha, destinos=destinos, usar_pendientes=usar_pend)


def main():
    modalidad = input("¿Deseas generar picks (P), revisar resultados (R), picks manuales (M) o enviar a Telegram (T)? ").strip().lower()
    if modalidad == "r":
        _flujo_resultados()
        return
    if modalidad == "t":
        _flujo_telegram()
        return
    if modalidad == "m":
        _flujo_picks(force_manual=True, solo_manual=True)
    else:
        _flujo_picks()


if __name__ == "__main__":
    main()
