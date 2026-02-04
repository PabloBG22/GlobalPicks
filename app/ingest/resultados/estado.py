from __future__ import annotations

from datetime import datetime, timedelta
from functools import lru_cache
from typing import Optional, Any

import pandas as pd
import requests

from app.common.config import settings

try:
    from zoneinfo import ZoneInfo
except Exception:
    import pytz

    ZoneInfo = pytz.timezone  # type: ignore

TZ_CDMX = ZoneInfo("America/Mexico_City")


@lru_cache(maxsize=256)
@lru_cache(maxsize=64)
def _fetch_league_matches(season_id: int) -> pd.DataFrame:
    url = f"{settings.footystats_url}/league-matches"
    params = {"key": settings.footystats_key, "season_id": season_id, "max_per_page": 1000}
    rows = []
    page = 1
    while True:
        params["page"] = page
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        payload = r.json()
        data = payload.get("data", []) or payload
        if not data:
            break
        rows.extend(data)
        pager = payload.get("pager", {})
        if not pager or page >= pager.get("max_page", page):
            break
        page += 1
    return pd.DataFrame(rows)


_TODAY_CACHE: dict[str, list[dict]] = {}


def _fetch_today_matches(date_str: str) -> list[dict]:
    """
    Consulta /todays-matches para una fecha concreta y cachea el resultado.
    Se usa como primera fuente para córners finales.
    """
    if date_str in _TODAY_CACHE:
        return _TODAY_CACHE[date_str]
    url = f"{settings.footystats_url}/todays-matches"
    params = {
        "key": settings.footystats_key,
        "timezone": "America/Mexico_City",
        "date": date_str,
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        payload = r.json()
        data = payload.get("data", []) or payload
        _TODAY_CACHE[date_str] = data if isinstance(data, list) else []
    except Exception:
        _TODAY_CACHE[date_str] = []
    return _TODAY_CACHE[date_str]


def _find_today_match(date_str: str, match_id: Any, home_id: Any, away_id: Any) -> dict:
    """
    Busca un partido en todays-matches por match_id o por IDs de equipos.
    """
    pool = _fetch_today_matches(date_str)
    if not pool:
        return {}
    match_str = str(match_id) if match_id not in (None, "") else None
    try:
        hid = int(home_id)
    except Exception:
        hid = None
    try:
        aid = int(away_id)
    except Exception:
        aid = None
    for m in pool:
        mid = m.get("id") or m.get("match_id")
        if match_str and str(mid) == match_str:
            return m
        if hid is not None and aid is not None:
            try:
                mh = int(m.get("homeID") or m.get("home_team_id") or -1)
                ma = int(m.get("awayID") or m.get("away_team_id") or -1)
                if mh == hid and ma == aid:
                    return m
            except Exception:
                continue
    return {}


def _coalesce_match_id(df: pd.DataFrame) -> pd.Series:
    candidates = [c for c in ("match_id", "id", "fixture_id", "_match_id") if c in df.columns]
    if not candidates:
        return pd.Series(pd.NA, index=df.index)
    result = df[candidates[0]].astype(str)
    for col in candidates[1:]:
        result = result.fillna(df[col].astype(str))
    return result.replace({"nan": pd.NA, "None": pd.NA})


def _first_series(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Devuelve la primera columna existente; si ninguna, serie vacía con mismo index."""
    for col in columns:
        if col in df.columns:
            return df[col]
    return pd.Series(pd.NA, index=df.index)


def _find_match_league(season_id: int, match_id: Any, home_id: Any, away_id: Any, hora_str: str | None = None) -> dict:
    if season_id in (None, ""):
        return {}
    try:
        sid = int(season_id)
    except Exception:
        return {}
    df = _fetch_league_matches(sid)
    if df.empty:
        return {}
    df = df.copy()
    df["homeID"] = pd.to_numeric(_first_series(df, ["homeID", "home_id"]), errors="coerce")
    df["awayID"] = pd.to_numeric(_first_series(df, ["awayID", "away_id"]), errors="coerce")
    df["match_std"] = _coalesce_match_id(df)
    if "date_unix" in df.columns:
        df["match_ts"] = pd.to_datetime(df["date_unix"], unit="s", errors="coerce")
    elif "date" in df.columns:
        df["match_ts"] = pd.to_datetime(df["date"], errors="coerce")

    mask = pd.Series(True, index=df.index)
    try:
        hid = int(home_id)
        aid = int(away_id)
        mask &= (df["homeID"] == hid) & (df["awayID"] == aid)
    except Exception:
        pass
    if match_id not in (None, ""):
        mask_id = df["match_std"].astype(str) == str(match_id)
        if mask_id.any():
            mask &= mask_id
    subset = df.loc[mask]
    if subset.empty and match_id not in (None, ""):
        subset = df.loc[df["match_std"].astype(str) == str(match_id)]
    if subset.empty:
        subset = df.loc[(df["homeID"] == df["homeID"]) & (df["homeID"] == df["homeID"])]  # noop, fallback to empty
    if subset.empty:
        return {}

    target_ts = _parse_hora(hora_str) if hora_str else None
    if target_ts is not None and target_ts.tzinfo is not None:
        target_ts = target_ts.replace(tzinfo=None)
    if target_ts is not None and "match_ts" in subset.columns:
        mt = subset["match_ts"]
        if hasattr(mt, "dt"):
            try:
                mt = mt.dt.tz_localize(None)
            except Exception:
                try:
                    mt = mt.dt.tz_convert(None)
                except Exception:
                    pass
        subset = subset.assign(delta_ts=(mt - target_ts).abs())
        subset = subset.sort_values("delta_ts")
    elif "date_unix" in subset.columns:
        subset = subset.sort_values("date_unix", ascending=False)
    return subset.iloc[0].to_dict()


def _fetch_match_fdapi(match_id: Any) -> dict:
    """
    Consulta el endpoint /match de football-data-api.com para un match_id dado.
    """
    if not settings.football_data_api_key or match_id in (None, ""):
        return {}
    try:
        url = f"{settings.football_data_api_url.rstrip('/')}/match"
        params = {"key": settings.football_data_api_key, "match_id": match_id}
        r = requests.get(url, params=params, timeout=12)
        r.raise_for_status()
        payload = r.json()
        if isinstance(payload, dict):
            data = payload.get("data", payload.get("Data", payload))
            if isinstance(data, dict):
                return data
        return {}
    except Exception:
        return {}


def _find_match_data(season_id: int, match_id: Any, home_id: Any, away_id: Any, hora_str: str | None = None) -> dict:
    """
    Intenta encontrar datos de partido con football-data-api (/match) y si falla,
    cae a footystats (/league-matches).
    """
    data = _fetch_match_fdapi(match_id)
    if data:
        return data
    return _find_match_league(season_id, match_id, home_id, away_id, hora_str)


def _parse_hora(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = datetime.strptime(value, "%Y-%m-%d %H:%M")
        return dt.replace(tzinfo=TZ_CDMX)
    except Exception:
        return None


def _int_or_none(value) -> Optional[int]:
    try:
        iv = int(value)
        return iv if iv >= 0 else None
    except (TypeError, ValueError):
        return None


def _bool_or_none(value) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value in (None, "", "None", "none", "null"):
        return None
    if isinstance(value, (int, float)):
        return bool(value)
    val = str(value).strip().lower()
    if val in {"true", "1", "yes", "y", "t"}:
        return True
    if val in {"false", "0", "no", "n", "f"}:
        return False
    return None


def _status_finalizado(status: str) -> bool:
    """
    Devuelve True si el status indica partido finalizado.
    """
    s = (status or "").strip().lower()
    return s in {"complete", "completed", "finished", "fulltime", "ft"}


def _status_en_juego(status: str) -> bool:
    """
    Devuelve True si el status indica partido en curso.
    """
    s = (status or "").strip().lower()
    return s in {"inplay", "in-play", "live", "playing", "ongoing", "2h", "1h"}


def _home_away_corners(match_data: dict) -> tuple[Optional[int], Optional[int]]:
    """
    Intenta extraer las cuentas finales de córners por equipo desde league-matches,
    aceptando varias claves alternativas.
    """
    home = None
    away = None
    home_keys = ["team_a_corners", "homeCorners", "home_corners", "home_corner_count", "home_total_corners"]
    away_keys = ["team_b_corners", "awayCorners", "away_corners", "away_corner_count", "away_total_corners"]
    for key in home_keys:
        home = _int_or_none(match_data.get(key))
        if home is not None:
            break
    for key in away_keys:
        away = _int_or_none(match_data.get(key))
        if away is not None:
            break
    return home, away


def _marcadores(match_data: dict) -> tuple[Optional[str], Optional[str]]:
    """
    Obtiene marcadores HT y FT si existen en la data del partido.
    """
    def _maybe_int(*keys):
        for k in keys:
            if k in match_data and match_data.get(k) not in (None, ""):
                return _int_or_none(match_data.get(k))
        return None

    def _maybe_score_str(*keys) -> Optional[tuple[int, int]]:
        for k in keys:
            val = match_data.get(k)
            if not val:
                continue
            try:
                parts = str(val).split("-")
                if len(parts) == 2:
                    a, b = int(parts[0].strip()), int(parts[1].strip())
                    return a, b
            except Exception:
                continue
        return None

    ht_home = _maybe_int(
        "ht_goals_team_a", "homeHTGoals", "home_ht_goals", "homeGoalHT", "home_goal_ht"
    )
    ht_away = _maybe_int(
        "ht_goals_team_b", "awayHTGoals", "away_ht_goals", "awayGoalHT", "away_goal_ht"
    )
    if ht_home is None or ht_away is None:
        ht_score = _maybe_score_str("ht_score", "half_time_score", "score_ht")
        if ht_score:
            ht_home, ht_away = ht_score

    hg = _maybe_int(
        "homeGoalCount", "home_goals", "homeScore", "home_score", "home_team_score"
    )
    ag = _maybe_int(
        "awayGoalCount", "away_goals", "awayScore", "away_score", "away_team_score"
    )
    if hg is None or ag is None:
        ft_score = _maybe_score_str("ft_score", "fulltime_score", "full_time_score", "score", "result")
        if ft_score:
            hg, ag = ft_score

    marcador_ht = f"{ht_home}-{ht_away}" if ht_home is not None and ht_away is not None else None
    marcador_ft = f"{hg}-{ag}" if hg is not None and ag is not None else None
    return marcador_ht, marcador_ft


def _total_corners(match_data: dict) -> Optional[int]:
    """
    Intenta inferir el total de córners con tolerancia a distintos campos
    devueltos por la API (nombres varían entre endpoints).
    """
    # Campos agregados más habituales
    candidatos_total = [
        "totalCornerCount", "total_corner_count", "corner_total_count",
        "corner_count", "cornerCount", "corners_total", "cornersTotal", "corners",
    ]
    for key in candidatos_total:
        total = _int_or_none(match_data.get(key))
        if total is not None:
            return total

    # Fallback: suma de locales/visitantes con nombres alternativos
    candidatos_home = [
        "team_a_corners", "homeCorners", "home_corner_count", "home_corners", "home_total_corners"
    ]
    candidatos_away = [
        "team_b_corners", "awayCorners", "away_corner_count", "away_corners", "away_total_corners"
    ]
    home_corners = None
    away_corners = None
    for key in candidatos_home:
        home_corners = _int_or_none(match_data.get(key))
        if home_corners is not None:
            break
    for key in candidatos_away:
        away_corners = _int_or_none(match_data.get(key))
        if away_corners is not None:
            break

    if home_corners is not None and away_corners is not None:
        return home_corners + away_corners
    return None


def _evaluar_pick(market_group: str, mercado: str, match_data: dict, oteam: str = "") -> Tuple[str, dict]:
    hg = _int_or_none(match_data.get("homeGoalCount") or match_data.get("homeGoals") or match_data.get("homeScore") or match_data.get("home_score"))
    ag = _int_or_none(match_data.get("awayGoalCount") or match_data.get("awayGoals") or match_data.get("awayScore") or match_data.get("away_score"))
    extras = {}
    marcador_ht, marcador_ft = _marcadores(match_data)
    if marcador_ft and (hg is None or ag is None):
        try:
            p1, p2 = marcador_ft.split("-")
            hg = _int_or_none(p1)
            ag = _int_or_none(p2)
        except Exception:
            pass
    # Si tenemos goles numéricos pero no marcador, constrúyelo.
    if marcador_ft is None and hg is not None and ag is not None:
        marcador_ft = f"{hg}-{ag}"
    if marcador_ht is None and "HTGoalCount" in match_data:
        try:
            ht_total = int(match_data.get("HTGoalCount"))
            marcador_ht = f"{ht_total//2}-{ht_total - (ht_total//2)}" if ht_total >= 0 else marcador_ht
        except Exception:
            pass
    total = None if hg is None or ag is None else hg + ag

    if marcador_ht:
        extras["Marcador_HT"] = marcador_ht
    if marcador_ft:
        extras["Marcador_FT"] = marcador_ft
        extras["Marcador"] = marcador_ft
    total_corners = _total_corners(match_data)
    if total_corners is not None:
        extras["Corners_total"] = str(total_corners)

    status = str(match_data.get("status") or "")
    if status and not _status_finalizado(status):
        # Si no está finalizado, no evaluamos: queda PENDIENTE/LIVE según flujo superior.
        return "PENDIENTE", extras

    if market_group == "BTTS":
        if "YES" in mercado.upper():
            resultado = "VERDE" if hg > 0 and ag > 0 else "ROJO"
        else:
            resultado = "VERDE" if hg == 0 or ag == 0 else "ROJO"
        return resultado, extras

    if market_group == "OVER" and "1.5" in mercado:
        if total is None:
            return "PENDIENTE", extras
        return ("VERDE" if total >= 2 else "ROJO"), extras
    if market_group == "OVER" and "2.5" in mercado:
        return ("VERDE" if total >= 3 else "ROJO"), extras

    if market_group == "GOLHT":
        ht = match_data.get("HTGoalCount")
        if ht is None and marcador_ht:
            try:
                partes = marcador_ht.split("-")
                if len(partes) == 2:
                    ht = int(partes[0]) + int(partes[1])
            except Exception:
                ht = None
        if ht is None:
            return "PENDIENTE", extras
        try:
            ht = int(ht)
        except Exception:
            return "PENDIENTE", extras
        return ("VERDE" if ht >= 1 else "ROJO"), extras

    if market_group == "CORNERS":
        # Prefiere la señal de API over85Corners; si no existe, usa total_corners.
        over85 = None
        for key in ("over85Corners", "over85corners", "over_85_corners", "over_8_5_corners"):
            over85 = _bool_or_none(match_data.get(key))
            if over85 is not None:
                break
        home_corners, away_corners = _home_away_corners(match_data)
        if total_corners is None:
            total_corners = _total_corners(match_data)
        if total_corners is None and home_corners is not None and away_corners is not None:
            total_corners = home_corners + away_corners
        if total_corners is not None:
            extras["Corners_total"] = str(total_corners)
            extras["Corners_FT"] = str(total_corners)
        if home_corners is not None:
            extras["Corners_local"] = home_corners
        if away_corners is not None:
            extras["Corners_visitante"] = away_corners
        if over85 is not None:
            return ("VERDE" if over85 else "ROJO"), extras
        if total_corners is None:
            return "PENDIENTE", extras
        extras["Corners_FT"] = str(total_corners)
        return ("VERDE" if total_corners >= 9 else "ROJO"), extras

    if market_group == "TEAM_OVER":
        if total is None or hg is None or ag is None:
            return "PENDIENTE", extras
        goles_eq = hg if oteam.lower() == "local" else ag
        extras["Marcador"] = f"{hg}-{ag}"
        return ("VERDE" if goles_eq >= 2 else "ROJO"), extras

    return "PENDIENTE", extras


def anotar_estado(df: pd.DataFrame, market_group: str, forzar_busqueda: bool = False) -> pd.DataFrame:
    if df.empty or "Hora" not in df.columns:
        return df

    df_out = df.copy()
    estados = []
    meta = []
    now = datetime.now(tz=TZ_CDMX)
    margen = timedelta(hours=2)

    for _, row in df_out.iterrows():
        hora = _parse_hora(str(row.get("Hora", "")))
        match_id = row.get("Match_id") or row.get("Partido") or row.get("ID_partido")
        season_id = row.get("Season_id") or row.get("season_id") or row.get("competition_id")
        home_id = row.get("home_id") or row.get("homeID")
        away_id = row.get("away_id") or row.get("awayID")

        # Si no tenemos llaves mínimas, no evaluamos.
        if match_id in (None, "") or season_id in (None, ""):
            estados.append("PENDIENTE")
            meta.append({})
            continue

        # Para córners, intenta primero todays-matches (misma fecha de Hora).
        if market_group == "CORNERS" and hora is not None:
            match_today = _find_today_match(str(hora.date()), match_id, home_id, away_id)
            if match_today:
                estado, info = _evaluar_pick(market_group, str(row.get("Mercado", "")), match_today, oteam=str(row.get("OTeam", "")))
                estados.append(estado)
                meta.append(info)
                continue

        # Para no marcar anticipadamente, si el partido no debe haber terminado, lo dejamos LIVE/PENDIENTE
        if hora is not None and not forzar_busqueda:
            if hora > now:
                estados.append("LIVE" if (hora - now) <= margen else "PENDIENTE")
                meta.append({})
                continue
            if now - hora <= margen:
                estados.append("LIVE")
                meta.append({})
                continue

        match_data = _find_match_data(season_id, match_id, home_id, away_id, str(row.get("Hora")))
        if not match_data:
            # Si la API falla, como último recurso usa Corners_total local si existe.
            if market_group == "CORNERS":
                total_local = pd.to_numeric(row.get("Corners_total"), errors="coerce")
                if pd.notna(total_local):
                    estado_local = "VERDE" if total_local >= 9 else "ROJO"
                    estados.append(estado_local)
                    meta.append({"Corners_FT": str(int(total_local))})
                    continue
            estados.append("PENDIENTE")
            meta.append({})
            continue

        status = str(match_data.get("status") or "")
        if status and not _status_finalizado(status):
            # Si el status no es finalizado, usa LIVE/PENDIENTE según tiempo/flag.
            if hora is not None:
                if hora > now:
                    estados.append("LIVE" if (hora - now) <= margen else "PENDIENTE")
                elif now - hora <= margen:
                    estados.append("LIVE")
                else:
                    estados.append("PENDIENTE")
            else:
                estados.append("PENDIENTE")
            meta.append({})
            continue

        # Se evalúa siempre con league-matches; si falta algún campo clave, se deja PENDIENTE.
        mercado = str(row.get("Mercado", ""))
        estado, info = _evaluar_pick(market_group, mercado, match_data, oteam=str(row.get("OTeam", "")))
        estados.append(estado)
        meta.append(info)

    df_out["Estado"] = estados
    if meta:
        extra_df = pd.DataFrame(meta, index=df_out.index)
        for col in extra_df.columns:
            if col not in df_out.columns:
                df_out[col] = extra_df[col]
            else:
                extra_col = extra_df[col]
                # Para métricas clave de resultado preferimos el valor de la API si llega.
                if col in {"Corners_total", "Corners_FT", "Marcador", "Marcador_HT", "Marcador_FT"}:
                    extra_valid = extra_col.notna() & (extra_col != "")
                    df_out[col] = df_out[col].where(~extra_valid, extra_col)
                else:
                    # Rellena NaN o cadenas vacías
                    keep_mask = df_out[col].notna() & (df_out[col] != "")
                    df_out[col] = df_out[col].where(keep_mask, extra_col)
    return df_out
