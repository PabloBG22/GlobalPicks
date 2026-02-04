import requests
from datetime import datetime, timezone,timedelta
from typing import Any, Dict, List, Optional
# from app.ingest.mercados_normalizacion import over25_to_df, golht_to_df
from app.ingest.normalizacion.normalizacion_btts import btts_to_df
from app.ingest.normalizacion.normalizacion_over25 import over25_to_df
import json 


try:
    from zoneinfo import ZoneInfo
    TZ_CDMX = ZoneInfo("America/Mexico_City")
except Exception:
    import pytz
    TZ_CDMX = pytz.timezone("America/Mexico_City")

from app.common.config import settings


def _to_cdmx_from_epoch(epoch_sec: Optional[int]) -> Optional[str]:
    if not epoch_sec:
        return None
    # epoch viene en segundos UTC
    dt_utc = datetime.fromtimestamp(int(epoch_sec), tz=timezone.utc)
    return dt_utc.astimezone(TZ_CDMX).strftime("%Y-%m-%d %H:%M")


def _to_float_pos(x):
    try:
        v = float(x)
        return v if v > 0 else None   # trata 0/"0"/<=0 como no disponible
    except (TypeError, ValueError):
        return None


def _to_nonneg_float(x):
    """Convierte a float >= 0; devuelve None para valores negativos o inválidos."""
    if x is None:
        return None
    try:
        v = float(x)
        return v if v >= 0 else None
    except (TypeError, ValueError):
        return None


def _to_nonneg_int(x):
    """Convierte a int >= 0 preservando ceros; devuelve None si no aplica."""
    if x is None:
        return None
    try:
        v = int(float(x))
        return v if v >= 0 else None
    except (TypeError, ValueError):
        return None

def _normalize_match(m: Dict[str, Any]) -> Dict[str, Any]:
    return {
        # básicos
        "match_id": m.get("id"),
        "competition_id": m.get("competition_id"),
        "season_id": m.get("season_id"),
        "season_label": m.get("season"),
        "home_id": m.get("homeID") or m.get("home_team_id"),
        "away_id": m.get("awayID") or m.get("away_team_id"),
        "home": m.get("home_name") or m.get("home"),
        "away": m.get("away_name") or m.get("away"),
        "kickoff_local_cdmx": _to_cdmx_from_epoch(m.get("date_unix")),
        "status": m.get("status"),
        "referee_id": _to_nonneg_int(m.get("refereeID")),

        # odds (mantén las claves originales del JSON + casteo)
        "odds_btts_yes": _to_float_pos(m.get("odds_btts_yes")),
        "odds_btts_no":  _to_float_pos(m.get("odds_btts_no")),

        "odds_over25":  _to_float_pos(m.get("odds_ft_over25") or m.get("odds_over25")),
        "odds_under25": _to_float_pos(m.get("odds_ft_under25") or m.get("odds_under25")),
        "odds_over15":  _to_float_pos(m.get("odds_ft_over15") or m.get("odds_over15") or m.get("odds_o15")),
        "odds_under15": _to_float_pos(m.get("odds_ft_under15") or m.get("odds_under15") or m.get("odds_u15")),

        # ⚽️ Gol HT (usa EXACTAMENTE las llaves del JSON)
        "odds_1st_half_over05":  _to_float_pos(m.get("odds_1st_half_over05")),
        "odds_1st_half_under05": _to_float_pos(m.get("odds_1st_half_under05")),

        # xG
        "team_a_xg_prematch": _to_float_pos(m.get("team_a_xg_prematch")),
        "team_b_xg_prematch": _to_float_pos(m.get("team_b_xg_prematch")),
        "total_xg_prematch":  _to_float_pos(m.get("total_xg_prematch")),

        # potentials (0-100)
        "btts_potential":     m.get("btts_potential"),
        "btts_fhg_potential": m.get("btts_fhg_potential"),   # señal HT
        "o05HT_potential":    m.get("o05HT_potential"),
        "o15HT_potential":    m.get("o15HT_potential"),
        "o15_potential":      m.get("o15_potential"),
        "o25_potential":      m.get("o25_potential"),
        "u25_potential":      m.get("u25_potential"),         # <-- corregido (antes copiabas o25)
        "cards_potential":    m.get("cards_potential"),

        # 🥅 Córners
        "corners_potential":     _to_nonneg_float(m.get("corners_potential")),
        "corners_o75_potential": _to_nonneg_float(m.get("corners_o75_potential")),
        "corners_o85_potential": _to_nonneg_float(m.get("corners_o85_potential")),
        "corners_o95_potential": _to_nonneg_float(m.get("corners_o95_potential")),
        "corners_o105_potential": _to_nonneg_float(m.get("corners_o105_potential")),

        # Odds FT de córners
        "odds_corners_over_75":  _to_float_pos(m.get("odds_corners_over_75")),
        "odds_corners_over_85":  _to_float_pos(m.get("odds_corners_over_85")),
        "odds_corners_over_95":  _to_float_pos(m.get("odds_corners_over_95")),
        "odds_corners_over_105": _to_float_pos(m.get("odds_corners_over_105")),
        "odds_corners_over_115": _to_float_pos(m.get("odds_corners_over_115")),
        "odds_corners_under_75": _to_float_pos(m.get("odds_corners_under_75")),
        "odds_corners_under_85": _to_float_pos(m.get("odds_corners_under_85")),
        "odds_corners_under_95": _to_float_pos(m.get("odds_corners_under_95")),
        "odds_corners_under_105": _to_float_pos(m.get("odds_corners_under_105")),
        "odds_corners_under_115": _to_float_pos(m.get("odds_corners_under_115")),
        "odds_corners_1":        _to_float_pos(m.get("odds_corners_1")),
        "odds_corners_x":        _to_float_pos(m.get("odds_corners_x")),
        "odds_corners_2":        _to_float_pos(m.get("odds_corners_2")),

        # Estadísticas registradas HT/FT
        "team_a_corners":        _to_nonneg_int(m.get("team_a_corners")),
        "team_b_corners":        _to_nonneg_int(m.get("team_b_corners")),
        "total_corner_count":    _to_nonneg_int(m.get("totalCornerCount")),
        "team_a_fh_corners":     _to_nonneg_int(m.get("team_a_fh_corners")),
        "team_b_fh_corners":     _to_nonneg_int(m.get("team_b_fh_corners")),
        "team_a_2h_corners":     _to_nonneg_int(m.get("team_a_2h_corners")),
        "team_b_2h_corners":     _to_nonneg_int(m.get("team_b_2h_corners")),
        "corner_fh_count":       _to_nonneg_int(m.get("corner_fh_count")),
        "corner_2h_count":       _to_nonneg_int(m.get("corner_2h_count")),
        "team_a_corners_0_10_min": _to_nonneg_int(m.get("team_a_corners_0_10_min")),
        "team_b_corners_0_10_min": _to_nonneg_int(m.get("team_b_corners_0_10_min")),
        "team_a_cards_num":      _to_nonneg_int(m.get("team_a_cards_num")),
        "team_b_cards_num":      _to_nonneg_int(m.get("team_b_cards_num")),
        "total_cards_num":       _to_nonneg_int(m.get("team_a_cards_num")) + _to_nonneg_int(m.get("team_b_cards_num")) if _to_nonneg_int(m.get("team_a_cards_num")) is not None and _to_nonneg_int(m.get("team_b_cards_num")) is not None else _to_nonneg_int(m.get("team_a_cards_num")) or _to_nonneg_int(m.get("team_b_cards_num")),
    }





def get_todays_matches_normalized(fecha) -> List[Dict[str, Any]]:
    """
    Llama /todays-matches, convierte la hora a CDMX con date_unix
    y devuelve una lista de dicts normalizados para BTTS / Gol HT / Over 2.5.
    """
    url = f"{settings.footystats_url}/todays-matches"
    params = {
        "key": settings.footystats_key,
        # Si el proveedor soporta timezone por query, mantenlo. Igual convertimos por epoch.
        "timezone": "America/Mexico_City",
        # "date" : hoy
        "date" : fecha
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        payload = r.json()
        matches_raw = payload.get("data", []) or payload  # algunos devuelven lista directa
      

# # Nombre del archivo
#         outfile = os.path.join(BASE_DIR, "match_day.json")
#         with open(outfile, "w", encoding="utf-8") as f:
#             json.dump(matches_raw, f, ensure_ascii=False, indent=2)


        normalized = [_normalize_match(m) for m in matches_raw if isinstance(m, dict)]

        print("✅ Conexión exitosa")
        print(f"📦 Partidos hoy: {len(normalized)}\n")

    
        return normalized

    except requests.exceptions.HTTPError:
        print(f"❌ Error HTTP {r.status_code}")
        print(r.text[:400])
        return []
    except requests.exceptions.RequestException as e:
        print(f"❌ Error de conexión: {e}")
        return []


# --- (Opcional) Resolver nombres de liga por competition_id ---
# Si tienes/encuentras un endpoint que entregue {competition_id -> competition_name},
# puedes cachearlo aquí y enriquecer los normalizados.

# def resolve_competition_names(ids: List[int]) -> Dict[int, str]:
#     url = f"{settings.footystats_url}/competition-list"
#     params = {"key": settings.footystats_key}
#     r = requests.get(url, params=params, timeout=10)
#     r.raise_for_status()
#     data = r.json().get("data", [])
#     mapping = {}
#     for comp in data:
#         cid = comp.get("competition_id") or comp.get("id")
#         name = comp.get("competition_name") or comp.get("name")
#         if cid and name:
#             mapping[int(cid)] = name
#     return mapping


# if __name__ == "__main__":
#     get_todays_matches_normalized()
    
