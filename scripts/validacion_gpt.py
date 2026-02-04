#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from openai import OpenAI

from app.common.config import settings

PROMPTS_DIR = Path("prompts")
HIST_WINDOW = 10
MIN_SAMPLE = 6
DEBUG_GPT = os.getenv("GPT_DEBUG", "").strip().lower() in {"1", "true", "yes"}
DEBUG_LIMIT = int(os.getenv("GPT_DEBUG_LIMIT", "5"))
_DEBUG_COUNT = 0


def _leer_system_prompt() -> str:
    archivo = PROMPTS_DIR / "system.txt"
    if archivo.exists():
        contenido = archivo.read_text(encoding="utf-8").strip()
        if contenido:
            return contenido
    return (
        "Eres un clasificador binario para apuestas deportivas. "
        "Responde únicamente con 'True' o 'False'."
    )


def _leer_prompt(mercado: str) -> str:
    nombre = (mercado or "generico").strip().lower().replace(" ", "_")
    nombre = nombre.replace(",", ".")
    # Alias explícitos
    alias = {
        "team_over_1.5": "team15",
        "team_over_15": "team15",
        "team_over1.5": "team15",
        "team_over1_5": "team15",
        "over1.5": "over_1.5",
        "over_1_5": "over_1.5",
        "over_1,5": "over_1.5",
        "over15": "over_1.5",
    }
    nombre = alias.get(nombre, nombre)
    prompt_file = PROMPTS_DIR / f"{nombre}.txt"
    if not prompt_file.exists():
        prompt_file = PROMPTS_DIR / "default.txt"
    if prompt_file.exists():
        return prompt_file.read_text(encoding="utf-8").strip()
    return "Genera un breve análisis con base en el pick y su histórico."


def _leer_json(path: str | None) -> Any:
    if not path:
        return None
    archivo = Path(path)
    if not archivo.exists():
        return None
    try:
        data = json.loads(archivo.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data


def _json_compact(data: Any) -> str:
    if data is None:
        return ""
    try:
        return json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return str(data)


def _to_number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(num) or num < 0:
        return None
    return num


def _to_int(value: Any) -> int | None:
    num = _to_number(value)
    if num is None:
        return None
    try:
        return int(num)
    except (TypeError, ValueError):
        return None


def _match_timestamp(match: dict) -> int:
    for key in ("date_unix", "dateUnix", "date_unix_ts"):
        ts = _to_number(match.get(key))
        if ts is not None:
            return int(ts)
    for key in ("Fecha", "date", "match_ts"):
        raw = match.get(key)
        if isinstance(raw, datetime):
            return int(raw.timestamp())
        if isinstance(raw, str) and raw:
            try:
                return int(datetime.fromisoformat(raw).timestamp())
            except Exception:
                pass
    return 0


def _rate(values: list[bool | None]) -> float | None:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return round(sum(1 for v in vals if v) / len(vals), 3)


def _avg(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 2)


def _streak(values: list[bool | None]) -> dict[str, Any]:
    vals = [int(v) for v in values if v is not None]
    if not vals:
        return {}
    seq = ",".join(str(v) for v in vals)
    zero_streak = 0
    one_streak = 0
    for v in reversed(vals):
        if v == 0:
            zero_streak += 1
        else:
            break
    for v in reversed(vals):
        if v == 1:
            one_streak += 1
        else:
            break
    return {"seq": seq, "zero_streak": zero_streak, "one_streak": one_streak}


def _filter_pick_data(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    block_exact = {
        "GPT_valido",
        "GPT_explicacion",
        "Estado",
        "Acierto",
        "Rescate",
        "Marcador",
        "Marcador_HT",
        "Marcador_FT",
        "Fecha_ejecucion",
        "homeGoals_today",
        "awayGoals_today",
    }
    block_substr = (
        "prob",
        "potencial",
        "ev_",
        "ventaja",
        "roi",
        "gpt",
        "estado",
        "acierto",
        "rescate",
        "unidades",
        "marcador",
        "goles_",
        "score",
    )
    filtered = {}
    for key, value in data.items():
        if key in block_exact:
            continue
        key_lower = str(key).lower()
        if any(part in key_lower for part in block_substr):
            continue
        if value is None:
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        filtered[key] = value
    return filtered


def _summarize_team(matches: list[dict], team_id: int | None) -> dict[str, Any]:
    if team_id is None:
        return {}
    if not matches:
        return {}

    btts = []
    over15 = []
    ht_goal = []
    team_scored = []
    team_over15 = []
    team_conceded = []
    team_conceded2 = []
    corners_o85 = []

    total_goals = []
    team_goals = []
    total_corners = []
    total_cards = []
    ht_goals = []

    for match in matches:
        home_id = _to_int(match.get("homeID"))
        away_id = _to_int(match.get("awayID"))
        if team_id not in (home_id, away_id):
            continue
        home_goals = _to_number(match.get("homeGoalCount"))
        away_goals = _to_number(match.get("awayGoalCount"))
        total = _to_number(match.get("totalGoalCount"))
        if total is None and home_goals is not None and away_goals is not None:
            total = home_goals + away_goals
        ht_total = _to_number(match.get("HTGoalCount"))
        if ht_total is None:
            ht_a = _to_number(match.get("ht_goals_team_a"))
            ht_b = _to_number(match.get("ht_goals_team_b"))
            if ht_a is not None and ht_b is not None:
                ht_total = ht_a + ht_b
        corners_a = _to_number(match.get("team_a_corners"))
        corners_b = _to_number(match.get("team_b_corners"))
        cards_a = _to_number(match.get("team_a_cards_num"))
        cards_b = _to_number(match.get("team_b_cards_num"))
        corners_total = (corners_a + corners_b) if corners_a is not None and corners_b is not None else None
        cards_total = (cards_a + cards_b) if cards_a is not None and cards_b is not None else None

        if home_goals is not None and away_goals is not None:
            btts.append(bool(home_goals > 0 and away_goals > 0))
        else:
            btts.append(None)

        over15.append(bool(total >= 2) if total is not None else None)
        ht_goal.append(bool(ht_total >= 1) if ht_total is not None else None)

        if team_id == home_id:
            t_goals = home_goals
            o_goals = away_goals
        else:
            t_goals = away_goals
            o_goals = home_goals

        team_scored.append(bool(t_goals >= 1) if t_goals is not None else None)
        team_over15.append(bool(t_goals >= 2) if t_goals is not None else None)
        team_conceded.append(bool(o_goals >= 1) if o_goals is not None else None)
        team_conceded2.append(bool(o_goals >= 2) if o_goals is not None else None)

        corners_o85.append(bool(corners_total >= 9) if corners_total is not None else None)

        total_goals.append(total)
        team_goals.append(t_goals)
        total_corners.append(corners_total)
        total_cards.append(cards_total)
        ht_goals.append(ht_total)

    samples = {
        "matches": len(matches),
        "goals": len([v for v in total_goals if v is not None]),
        "ht": len([v for v in ht_goals if v is not None]),
        "corners": len([v for v in total_corners if v is not None]),
        "cards": len([v for v in total_cards if v is not None]),
    }
    summary = {
        "matches": len(matches),
        "samples": samples,
        "rates": {
            "btts": _rate(btts),
            "over15": _rate(over15),
            "ht_goal": _rate(ht_goal),
            "team_scored": _rate(team_scored),
            "team_over15": _rate(team_over15),
            "team_conceded": _rate(team_conceded),
            "team_conceded2": _rate(team_conceded2),
            "corners_o85": _rate(corners_o85),
        },
        "averages": {
            "total_goals": _avg(total_goals),
            "team_goals": _avg(team_goals),
            "total_corners": _avg(total_corners),
            "total_cards": _avg(total_cards),
        },
        "streaks": {
            "btts": _streak(btts),
            "over15": _streak(over15),
            "ht_goal": _streak(ht_goal),
            "team_over15": _streak(team_over15),
            "corners_o85": _streak(corners_o85),
        },
    }
    return summary


def _summarize_h2h(matches: list[dict]) -> dict[str, Any]:
    if not matches:
        return {}
    btts = []
    over15 = []
    ht_goal = []
    corners_o85 = []
    total_goals = []
    total_corners = []
    total_cards = []
    ht_goals = []

    for match in matches:
        home_goals = _to_number(match.get("homeGoalCount"))
        away_goals = _to_number(match.get("awayGoalCount"))
        total = _to_number(match.get("totalGoalCount"))
        if total is None and home_goals is not None and away_goals is not None:
            total = home_goals + away_goals
        ht_total = _to_number(match.get("HTGoalCount"))
        if ht_total is None:
            ht_a = _to_number(match.get("ht_goals_team_a"))
            ht_b = _to_number(match.get("ht_goals_team_b"))
            if ht_a is not None and ht_b is not None:
                ht_total = ht_a + ht_b
        corners_a = _to_number(match.get("team_a_corners"))
        corners_b = _to_number(match.get("team_b_corners"))
        cards_a = _to_number(match.get("team_a_cards_num"))
        cards_b = _to_number(match.get("team_b_cards_num"))
        corners_total = (corners_a + corners_b) if corners_a is not None and corners_b is not None else None
        cards_total = (cards_a + cards_b) if cards_a is not None and cards_b is not None else None

        if home_goals is not None and away_goals is not None:
            btts.append(bool(home_goals > 0 and away_goals > 0))
        else:
            btts.append(None)
        over15.append(bool(total >= 2) if total is not None else None)
        ht_goal.append(bool(ht_total >= 1) if ht_total is not None else None)
        corners_o85.append(bool(corners_total >= 9) if corners_total is not None else None)

        total_goals.append(total)
        total_corners.append(corners_total)
        total_cards.append(cards_total)
        ht_goals.append(ht_total)

    samples = {
        "matches": len(matches),
        "goals": len([v for v in total_goals if v is not None]),
        "ht": len([v for v in ht_goals if v is not None]),
        "corners": len([v for v in total_corners if v is not None]),
        "cards": len([v for v in total_cards if v is not None]),
    }
    return {
        "matches": len(matches),
        "samples": samples,
        "rates": {
            "btts": _rate(btts),
            "over15": _rate(over15),
            "ht_goal": _rate(ht_goal),
            "corners_o85": _rate(corners_o85),
        },
        "averages": {
            "total_goals": _avg(total_goals),
            "total_corners": _avg(total_corners),
            "total_cards": _avg(total_cards),
        },
        "streaks": {
            "btts": _streak(btts),
            "over15": _streak(over15),
            "ht_goal": _streak(ht_goal),
            "corners_o85": _streak(corners_o85),
        },
    }


def _build_summary(pick_data: Any, hist_data: Any) -> dict[str, Any]:
    if not isinstance(hist_data, list):
        return {}
    if not hist_data:
        return {}
    home_id = None
    away_id = None
    if isinstance(pick_data, dict):
        home_id = _to_int(pick_data.get("homeID") or pick_data.get("home_id"))
        away_id = _to_int(pick_data.get("awayID") or pick_data.get("away_id"))

    matches_sorted = sorted(hist_data, key=_match_timestamp)

    def _recent(matches: list[dict]) -> list[dict]:
        if len(matches) <= HIST_WINDOW:
            return matches
        return matches[-HIST_WINDOW:]

    summary: dict[str, Any] = {
        "window": HIST_WINDOW,
        "min_sample": MIN_SAMPLE,
        "order": "oldest_to_newest",
    }

    if home_id is not None:
        home_matches = [
            m for m in matches_sorted
            if _to_int(m.get("homeID")) == home_id or _to_int(m.get("awayID")) == home_id
        ]
        local_summary = _summarize_team(_recent(home_matches), home_id)
        if local_summary:
            summary["local"] = local_summary
        home_home = [m for m in home_matches if _to_int(m.get("homeID")) == home_id]
        if home_home:
            summary["local_home"] = _summarize_team(_recent(home_home), home_id)

    if away_id is not None:
        away_matches = [
            m for m in matches_sorted
            if _to_int(m.get("homeID")) == away_id or _to_int(m.get("awayID")) == away_id
        ]
        away_summary = _summarize_team(_recent(away_matches), away_id)
        if away_summary:
            summary["visitante"] = away_summary
        away_away = [m for m in away_matches if _to_int(m.get("awayID")) == away_id]
        if away_away:
            summary["visitante_away"] = _summarize_team(_recent(away_away), away_id)

    if home_id is not None and away_id is not None:
        h2h_matches = [
            m for m in matches_sorted
            if { _to_int(m.get("homeID")), _to_int(m.get("awayID")) } == {home_id, away_id}
        ]
        if h2h_matches:
            summary["h2h"] = _summarize_h2h(_recent(h2h_matches))

    return summary


def _parse_bool_response(text: str | None) -> bool | None:
    if not text:
        return None
    cleaned = text.strip().lower()
    match = re.search(r"\b(true|false)\b", cleaned)
    if not match:
        return None
    return match.group(1) == "true"


def resultados_gpt(
    mercado: str,
    local: str,
    visitante: str,
    fecha: str,
    hora: str,
    json_pick_path: str | None = None,
    json_hist_path: str | None = None,
    trends_text: str | None = None,
) -> dict:
    """Construye un prompt en función del mercado e incluye datos JSON."""
    prompt_base = _leer_prompt(mercado)
    local = (local or "").strip()
    visitante = (visitante or "").strip()
    fecha = (fecha or "").strip() or datetime.now(ZoneInfo("America/Mexico_City")).strftime("%Y-%m-%d")
    hora = (hora or "").strip() or "00:00"
    datos_pick = _leer_json(json_pick_path)
    datos_hist = _leer_json(json_hist_path)
    pick_filtrado = _filter_pick_data(datos_pick)
    resumen_hist = _build_summary(datos_pick, datos_hist)
    trends_text = (trends_text or "").strip()
    if trends_text:
        trends_block = f"\n\nTendencias (Trends):\n{trends_text}"
    else:
        trends_block = "\n\nTendencias (Trends): Sin datos"
    oteam = ""
    if isinstance(datos_pick, dict):
        oteam = str(datos_pick.get("OTeam") or "").strip()
    oteam_line = f"\nEquipo objetivo (OTeam): {oteam}" if oteam else ""
    contenido_usuario = (
        f"Mercado: {mercado}\nPartido: {local} vs {visitante}\nFecha: {fecha} {hora}"
        f"{oteam_line}\n\n"
        f"Instrucciones: {prompt_base}\n\n"
        f"Datos del pick (filtrado, sin modelo): {_json_compact(pick_filtrado)}\n\n"
        f"Resumen histórico (ventana={HIST_WINDOW}, min_muestra={MIN_SAMPLE}, orden=antiguo->reciente): "
        f"{_json_compact(resumen_hist)}\n"
        "Notas: rachas/streaks usan 1=Sí, 0=No; el final de la secuencia es lo más reciente."
        f"{trends_block}"
    )
    system_prompt = _leer_system_prompt()
    if DEBUG_GPT:
        global _DEBUG_COUNT
        if _DEBUG_COUNT < DEBUG_LIMIT:
            _DEBUG_COUNT += 1
            resumen_keys = list(resumen_hist.keys()) if isinstance(resumen_hist, dict) else []
            local_samples = {}
            visitante_samples = {}
            h2h_samples = {}
            if isinstance(resumen_hist, dict):
                local_samples = (resumen_hist.get("local") or {}).get("samples") or {}
                visitante_samples = (resumen_hist.get("visitante") or {}).get("samples") or {}
                h2h_samples = (resumen_hist.get("h2h") or {}).get("samples") or {}
            print(
                "[GPT_DEBUG] mercado=%s partido=%s vs %s oteam=%s resumen_keys=%s local_samples=%s visitante_samples=%s h2h_samples=%s"
                % (mercado, local, visitante, oteam, resumen_keys, local_samples, visitante_samples, h2h_samples)
            )
    cliente = OpenAI(api_key=settings.gp_key)
    response = cliente.chat.completions.create(
        model="gpt-5-mini",
        max_completion_tokens=4,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": contenido_usuario,
            },
        ]
    )
    mensaje = response.choices[0].message.content
    if not mensaje:
        return {"valido": False, "explicacion": "Sin respuesta del modelo"}
    parsed = _parse_bool_response(mensaje)
    if DEBUG_GPT:
        if _DEBUG_COUNT <= DEBUG_LIMIT:
            print(f"[GPT_DEBUG] raw={mensaje!r} parsed={parsed}")
    if parsed is None:
        return {"valido": False, "explicacion": f"Respuesta inválida: {mensaje.strip()}"}
    return {"valido": parsed, "explicacion": "" if parsed else mensaje.strip()}


def main() -> int:
    """Ejecución de prueba."""
    mensaje = resultados_gpt(
        "Gol HT",
        "Local",
        "Visitante",
        datetime.now().date().isoformat(),
        "12:00",
    )
    print(mensaje)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
