import functools
from typing import List, Dict

import pandas as pd
import requests

from app.common.config import settings


@functools.lru_cache(maxsize=8)
def _fetch_league_list() -> List[Dict]:
    url = f"{settings.footystats_url}/league-list"
    params = {"key": settings.footystats_key}
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    payload = resp.json()
    return payload.get("data", []) or []


def _resolve_recent_seasons(competition_id: int, season_id: int = None,
                            season_label: str = None, n: int = 2) -> List[int]:
    leagues = _fetch_league_list()
    for league in leagues:
        comp = league.get("competition_id") or league.get("id")
        if not comp or int(comp) != int(competition_id):
            continue
        entries = sorted(
            [
                (
                    int(str(s.get("year", "0"))[:4] or 0),
                    int(s.get("id")),
                    str(s.get("year"))
                )
                for s in league.get("season", [])
                if s.get("id")
            ],
            key=lambda x: x[0]
        )
        ids = [sid for _, sid, _ in entries]
        idx = None
        if season_id and season_id in ids:
            idx = ids.index(season_id)
        elif season_label:
            idx = next((i for i, entry in enumerate(entries) if entry[2] == str(season_label)), None)
        if idx is None:
            idx = len(ids) - 1
        start = max(idx - (n - 1), 0)
        return ids[start:idx + 1]
    return [season_id] if season_id else []


@functools.lru_cache(maxsize=64)
def _fetch_league_matches(season_id: int) -> pd.DataFrame:
    url = f"{settings.footystats_url}/league-matches"
    params = {
        "key": settings.footystats_key,
        "season_id": season_id,
        "max_per_page": 1000,
    }
    rows = []
    page = 1
    while True:
        params["page"] = page
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        payload = resp.json()
        data = payload.get("data", []) or payload
        if not data:
            break
        rows.extend(data)
        pager = payload.get("pager", {})
        if not pager or page >= pager.get("max_page", page):
            break
        page += 1
    return pd.DataFrame(rows)


def _current_streak_zero(series: pd.Series) -> int:
    streak = 0
    for val in reversed(series.tolist()):
        if val == 0:
            streak += 1
        else:
            break
    return streak


def _build_corner_stats(df_matches: pd.DataFrame) -> pd.DataFrame:
    if df_matches.empty:
        return pd.DataFrame()

    df = df_matches.copy()
    # Acepta múltiples nombres de campos para córners por equipo
    def _corner_series(df: pd.DataFrame, keys: list[str]) -> pd.Series:
        for k in keys:
            if k in df.columns:
                s = pd.to_numeric(df.get(k), errors="coerce")
                if s.notna().any():
                    return s
        return pd.Series(pd.NA, index=df.index)

    df["team_a_corners"] = _corner_series(df, [
        "team_a_corners", "homeCorners", "home_corners", "home_corner_count", "corners_home",
        "team_a_corner_count"
    ])
    df["team_b_corners"] = _corner_series(df, [
        "team_b_corners", "awayCorners", "away_corners", "away_corner_count", "corners_away",
        "team_b_corner_count"
    ])

    df["total_corners"] = pd.to_numeric(
        df.get("totalCornerCount") or df.get("total_corners"), errors="coerce"
    )
    df["total_corners"] = df["total_corners"].fillna(
        df["team_a_corners"].fillna(0) + df["team_b_corners"].fillna(0)
    )
    if "date_unix" in df.columns:
        df["match_ts"] = pd.to_datetime(df["date_unix"], unit="s", errors="coerce")
    else:
        df["match_ts"] = pd.to_datetime(df.get("date"), errors="coerce")

    records = []
    team_frames = []
    for side in ("homeID", "awayID"):
        if side not in df.columns:
            continue
        if side == "homeID":
            part = df.assign(
                team_id=df["homeID"],
                team_corners=df["team_a_corners"],
                rival_corners=df["team_b_corners"],
            )
        else:
            part = df.assign(
                team_id=df["awayID"],
                team_corners=df["team_b_corners"],
                rival_corners=df["team_a_corners"],
            )
        team_frames.append(part[["team_id", "match_ts", "team_corners", "rival_corners", "total_corners"]])

    merged = pd.concat(team_frames, ignore_index=True)

    for team_id, grp in merged.groupby("team_id"):
        grp = grp.sort_values("match_ts")
        over85_series = grp["total_corners"].fillna(0) >= 9
        stats = {
            "team_id": team_id,
            "matches_corners": len(grp),
            "corners_over85_rate": over85_series.mean() if len(grp) else 0,
            "corners_total_avg": grp["total_corners"].mean(),
            "team_corners_avg": grp["team_corners"].mean(),
            "team_corners_against_avg": grp["rival_corners"].mean(),
            "under85_streak": _current_streak_zero(over85_series.astype(int)),
        }
        records.append(stats)

    return pd.DataFrame(records).set_index("team_id")


def obtener_contexto_corners(competition_id: int, season_id: int = None,
                             season_label: str = None) -> pd.DataFrame:
    season_ids = _resolve_recent_seasons(competition_id, season_id, season_label, n=2)
    frames = [_fetch_league_matches(sid) for sid in season_ids if sid]
    if not frames:
        return pd.DataFrame()
    df_matches = pd.concat(frames, ignore_index=True)
    stats = _build_corner_stats(df_matches)
    if stats.empty:
        return stats
    try:
        stats.index = stats.index.astype(int)
    except Exception:
        pass
    return stats


def enriquecer_corners(df_corners: pd.DataFrame) -> pd.DataFrame:
    df = df_corners.copy()
    required = {"competition_id", "season_id", "home_id", "away_id"}
    if df.empty or not required <= set(df.columns):
        return df
    # Rellenar home_id/away_id con homeID/awayID si existen
    if "homeID" in df.columns:
        df["home_id"] = df.get("home_id").fillna(df["homeID"])
    if "awayID" in df.columns:
        df["away_id"] = df.get("away_id").fillna(df["awayID"])

    parts = []
    grouped = df.groupby(["competition_id", "season_id"])
    for (comp_id, season_id), part in grouped:
        try:
            stats = obtener_contexto_corners(
                int(comp_id) if pd.notna(comp_id) else None,
                int(season_id) if pd.notna(season_id) else None,
            )
        except Exception:
            parts.append(part)
            continue
        if stats.empty:
            parts.append(part)
            continue
        part = part.merge(
            stats.add_prefix("local_"),
            left_on="home_id",
            right_index=True,
            how="left"
        )
        part = part.merge(
            stats.add_prefix("visita_"),
            left_on="away_id",
            right_index=True,
            how="left"
        )
        parts.append(part)

    if parts:
        df = pd.concat(parts, ignore_index=True)
    return df
