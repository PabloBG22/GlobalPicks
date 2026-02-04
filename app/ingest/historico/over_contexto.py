import functools
from typing import List, Any

import pandas as pd
import requests

from app.common.config import settings


@functools.lru_cache(maxsize=8)
def _fetch_league_list() -> List[dict]:
    url = f"{settings.footystats_url}/league-list"
    params = {"key": settings.footystats_key}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    payload = r.json()
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
        r = requests.get(url, params=params, timeout=20)
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


def _current_streak(series: pd.Series) -> int:
    streak = 0
    for val in reversed(series.tolist()):
        if val:
            streak += 1
        else:
            break
    return streak


def _max_streak(series: pd.Series) -> int:
    best = cur = 0
    for val in series.tolist():
        if val:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def _build_team_over_stats(df_matches: pd.DataFrame) -> pd.DataFrame:
    if df_matches.empty:
        return pd.DataFrame()
    df = df_matches.copy()
    df["homeGoals"] = pd.to_numeric(df["homeGoals"], errors="coerce")
    df["awayGoals"] = pd.to_numeric(df["awayGoals"], errors="coerce")
    df["total_goals"] = df["homeGoals"] + df["awayGoals"]

    if "date_unix" in df.columns:
        df["match_ts"] = pd.to_datetime(df["date_unix"], unit="s", errors="coerce")
    else:
        df["match_ts"] = pd.to_datetime(df.get("date"), errors="coerce")

    records = []
    for team_id, grp in pd.concat([
        df.assign(team_id=df["homeID"]),
        df.assign(team_id=df["awayID"])
    ]).groupby("team_id"):
        grp = grp.sort_values("match_ts")
        over15_series = grp["total_goals"] >= 2
        over25_series = grp["total_goals"] >= 3
        stats = {
            "team_id": team_id,
            "matches_total": len(grp),
            "over15_rate": over15_series.mean() if len(grp) else 0,
            "over25_rate": over25_series.mean() if len(grp) else 0,
            "under15_streak": _current_streak(~over15_series),
            "under15_max": _max_streak(~over15_series),
            "under25_streak": _current_streak(~over25_series),
            "under25_max": _max_streak(~over25_series),
        }
        records.append(stats)
    return pd.DataFrame(records).set_index("team_id")


def obtener_contexto_over(competition_id: int, season_id: int = None, season_label: str = None) -> pd.DataFrame:
    season_ids = _resolve_recent_seasons(competition_id, season_id, season_label, n=2)
    frames = [_fetch_league_matches(sid) for sid in season_ids if sid]
    if not frames:
        return pd.DataFrame()
    df_matches = pd.concat(frames, ignore_index=True)
    stats = _build_team_over_stats(df_matches)
    if stats.empty:
        return stats
    try:
        stats.index = stats.index.astype(int)
    except Exception:
        pass
    stats["flag_romper_over15"] = (
        (stats["under15_streak"].fillna(0) >= 2) &
        (stats["under15_max"].fillna(0) <= 2)
    )
    stats["flag_romper_over25"] = (
        (stats["under25_streak"].fillna(0) >= 2) &
        (stats["under25_max"].fillna(0) <= 2)
    )
    return stats


def enriquecer_over(df_over: pd.DataFrame) -> pd.DataFrame:
    df = df_over.copy()
    required_cols = {"competition_id", "season_id", "season_label", "home_id", "away_id"}
    if df.empty or not required_cols <= set(df.columns):
        return df

    grouped = df.groupby(["competition_id", "season_id", "season_label"])
    parts = []
    for (comp_id, season_id, season_label), part in grouped:
        try:
            stats = obtener_contexto_over(
                int(comp_id) if pd.notna(comp_id) else None,
                int(season_id) if pd.notna(season_id) else None,
                season_label
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
