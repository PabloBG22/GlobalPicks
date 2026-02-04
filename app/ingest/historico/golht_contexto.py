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


def _current_streak_true(series: pd.Series) -> int:
    streak = 0
    for val in reversed(series.tolist()):
        if val:
            streak += 1
        else:
            break
    return streak


def _build_ht_stats(df_matches: pd.DataFrame) -> pd.DataFrame:
    if df_matches.empty:
        return pd.DataFrame()

    df = df_matches.copy()
    df["ht_goals_team_a"] = pd.to_numeric(
        df.get("ht_goals_team_a") or df.get("htGoalsHome"), errors="coerce"
    )
    df["ht_goals_team_b"] = pd.to_numeric(
        df.get("ht_goals_team_b") or df.get("htGoalsAway"), errors="coerce"
    )
    if "date_unix" in df.columns:
        df["match_ts"] = pd.to_datetime(df["date_unix"], unit="s", errors="coerce")
    else:
        df["match_ts"] = pd.to_datetime(df.get("date"), errors="coerce")

    records = []
    for team_id, grp in pd.concat([
        df.assign(team_id=df["homeID"],
                  ht_for=df["ht_goals_team_a"],
                  ht_against=df["ht_goals_team_b"]),
        df.assign(team_id=df["awayID"],
                  ht_for=df["ht_goals_team_b"],
                  ht_against=df["ht_goals_team_a"]),
    ], ignore_index=True).groupby("team_id"):
        grp = grp.sort_values("match_ts")
        scored_series = grp["ht_for"].fillna(0) > 0
        conceded_series = grp["ht_against"].fillna(0) > 0
        any_goal = scored_series | conceded_series
        stats = {
            "team_id": team_id,
            "matches_ht": len(grp),
            "ht_scored_rate": scored_series.mean() if len(grp) else 0,
            "ht_conceded_rate": conceded_series.mean() if len(grp) else 0,
            "ht_goal_rate": any_goal.mean() if len(grp) else 0,
            "ht_score_streak": _current_streak_true(scored_series),
            "ht_cs_streak": _current_streak_zero(grp["ht_against"].fillna(0)),
        }
        records.append(stats)

    return pd.DataFrame(records).set_index("team_id")


def _prepare_matches_for_ht(df_matches: pd.DataFrame) -> pd.DataFrame:
    if df_matches.empty:
        return df_matches
    df = df_matches.copy()
    if "date_unix" in df.columns:
        df["match_ts"] = pd.to_datetime(df["date_unix"], unit="s", errors="coerce")
    else:
        df["match_ts"] = pd.to_datetime(df.get("date"), errors="coerce")

    def _first_available(columns):
        for col in columns:
            if col in df.columns:
                return df[col]
        return pd.Series(pd.NA, index=df.index)

    df["_home_id"] = pd.to_numeric(_first_available(["homeID", "home_id", "homeId"]), errors="coerce")
    df["_away_id"] = pd.to_numeric(_first_available(["awayID", "away_id", "awayId"]), errors="coerce")
    df["homeID"] = df["_home_id"]
    df["awayID"] = df["_away_id"]
    df["_home_name"] = _first_available(["home_name", "homeName", "home_team"])
    df["_away_name"] = _first_available(["away_name", "awayName", "away_team"])
    df["_league_name"] = _first_available(["league_name", "competition_name", "season"])
    df["_match_id"] = _first_available(["id", "fixture_id", "match_id"])

    ht_home = _first_available(["ht_goals_team_a", "htGoalsHome", "ht_score_home", "ht_home_goals"])
    ht_away = _first_available(["ht_goals_team_b", "htGoalsAway", "ht_score_away", "ht_away_goals"])
    df["_ht_home"] = pd.to_numeric(ht_home, errors="coerce")
    df["_ht_away"] = pd.to_numeric(ht_away, errors="coerce")

    # asegura columnas estándar usadas en stats
    df["ht_goals_team_a"] = df["_ht_home"]
    df["ht_goals_team_b"] = df["_ht_away"]
    df["HTGoalCount"] = df["_ht_home"].fillna(0) + df["_ht_away"].fillna(0)
    df["_o05ht_potential"] = _first_available(["o05HT_potential", "o05ht_potential"])

    return df


@functools.lru_cache(maxsize=64)
def _cached_partidos_golht(comp_id: int, season_id: int = None, season_label: str = None) -> pd.DataFrame:
    if comp_id is None:
        return pd.DataFrame()
    season_ids = _resolve_recent_seasons(comp_id, season_id, season_label, n=2)
    frames = [_fetch_league_matches(sid) for sid in season_ids if sid]
    if not frames:
        return pd.DataFrame()
    df_matches = pd.concat(frames, ignore_index=True)
    return _prepare_matches_for_ht(df_matches)


def obtener_partidos_golht(competition_id: int, season_id: int = None,
                           season_label: str = None) -> pd.DataFrame:
    try:
        comp = int(competition_id)
    except (TypeError, ValueError):
        return pd.DataFrame()
    sid = None
    if season_id not in (None, ""):
        try:
            sid = int(season_id)
        except (TypeError, ValueError):
            sid = None
    label = None
    if season_label not in (None, "") and not pd.isna(season_label):
        label = str(season_label)
    df = _cached_partidos_golht(comp, sid, label)
    return df.copy()


def obtener_contexto_golht(competition_id: int, season_id: int = None,
                           season_label: str = None) -> pd.DataFrame:
    df_matches = obtener_partidos_golht(competition_id, season_id, season_label)
    if df_matches.empty:
        return pd.DataFrame()
    stats = _build_ht_stats(df_matches)
    if stats.empty:
        return stats
    try:
        stats.index = stats.index.astype(int)
    except Exception:
        pass
    return stats


def enriquecer_golht(df_ht: pd.DataFrame) -> pd.DataFrame:
    df = df_ht.copy()
    required = {"competition_id", "season_id", "season_label", "home_id", "away_id"}
    if df.empty or not required <= set(df.columns):
        return df

    # Alinea tipos a enteros para que el merge con stats no pierda coincidencias
    for col in ["competition_id", "season_id", "home_id", "away_id"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    parts = []
    grouped = df.groupby(["competition_id", "season_id", "season_label"])
    for (comp_id, season_id, season_label), part in grouped:
        try:
            stats = obtener_contexto_golht(
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
