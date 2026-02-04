import functools
from typing import Dict, Any, List, Tuple

import pandas as pd
import requests

from app.common.config import settings


@functools.lru_cache(maxsize=8)
def _fetch_league_list() -> List[Dict[str, Any]]:
    url = f"{settings.footystats_url}/league-list"
    params = {"key": settings.footystats_key}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    payload = r.json()
    return payload.get("data", []) or []


def _resolve_recent_seasons(competition_id: int, current_season_id: int = None, current_season_label: str = None, n: int = 2) -> List[int]:
    leagues = _fetch_league_list()
    for league in leagues:
        comp = league.get("competition_id") or league.get("id")
        if not comp or int(comp) != int(competition_id):
            continue
        seasons = league.get("season", [])
        entries = sorted(
            [
                (
                    int(str(s.get("year", "0"))[:4] or 0),
                    int(s.get("id")),
                    str(s.get("year"))
                )
                for s in seasons
                if s.get("id")
            ],
            key=lambda x: x[0]
        )
        ids = [sid for _, sid, _ in entries]
        idx = None
        if current_season_id and current_season_id in ids:
            idx = ids.index(current_season_id)
        elif current_season_label:
            idx = next((i for i, entry in enumerate(entries) if entry[2] == str(current_season_label)), None)
        if idx is None:
            idx = len(ids) - 1
        start = max(idx - (n - 1), 0)
        return ids[start:idx + 1]
    return [current_season_id]


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
        if not pager:
            break
        if page >= pager.get("max_page", page):
            break
        page += 1
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _rate_home(row):
    juegos = row["matches_home"]
    if juegos >= 5:
        return row["goles_home"] / juegos if juegos else 0
    total = row["matches_total"]
    return row["goles_total"] / total if total else 0


def _rate_concede_home(row):
    juegos = row["matches_home"]
    if juegos >= 5:
        return row["concede_home"] / juegos if juegos else 0
    total = row["matches_total"]
    return row["concede_total"] / total if total else 0


def _rate_away(row):
    juegos = row["matches_away"]
    if juegos >= 5:
        return row["goles_away"] / juegos if juegos else 0
    total = row["matches_total"]
    return row["goles_total"] / total if total else 0


def _rate_concede_away(row):
    juegos = row["matches_away"]
    if juegos >= 5:
        return row["concede_away"] / juegos if juegos else 0
    total = row["matches_total"]
    return row["concede_total"] / total if total else 0


def _compute_streaks(series: pd.Series) -> Tuple[int, int]:
    streak = 0
    max_streak = 0
    for val in series:
        if val == 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return streak, max_streak


def _build_team_stats(df_matches: pd.DataFrame) -> pd.DataFrame:
    if df_matches.empty:
        return pd.DataFrame()

    df = df_matches.copy()
    for col in ["homeGoals", "awayGoals"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "date_unix" in df.columns:
        df["match_ts"] = pd.to_datetime(df["date_unix"], unit="s", errors="coerce")
    else:
        df["match_ts"] = pd.to_datetime(df.get("date"), errors="coerce")

    home_group = df.groupby("homeID").agg(
        goles_home=("homeGoals", "sum"),
        concede_home=("awayGoals", "sum"),
        matches_home=("homeGoals", "count"),
    )

    away_group = df.groupby("awayID").agg(
        goles_away=("awayGoals", "sum"),
        concede_away=("homeGoals", "sum"),
        matches_away=("awayGoals", "count"),
    )

    total_df = pd.concat([
        df.rename(columns={"homeID": "team_id", "homeGoals": "gf", "awayGoals": "ga"})[["team_id", "gf", "ga"]],
        df.rename(columns={"awayID": "team_id", "awayGoals": "gf", "homeGoals": "ga"})[["team_id", "gf", "ga"]],
    ], ignore_index=True)

    total_group = total_df.groupby("team_id").agg(
        goles_total=("gf", "sum"),
        concede_total=("ga", "sum"),
        matches_total=("gf", "count"),
    )

    stats = home_group.join(away_group, how="outer").join(total_group, how="outer")

    home_streaks = []
    for team_id, grp in df.groupby("homeID"):
        grp = grp.sort_values("match_ts")
        streak, max_streak = _compute_streaks(grp["awayGoals"])
        home_streaks.append((team_id, streak, max_streak))
    home_streaks = pd.DataFrame(home_streaks, columns=["team_id", "cs_home_streak", "cs_home_max"]).set_index("team_id")

    away_streaks = []
    for team_id, grp in df.groupby("awayID"):
        grp = grp.sort_values("match_ts")
        streak, max_streak = _compute_streaks(grp["homeGoals"])
        away_streaks.append((team_id, streak, max_streak))
    away_streaks = pd.DataFrame(away_streaks, columns=["team_id", "cs_away_streak", "cs_away_max"]).set_index("team_id")

    stats = stats.join(home_streaks, how="left").join(away_streaks, how="left")
    stats = stats.fillna(0)
    try:
        stats.index = stats.index.astype(int)
    except Exception:
        pass
    stats["rate_anota_home"] = stats.apply(_rate_home, axis=1)
    stats["rate_encaja_home"] = stats.apply(_rate_concede_home, axis=1)
    stats["rate_anota_away"] = stats.apply(_rate_away, axis=1)
    stats["rate_encaja_away"] = stats.apply(_rate_concede_away, axis=1)
    return stats


def obtener_contexto_btts(competition_id: int, season_id: int = None, season_label: str = None) -> pd.DataFrame:
    season_ids = _resolve_recent_seasons(competition_id, season_id, season_label, n=2)
    df_matches = pd.concat(
        [_fetch_league_matches(sid) for sid in season_ids if sid],
        ignore_index=True
    )
    return _build_team_stats(df_matches)


def enriquecer_btts(df_btts: pd.DataFrame) -> pd.DataFrame:
    df = df_btts.copy()
    if df.empty or "competition_id" not in df.columns:
        return df

    grouped = df.groupby(["competition_id", "season_id", "season_label"])
    enriched_parts = []
    for (comp_id, season_id, season_label), part in grouped:
        try:
            stats = obtener_contexto_btts(
                int(comp_id) if pd.notna(comp_id) else None,
                int(season_id) if pd.notna(season_id) else None,
                season_label
            )
        except Exception:
            enriched_parts.append(part)
            continue
        stats = stats.reset_index().rename(columns={"index": "team_id"})
        stats.index = stats["team_id"]

        local_cols = ["rate_anota_home", "rate_encaja_home", "cs_home_streak", "cs_home_max"]
        visita_cols = ["rate_anota_away", "rate_encaja_away", "cs_away_streak", "cs_away_max"]

        part = part.merge(
            stats[local_cols],
            left_on="home_id",
            right_index=True,
            how="left"
        ).rename(columns={
            "rate_anota_home": "local_rate_anota_home",
            "rate_encaja_home": "local_rate_encaja_home",
            "cs_home_streak": "local_cs_streak",
            "cs_home_max": "local_cs_max",
        })

        part = part.merge(
            stats[visita_cols],
            left_on="away_id",
            right_index=True,
            how="left"
        ).rename(columns={
            "rate_anota_away": "visita_rate_anota_away",
            "rate_encaja_away": "visita_rate_encaja_away",
            "cs_away_streak": "visita_cs_streak",
            "cs_away_max": "visita_cs_max",
        })
        part["flag_romper_racha_local"] = (
            (part["local_cs_streak"].fillna(0) >= 2) &
            (part["local_cs_max"].fillna(0) <= 2)
        )
        part["flag_romper_racha_visita"] = (
            (part["visita_cs_streak"].fillna(0) >= 2) &
            (part["visita_cs_max"].fillna(0) <= 2)
        )
        enriched_parts.append(part)

    if enriched_parts:
        df = pd.concat(enriched_parts, ignore_index=True)
    return df
