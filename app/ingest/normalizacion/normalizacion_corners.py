from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd


_NUMERIC_COLS = [
    "team_a_fh_corners",
    "team_b_fh_corners",
    "team_a_2h_corners",
    "team_b_2h_corners",
    "team_a_corners",
    "team_b_corners",
    "corner_fh_count",
    "corner_2h_count",
    "total_corner_count",
    "team_a_corners_0_10_min",
    "team_b_corners_0_10_min",
    "corners_potential",
    "corners_o75_potential",
    "corners_o85_potential",
    "corners_o95_potential",
    "corners_o105_potential",
    "odds_corners_over_75",
    "odds_corners_over_85",
    "odds_corners_over_95",
    "odds_corners_over_105",
    "odds_corners_over_115",
    "odds_corners_under_75",
    "odds_corners_under_85",
    "odds_corners_under_95",
    "odds_corners_under_105",
    "odds_corners_under_115",
    "odds_corners_1",
    "odds_corners_x",
    "odds_corners_2",
]


def _ensure_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df


def _numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def corners_to_df(normalized: List[Dict[str, Any]]) -> pd.DataFrame:
    """Normaliza la respuesta del endpoint para métricas de córners HT/FT."""

    if not normalized:
        return pd.DataFrame(columns=[
            "match_id",
            "season_id",
            "competition_id",
            "home_id",
            "away_id",
            "home",
            "away",
            "kickoff_local_cdmx",
            "team_a_fh_corners",
            "team_b_fh_corners",
            "team_a_2h_corners",
            "team_b_2h_corners",
            "team_a_corners",
            "team_b_corners",
            "corner_fh_count",
            "corner_2h_count",
            "total_corner_count",
            "corners_potential",
            "corners_o85_potential",
            "corners_o95_potential",
            "corners_o105_potential",
            "odds_corners_over_85",
            "odds_corners_over_95",
            "odds_corners_over_105",
        ])

    df = pd.DataFrame(normalized)
    df = _ensure_cols(df, _NUMERIC_COLS)
    df = _numeric(df, _NUMERIC_COLS)

    df["home_ht_corners"] = df["team_a_fh_corners"]
    df["away_ht_corners"] = df["team_b_fh_corners"]
    df["ht_total"] = df[["home_ht_corners", "away_ht_corners"]].sum(axis=1, min_count=1)
    df["ht_total"] = df["ht_total"].fillna(df["corner_fh_count"])

    df["home_2h_corners"] = df["team_a_2h_corners"]
    df["away_2h_corners"] = df["team_b_2h_corners"]
    df["second_half_total"] = df[["home_2h_corners", "away_2h_corners"]].sum(axis=1, min_count=1)
    df["second_half_total"] = df["second_half_total"].fillna(df["corner_2h_count"])

    df["home_ft_corners"] = df["team_a_corners"]
    df["away_ft_corners"] = df["team_b_corners"]

    both_halves_home = df["team_a_fh_corners"].notna() & df["team_a_2h_corners"].notna()
    both_halves_away = df["team_b_fh_corners"].notna() & df["team_b_2h_corners"].notna()

    df.loc[df["home_ft_corners"].isna() & both_halves_home, "home_ft_corners"] = (
        df.loc[df["home_ft_corners"].isna() & both_halves_home, "team_a_fh_corners"] +
        df.loc[df["home_ft_corners"].isna() & both_halves_home, "team_a_2h_corners"]
    )
    df.loc[df["away_ft_corners"].isna() & both_halves_away, "away_ft_corners"] = (
        df.loc[df["away_ft_corners"].isna() & both_halves_away, "team_b_fh_corners"] +
        df.loc[df["away_ft_corners"].isna() & both_halves_away, "team_b_2h_corners"]
    )

    df["ft_total"] = df[["home_ft_corners", "away_ft_corners"]].sum(axis=1, min_count=1)
    df["ft_total"] = df["ft_total"].fillna(df["total_corner_count"])

    df["ft_total_proxy"] = df["ft_total"].fillna(df["corners_potential"])

    df["corner_ratio_ht_ft"] = np.where(
        df["ft_total"].notna() & df["ft_total"] > 0,
        df["ht_total"] / df["ft_total"],
        np.nan,
    )

    # Añade duplicados homeID/awayID para facilitar joins históricos
    df["homeID"] = df.get("home_id")
    df["awayID"] = df.get("away_id")

    ordered_cols = [
        "match_id",
        "season_id",
        "competition_id",
        "home_id",
        "away_id",
        "homeID",
        "awayID",
        "home",
        "away",
        "kickoff_local_cdmx",
        "home_ht_corners",
        "away_ht_corners",
        "ht_total",
        "home_2h_corners",
        "away_2h_corners",
        "second_half_total",
        "home_ft_corners",
        "away_ft_corners",
        "ft_total",
        "ft_total_proxy",
        "corner_ratio_ht_ft",
        "corners_potential",
        "corners_o75_potential",
        "corners_o85_potential",
        "corners_o95_potential",
        "corners_o105_potential",
        "odds_corners_over_75",
        "odds_corners_over_85",
        "odds_corners_over_95",
        "odds_corners_over_105",
        "odds_corners_over_115",
        "odds_corners_under_75",
        "odds_corners_under_85",
        "odds_corners_under_95",
        "odds_corners_under_105",
        "odds_corners_under_115",
        "odds_corners_1",
        "odds_corners_x",
        "odds_corners_2",
    ]

    for c in ordered_cols:
        if c not in df.columns:
            df[c] = np.nan

    return df[ordered_cols]
