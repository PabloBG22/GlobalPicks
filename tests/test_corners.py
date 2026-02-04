import json
import unittest
from unittest.mock import patch

import pandas as pd

from app.ingest.normalizacion.normalizacion_corners import corners_to_df
from app.ingest.today_match import _normalize_match
from app.ingest.mercados import corners as corners_market


class CornersNormalizationTest(unittest.TestCase):

    def setUp(self):
        with open("tests/data/match_day.json", encoding="utf-8") as f:
            payload = json.load(f)
        # fixture del endpoint (un único partido en data)
        self.raw_match = payload["data"]

    def test_normalize_match_includes_corner_fields(self):
        normalized = _normalize_match(self.raw_match)
        self.assertIn("corners_potential", normalized)
        self.assertEqual(normalized["corners_potential"], 13.15)
        self.assertEqual(normalized["odds_corners_over_95"], 1.6)
        self.assertIsNone(normalized["team_a_corners"], "-1 debe normalizarse a None para stats faltantes")

    def test_corners_to_df_builds_totals(self):
        normalized = [{
            "match_id": 1,
            "season_id": 10,
            "competition_id": 999,
            "home": "Local",
            "away": "Visita",
            "kickoff_local_cdmx": "2024-01-01 10:00",
            "team_a_fh_corners": 3,
            "team_b_fh_corners": 2,
            "team_a_2h_corners": 4,
            "team_b_2h_corners": 3,
            "team_a_corners": 7,
            "team_b_corners": 5,
            "corner_fh_count": 5,
            "corner_2h_count": 7,
            "total_corner_count": 12,
            "corners_potential": 10.5,
            "corners_o85_potential": 65,
            "corners_o95_potential": 55,
            "corners_o105_potential": 40,
            "odds_corners_over_85": 1.45,
            "odds_corners_over_95": 1.7,
            "odds_corners_over_105": 2.1,
        }]

        df = corners_to_df(normalized)
        self.assertAlmostEqual(df.loc[0, "ht_total"], 5)
        self.assertAlmostEqual(df.loc[0, "ft_total"], 12)
        self.assertAlmostEqual(df.loc[0, "corner_ratio_ht_ft"], 5 / 12)
        self.assertAlmostEqual(df.loc[0, "ft_total_proxy"], 12)

    def test_build_corners_market_df_creates_rows(self):
        normalized = [{
            "match_id": 25,
            "season_id": 33,
            "competition_id": 555,
            "home": "Belgrano",
            "away": "San Martín",
            "kickoff_local_cdmx": "2024-01-01 12:00",
            "team_a_fh_corners": 4,
            "team_b_fh_corners": 2,
            "team_a_2h_corners": 3,
            "team_b_2h_corners": 4,
            "team_a_corners": 7,
            "team_b_corners": 6,
            "corner_fh_count": 6,
            "corner_2h_count": 7,
            "total_corner_count": 13,
            "corners_potential": 80.0,
            "corners_o75_potential": 75.0,
            "corners_o85_potential": 70.0,
            "corners_o95_potential": 55.0,
            "corners_o105_potential": 45.0,
            "odds_corners_over_75": 1.35,
            "odds_corners_over_85": 1.55,
            "odds_corners_over_95": 1.8,
            "odds_corners_over_105": 2.2,
        },
        {
            "match_id": 30,
            "season_id": 33,
            "competition_id": 555,
            "home": "Equipo B",
            "away": "Equipo C",
            "kickoff_local_cdmx": "2024-01-01 15:00",
            "team_a_fh_corners": 2,
            "team_b_fh_corners": 2,
            "team_a_2h_corners": 3,
            "team_b_2h_corners": 3,
            "team_a_corners": 5,
            "team_b_corners": 5,
            "corner_fh_count": 4,
            "corner_2h_count": 6,
            "total_corner_count": 10,
            "corners_potential": 80.0,
            "corners_o75_potential": 70.0,
            "corners_o85_potential": 50.0,
            "corners_o95_potential": 40.0,
            "corners_o105_potential": 30.0,
            "odds_corners_over_75": 1.30,
            "odds_corners_over_85": 1.60,
            "odds_corners_over_95": 1.9,
            "odds_corners_over_105": 2.5,
        }]
        base = corners_to_df(normalized)
        fake_ligas = pd.DataFrame([
            {"season_id": 555, "Liga": "Primera", "Pais": "Argentina"}
        ])
        cfg = {
            "min_corners_ht": 4,
            "min_corners_ft": 9,
            "min_potential_ft": 10,
            "min_o85_potential": 70,
            "min_odds_over85": 1.25,
            "max_odds_over85": 2.75,
        }

        with patch.object(corners_market, "ligas", return_value=fake_ligas), \
             patch.object(corners_market, "anotar_estado", side_effect=lambda df, *_: df) as mock_estado:
            df = corners_market.build_corners_market_df(base, cfg)

        mock_estado.assert_called_once()
        self.assertEqual(set(df["Mercado"]), {"Over 8.5 corners"})
        self.assertEqual(len(df), 1)

        ft_row = df.iloc[0]
        self.assertAlmostEqual(ft_row["Corners_total"], 13)
        self.assertAlmostEqual(ft_row["Potencial_o85"], 70)
        self.assertAlmostEqual(ft_row["ODDS"], 1.55)
        self.assertEqual(ft_row["Estado"], "VERDE")
        self.assertEqual(ft_row["Acierto"], "Acierto")
        self.assertEqual(ft_row["PROB_Historico"], 70)
        self.assertIn("Prob_modelo", df.columns)
        self.assertIn("EV_modelo", df.columns)


if __name__ == "__main__":
    unittest.main()
