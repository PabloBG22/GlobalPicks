import unittest

from app.ingest.resultados import estado


class EstadoCornersTest(unittest.TestCase):

    def test_corner_pick_green(self):
        data = {
            "totalCornerCount": 10,
            "status": "complete",
        }
        result = estado._evaluar_pick("CORNERS", "Corners FT", data)
        self.assertEqual(result, "VERDE")

    def test_corner_pick_red_with_team_totals(self):
        data = {
            "team_a_corners": 3,
            "team_b_corners": 4,
            "status": "complete",
        }
        result = estado._evaluar_pick("CORNERS", "Corners FT", data)
        self.assertEqual(result, "ROJO")

    def test_corner_pick_pending_when_missing(self):
        data = {"status": "complete"}
        result = estado._evaluar_pick("CORNERS", "Corners FT", data)
        self.assertEqual(result, "PENDIENTE")


if __name__ == "__main__":
    unittest.main()
