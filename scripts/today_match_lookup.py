import argparse
import json
from datetime import date
from typing import Any

from app.ingest.today_match import get_todays_matches_normalized


def main():
    parser = argparse.ArgumentParser(description="Consulta un partido por match_id en todays-matches.")
    parser.add_argument("--id", required=True, help="match_id a consultar")
    parser.add_argument("--date", default=date.today().isoformat(), help="Fecha YYYY-MM-DD (por defecto hoy)")
    args = parser.parse_args()

    matches = get_todays_matches_normalized(args.date)
    objetivo = [m for m in matches if str(m.get("match_id")) == str(args.id)]

    if not objetivo:
        print(f"No se encontró el match_id {args.id} en todays-matches para {args.date}")
        return

    print(f"Se encontró {len(objetivo)} coincidencia(s) para match_id {args.id} en {args.date}:\n")
    for m in objetivo:
        print(json.dumps(m, ensure_ascii=False, indent=2, default=str))
        print("-" * 40)


if __name__ == "__main__":
    main()
