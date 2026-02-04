import argparse
import json
import sys
import requests

from app.common.config import settings


def _matches_for_season(season_id: int):
    """Itera todas las páginas de league-matches para una season."""
    url = f"{settings.footystats_url}/league-matches"
    params = {
        "key": settings.footystats_key,
        "season_id": season_id,
        "max_per_page": 1000,
    }
    page = 1
    while True:
        params["page"] = page
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        payload = r.json()
        data = payload.get("data", []) or payload
        for row in data:
            yield row
        pager = payload.get("pager", {})
        if not pager or page >= pager.get("max_page", page):
            break
        page += 1


def _match_id_value(row):
    for key in ("match_id", "id", "fixture_id", "_match_id"):
        if key in row and row.get(key) not in (None, ""):
            return str(row.get(key))
    return None


def main():
    parser = argparse.ArgumentParser(description="Consulta league-matches por match_id filtrando en una season.")
    parser.add_argument("--id", required=True, help="match_id a buscar (se compara contra match_id/id/fixture_id)")
    parser.add_argument("--season-id", type=int, required=True, help="season_id en la que buscar")
    args = parser.parse_args()

    target = str(args.id)
    encontrados = []
    try:
        for row in _matches_for_season(args.season_id):
            mid = _match_id_value(row)
            if mid is not None and mid == target:
                encontrados.append(row)
    except requests.HTTPError as e:
        print(f"HTTP error: {e}")
        sys.exit(1)
    except requests.RequestException as e:
        print(f"Request error: {e}")
        sys.exit(1)

    if not encontrados:
        print(f"No se encontró match_id {target} en season {args.season_id}")
        return

    print(f"Encontrado(s) {len(encontrados)} partido(s) para match_id {target} en season {args.season_id}:\n")
    for row in encontrados:
        print(json.dumps(row, ensure_ascii=False, indent=2, default=str))
        print("-" * 60)


if __name__ == "__main__":
    main()
