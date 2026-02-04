import re
from functools import lru_cache

import pandas as pd
import requests

from app.common.config import settings

EMPTY_COLUMNS = [
    "league_name",
    "country",
    "season_id",
    "competition_id",
    "year",
    "season_label",
    "season_start",
    "season_end",
]

def _parse_season_year(raw_year):
    if raw_year is None:
        return None, None, ""
    s = str(raw_year).strip()
    if not s:
        return None, None, ""
    # Busca años con 4 dígitos (maneja "2025/2026", "2025-2026", etc.)
    years = re.findall(r"\d{4}", s)
    if len(years) >= 2:
        y0, y1 = int(years[0]), int(years[1])
        return y0, y1, f"{y0}/{y1}"
    if len(years) == 1:
        y0 = int(years[0])
        return y0, y0, str(y0)
    # Fallback por dígitos puros ("20252026")
    digits = re.sub(r"\D", "", s)
    if len(digits) == 8:
        y0, y1 = int(digits[:4]), int(digits[4:])
        return y0, y1, f"{y0}/{y1}"
    if len(digits) == 4:
        y0 = int(digits)
        return y0, y0, str(y0)
    return None, None, s


@lru_cache(maxsize=1)
def _ligas_cached():
    url = f"{settings.footystats_url}/league-list"
    params = {
        "key": settings.footystats_key,
        "chosen_leagues_only": "true",
            }

    empty_df = pd.DataFrame(columns=EMPTY_COLUMNS)
    try:
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()

        # print("✅ Conexión exitosa")
        leagues = data.get("data", [])
        # print(f"📦 Total ligas disponibles: {len(leagues)}")

        # Mostrar las primeras 10 ligas como ejemplo
        # print(leagues)
        rows = []

        for league in leagues:  # cada liga del JSON
            name = league.get("name") or league.get("league_name")
            country = league.get("country")
            comp_id = league.get("competition_id") or league.get("id")

            for s in league.get("season", []):  # cada temporada dentro de la liga
                season_start, season_end, season_label = _parse_season_year(s.get("year"))
                rows.append({
                    "league_name": name,
                    "country": country,
                    "season_id": s.get("id"),
                    "competition_id": comp_id,
                    "year": s.get("year"),
                    "season_label": season_label,
                    "season_start": season_start,
                    "season_end": season_end,
                })

        df = pd.DataFrame(rows)

        if df.empty:
            return empty_df

        df["season_start"] = pd.to_numeric(df.get("season_start"), errors="coerce")
        df["season_end"] = pd.to_numeric(df.get("season_end"), errors="coerce")

        df_valid = df.dropna(subset=["season_start"]).copy()
        # 1) Filtra por temporada inicial >= 2024 (si no hay datos, conserva)
        df_filtrado = df_valid[df_valid["season_start"] >= 2024]
        if not df_filtrado.empty:
            return df_filtrado
        return df_valid if not df_valid.empty else empty_df

   
    except requests.exceptions.HTTPError as e:
        print(f"❌ Error HTTP {resp.status_code}")
        print(resp.text[:400])
    except requests.exceptions.RequestException as e:
        print(f"❌ Error de conexión: {e}")
    return empty_df


def ligas():
    # Devuelve una copia para evitar mutaciones externas sobre el cache.
    return _ligas_cached().copy()


@lru_cache(maxsize=1)
def _league_maps() -> dict:
    df = ligas()
    if df is None or df.empty:
        return {
            "season_to_liga": {},
            "season_to_pais": {},
            "comp_to_liga": {},
            "comp_to_pais": {},
        }
    d = df.rename(columns={"league_name": "Liga", "country": "Pais"}).copy()
    d["season_id"] = pd.to_numeric(d.get("season_id"), errors="coerce")
    d["competition_id"] = pd.to_numeric(d.get("competition_id"), errors="coerce")

    season_df = d.dropna(subset=["season_id"])
    season_to_liga = season_df.set_index("season_id")["Liga"].to_dict()
    season_to_pais = season_df.set_index("season_id")["Pais"].to_dict()

    comp_df = d.dropna(subset=["competition_id"])
    if "season_start" in comp_df.columns:
        comp_df = comp_df.sort_values(
            ["competition_id", "season_start", "season_end"],
            ascending=[True, False, False],
        )
    comp_df = comp_df.drop_duplicates("competition_id")
    comp_to_liga = comp_df.set_index("competition_id")["Liga"].to_dict()
    comp_to_pais = comp_df.set_index("competition_id")["Pais"].to_dict()

    return {
        "season_to_liga": season_to_liga,
        "season_to_pais": season_to_pais,
        "comp_to_liga": comp_to_liga,
        "comp_to_pais": comp_to_pais,
    }


def enrich_liga_cols(
    df: pd.DataFrame,
    competition_id_col: str = "competition_id",
    season_id_col: str = "season_id",
    liga_col: str = "Liga",
    pais_col: str = "Pais",
) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    d = df.copy()
    if liga_col not in d.columns:
        d[liga_col] = pd.NA
    if pais_col not in d.columns:
        d[pais_col] = pd.NA
    # Tratar strings vacíos como NA para permitir fillna en enriquecimiento.
    d[liga_col] = d[liga_col].replace("", pd.NA)
    d[pais_col] = d[pais_col].replace("", pd.NA)

    maps = _league_maps()
    comp_ids = None
    if competition_id_col in d.columns:
        comp_ids = pd.to_numeric(d[competition_id_col], errors="coerce")

    if season_id_col in d.columns:
        season_ids = pd.to_numeric(d[season_id_col], errors="coerce")
        d[liga_col] = d[liga_col].fillna(season_ids.map(maps["season_to_liga"]))
        d[pais_col] = d[pais_col].fillna(season_ids.map(maps["season_to_pais"]))
    if comp_ids is not None:
        # Si competition_id en realidad es season_id (caso común en /todays-matches),
        # intenta mapearlo también contra el catálogo de temporadas.
        d[liga_col] = d[liga_col].fillna(comp_ids.map(maps["season_to_liga"]))
        d[pais_col] = d[pais_col].fillna(comp_ids.map(maps["season_to_pais"]))
        # Fallback final: usa el mapeo por competition_id (league-level)
        d[liga_col] = d[liga_col].fillna(comp_ids.map(maps["comp_to_liga"]))
        d[pais_col] = d[pais_col].fillna(comp_ids.map(maps["comp_to_pais"]))
    return d

if __name__ == "__main__":
    ligas()
