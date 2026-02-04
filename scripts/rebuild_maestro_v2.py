from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from app.common.config import settings
from app.common.db import get_engine
from app.ingest.maestro_schema import reform_maestro


DEFAULT_SOURCE_TABLE = "maestro_picks"
DEFAULT_TARGET_TABLE = "maestro_picks_v2"
DEFAULT_SOURCE_FILE = Path("df_maestro/maestro_picks.pkl")
DEFAULT_TARGET_FILE = Path("df_maestro/maestro_picks_v2.pkl")


def _load_source(source_table: str, source_file: Path) -> pd.DataFrame:
    if settings.database_url:
        engine = get_engine()
        try:
            return pd.read_sql_table(source_table, engine)
        except Exception as exc:
            print(f"Warning: failed to read {source_table} from DB: {exc}")
    if source_file.exists():
        return pd.read_pickle(source_file)
    return pd.DataFrame()


def _write_target(df: pd.DataFrame, target_table: str, target_file: Path, write_pkl: bool) -> None:
    if not settings.database_url:
        raise RuntimeError("DATABASE_URL is not set. Configure Postgres before writing.")
    engine = get_engine()
    df.to_sql(target_table, engine, index=False, if_exists="replace", method="multi", chunksize=1000)
    if write_pkl:
        target_file.parent.mkdir(exist_ok=True)
        df.to_pickle(target_file)


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild maestro table with the v2 schema and load to Postgres.")
    parser.add_argument("--source-table", default=DEFAULT_SOURCE_TABLE)
    parser.add_argument("--target-table", default=DEFAULT_TARGET_TABLE)
    parser.add_argument("--source-file", default=str(DEFAULT_SOURCE_FILE))
    parser.add_argument("--target-file", default=str(DEFAULT_TARGET_FILE))
    parser.add_argument("--skip-pkl", action="store_true", help="Do not write the v2 pickle file.")
    args = parser.parse_args()

    if args.source_table == args.target_table:
        raise ValueError("source-table and target-table must be different.")

    df_raw = _load_source(args.source_table, Path(args.source_file))
    if df_raw.empty:
        raise RuntimeError("No data found in source table or source file.")

    df_v2 = reform_maestro(df_raw)
    _write_target(df_v2, args.target_table, Path(args.target_file), write_pkl=not args.skip_pkl)

    print(f"Rebuilt maestro v2: {len(df_v2)} rows, {len(df_v2.columns)} columns")
    print(f"Target table: {args.target_table}")
    if not args.skip_pkl:
        print(f"Target pickle: {args.target_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
