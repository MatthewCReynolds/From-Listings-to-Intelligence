"""Rebuild the DuckDB catalog over the local parquet files.

The shipped accolade_synthetic.duckdb hardcodes the generator machine's absolute
paths, so its views resolve to nothing here.
"""

import sys
from pathlib import Path

import duckdb

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "realistic-v2"
OUT = REPO / "data" / "accolade.duckdb"
SCHEMAS = ("market", "pms", "truth")


def main() -> int:
    if not DATA.is_dir():
        sys.exit(f"missing dataset: {DATA}\nrun: python preprocessing/download_data.py")

    OUT.unlink(missing_ok=True)
    con = duckdb.connect(str(OUT))
    n = 0
    for schema in SCHEMAS:
        con.execute(f"create schema if not exists {schema}")
        for table in sorted(p for p in (DATA / schema).iterdir() if p.is_dir()):
            glob = (table / "**" / "*.parquet").as_posix().replace("'", "''")
            con.execute(
                f'create or replace view {schema}."{table.name}" as select * from '
                f"read_parquet('{glob}', hive_partitioning=true, union_by_name=true)"
            )
            n += 1
    con.close()
    print(f"wrote {OUT} ({n} views)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
