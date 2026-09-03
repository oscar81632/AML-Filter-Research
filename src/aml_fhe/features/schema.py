"""Feature parquet schema inspection."""

from __future__ import annotations

from pathlib import Path


def read_parquet_schema(path: str | Path) -> list[str]:
    """Return column names for a parquet file or directory."""
    import pyarrow.parquet as pq

    schema = pq.ParquetDataset(Path(path)).schema
    return list(schema.names)


def format_schema(path: str | Path) -> str:
    """Format parquet schema like the reference schema helper."""
    parquet_path = Path(path)
    columns = read_parquet_schema(parquet_path)
    lines = [f"path={parquet_path}", f"columns={len(columns)}"]
    lines.extend(columns)
    return "\n".join(lines)
