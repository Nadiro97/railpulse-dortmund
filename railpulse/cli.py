"""Command-line interface for a local-only data pipeline."""
import argparse
from pathlib import Path
from .config import DEFAULT_MONTH, checked_month, source_filename
from .download import download_month
from .pipeline import ingest_dortmund
from .analytics import build_reports

def main(argv=None):
    parser = argparse.ArgumentParser(description="RailPulse Dortmund ETL")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("download", "extract", "analyze", "run"):
        command = sub.add_parser(name)
        command.add_argument("--month", default=DEFAULT_MONTH, help="YYYY-MM")
        command.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args(argv)
    month = checked_month(args.month)
    source = args.data_root / "raw" / source_filename(month)
    processed = args.data_root / "processed" / f"dortmund_{month}.parquet"
    if args.command in ("download", "run"):
        source = download_month(month, args.data_root)
        print(f"Downloaded: {source}")
    if args.command in ("extract", "run"):
        processed, metadata = ingest_dortmund(source, args.data_root)
        print(f"Extracted {metadata['dortmund_rows']:,} rows: {processed}")
    if args.command in ("analyze", "run"):
        for path in build_reports(processed, Path("reports")):
            print(f"Report: {path}")

if __name__ == "__main__":
    main()
