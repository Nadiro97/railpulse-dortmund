"""Command-line interface for the local railway ETL and validation pipeline."""
import argparse
from pathlib import Path
from .config import DEFAULT_MONTH, checked_month, source_filename
from .download import download_month
from .pipeline import ingest_dortmund
from .analytics import build_reports

def main(argv=None):
    parser = argparse.ArgumentParser(description="RailPulse Dortmund ETL")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("download", "extract", "analyze", "run", "validate-updates", "ml-baseline"):
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
    if args.command in ("validate-updates", "ml-baseline"):
        from .observed_updates import change_filename, download_change_month, validate_month
        change = args.data_root / "raw" / change_filename(month)
        if not processed.is_file():
            raise FileNotFoundError(
                f"Missing Dortmund processed file: {processed}. "
                "Run 'python -m railpulse download' and 'python -m railpulse extract' first."
            )
        if not change.is_file():
            change = download_change_month(month, args.data_root)
        if args.command == "validate-updates":
            paths, metadata = validate_month(month, processed, change)
            print(f"Validated explicit updates for {metadata['processed_dortmund_rows']:,} Dortmund rows")
            for path in paths:
                print(f"Aggregate: {path}")
        else:
            from .ml_baseline import run_baseline
            report = run_baseline(month, processed, change)
            print(f"ML baseline train_n={report['train_n']}, test_n={report['test_n']}")
            print(f"Test ROC AUC: {report['logistic_regression']['roc_auc']}")
            print("Metrics: reports/ml_baseline_metrics.json")

if __name__ == "__main__":
    main()
