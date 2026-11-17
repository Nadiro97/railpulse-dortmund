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
    for name in ("download", "extract", "analyze", "run", "validate-updates", "ml-baseline", "ml-improve", "ml-cross-month", "ml-multimonth", "train-demo"):
        command = sub.add_parser(name)
        command.add_argument("--month", default=DEFAULT_MONTH, help="YYYY-MM")
        command.add_argument("--data-root", type=Path, default=Path("data"))
    cross = sub.choices["ml-cross-month"]
    cross.add_argument("--test-month", default="2026-10", help="Unseen next month YYYY-MM")
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
    if args.command == "train-demo":
        from .demo_model import train_local_model, MODEL_PATH
        artifact = train_local_model(args.data_root, MODEL_PATH)
        print(f"Local-only model saved: {MODEL_PATH}")
        print(f"Trained on {artifact['n_train']:,} archived September 1–16 records.")
        print("Timetable updates only; not confirmed actual train arrivals.")
        return
    if args.command == "ml-multimonth":
        from .ml_multimonth import run_multi_month
        report = run_multi_month(args.data_root)
        print(f"Training: {', '.join(report['training_months'])}")
        print(f"Validation: {report['validation_month']}; benchmark: {report['benchmark_month']} (previously inspected)")
        print(f"Proposed candidate: {report['chosen_candidate']} at {report['candidate_threshold']}")
        print("Metrics: reports/ml_multimonth_metrics.json")
        return
    if args.command == "ml-cross-month":
        from .ml_cross_month import run_cross_month
        from .download import download_month
        for selected in (month, checked_month(args.test_month)):
            monthly_source = args.data_root / "raw" / source_filename(selected)
            monthly_processed = args.data_root / "processed" / f"dortmund_{selected}.parquet"
            if not monthly_processed.is_file():
                if not monthly_source.is_file():
                    monthly_source = download_month(selected, args.data_root)
                ingest_dortmund(monthly_source, args.data_root)
        results = run_cross_month(month, args.test_month, args.data_root)
        report = results["unseen_month_holdout"]["selected_model"]
        print(f"Unseen {args.test_month} records: {results['unseen_month_holdout']['sample_n']:,}")
        print(f"ROC AUC: {report['ranking']['roc_auc']}; recall at locked 0.225: {report['locked_alert_threshold']['recall']}")
        print("Metrics: reports/ml_cross_month_metrics.json")
        return
    if args.command in ("validate-updates", "ml-baseline", "ml-improve"):
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
        elif args.command == "ml-baseline":
            from .ml_baseline import run_baseline
            report = run_baseline(month, processed, change)
            print(f"ML baseline train_n={report['train_n']}, test_n={report['test_n']}")
            print(f"Test ROC AUC: {report['logistic_regression']['roc_auc']}")
            print("Metrics: reports/ml_baseline_metrics.json")
        else:
            from .ml_improved import run_improved
            result = run_improved(month, processed, change)
            print(f"Selected model: {result['selected_model']}")
            print(f"Validation-selected threshold: {result['selected_threshold']}")
            print("Metrics: reports/ml_improved_metrics.json")

if __name__ == "__main__":
    main()
