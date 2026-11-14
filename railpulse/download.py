"""Download a single reviewed monthly Parquet release from Hugging Face."""
from pathlib import Path
from .config import DATASET_REPO, source_filename

def download_month(month: str, data_root: Path = Path("data")) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("Install dependencies: pip install -e .") from exc
    filename = source_filename(month)
    raw_root = Path(data_root) / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    return Path(hf_hub_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        filename=filename,
        local_dir=str(raw_root),
    ))
