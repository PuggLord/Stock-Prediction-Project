from __future__ import annotations

from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parent
DOWNLOADS_DIR = ROOT / "downloads"
PACKAGE_DIR = DOWNLOADS_DIR / "pi5_package"
ZIP_PATH = DOWNLOADS_DIR / "pi5_package.zip"

FILES_TO_INCLUDE = [
    "README.md",
    "PI5_QUICKSTART.md",
    "requirements.txt",
    "run_signal.py",
    "src/__init__.py",
    "src/data.py",
    "src/features.py",
    "src/train.py",
    "src/signal.py",
    "src/paper_trade.py",
    "src/utils.py",
    "models/logistic_regression.joblib",
    "models/train_metadata.json",
]


def _copy_file(rel_path: str) -> None:
    source = ROOT / rel_path
    if not source.exists():
        raise FileNotFoundError(f"Missing required file for Pi5 bundle: {source}")

    destination = PACKAGE_DIR / rel_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def build_pi5_bundle() -> Path:
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)

    if PACKAGE_DIR.exists():
        shutil.rmtree(PACKAGE_DIR)
    PACKAGE_DIR.mkdir(parents=True, exist_ok=True)

    for rel_path in FILES_TO_INCLUDE:
        _copy_file(rel_path)

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for file_path in PACKAGE_DIR.rglob("*"):
            if file_path.is_file():
                zip_file.write(file_path, arcname=file_path.relative_to(PACKAGE_DIR))

    return ZIP_PATH


def main() -> None:
    zip_path = build_pi5_bundle()
    print(f"Created Pi5 bundle at: {zip_path}")


if __name__ == "__main__":
    main()
