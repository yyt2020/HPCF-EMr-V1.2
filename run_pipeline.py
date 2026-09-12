"""Run the complete HPCF example workflow in the required stage order."""

import configparser
from pathlib import Path
import subprocess
import sys
import time


STAGES = (
    "01_fas_corner_frequency.py",
    "02_em_score.py",
    "03_aic_hpcf.py",
)


def resolve_config_path(root: Path, value: str) -> Path:
    """Resolve a path from config.ini without changing the stage calculations."""
    path = Path(value.split()[0])
    return path if path.is_absolute() else root / path


def check_required_inputs(root: Path) -> None:
    """Fail early when restricted, user-supplied inputs have not been added."""
    config_path = root / "config.ini"
    config = configparser.ConfigParser()
    config.read(config_path, encoding="utf-8")

    record_dir = resolve_config_path(root, config["Paths"]["in_dir"])
    metadata_file = resolve_config_path(root, config["Paths"]["station_metadata"])
    missing = []

    if not metadata_file.is_file():
        missing.append(f"metadata CSV: {metadata_file}")
    if not record_dir.is_dir() or not any(record_dir.glob("*.dat")):
        missing.append(f"acceleration records (*.dat): {record_dir}")

    if missing:
        details = "\n  - ".join(missing)
        raise SystemExit(
            "Required user-supplied inputs are missing:\n"
            f"  - {details}\n\n"
            "Restricted CENC waveform files are intentionally excluded from this "
            "repository. See input/README.md for access and placement instructions."
        )


def main() -> None:
    root = Path(__file__).resolve().parent
    check_required_inputs(root)
    start = time.time()

    for index, script_name in enumerate(STAGES, start=1):
        print(f"\n[{index}/{len(STAGES)}] Running {script_name}", flush=True)
        subprocess.run(
            [sys.executable, str(root / script_name)],
            cwd=root,
            check=True,
        )

    print(f"\nPipeline completed in {time.time() - start:.2f} s.")
    print(f"Results are available under {root / 'OUTPUT'}")


if __name__ == "__main__":
    main()
