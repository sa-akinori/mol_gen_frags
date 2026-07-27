"""Measure and record the wall-clock time of a molecule generation run.

The generation entry points (``gen_rffmg.py`` / ``gen_safe.py``) launch their
backend as a subprocess: an externally installed ``t5chem predict`` CLI or one of
the ``generation_*_func.py`` scripts. Since the external CLI cannot be modified,
the only timing both backends share is the elapsed time of the whole generation
process, **including model loading**. That number is written next to the
predictions as a small JSON record so runs can be compared afterwards.
"""

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def count_prediction_rows(output_dir: Path, pattern: str = "predictions*.csv") -> int | None:
    """Count the rows of the predictions CSV(s), i.e. the molecules processed.

    Args:
        output_dir: Directory holding the predictions CSV files.
        pattern: Glob selecting the files to count. A sharded run passes its own
            shard only, e.g. ``predictions_{machine_id}.csv``.

    Returns:
        Total number of data rows across the matched CSVs, or None when no file
        matches the pattern.
    """
    paths = sorted(output_dir.glob(pattern))
    if not paths:
        return None
    return sum(len(pd.read_csv(path, usecols=[0])) for path in paths)


def run_and_record_time(
    cmd: list[str],
    output_dir: Path,
    n_samples: int,
    params: dict[str, Any] | None = None,
    record_name: str = "generation_time.json",
    predictions_pattern: str = "predictions*.csv",
) -> Path:
    """Run a generation command and save its elapsed time as JSON.

    ``subprocess.run(cmd, check=True)`` is wrapped in a ``time.perf_counter()``
    measurement and the record is written to ``output_dir/record_name`` once the
    command succeeds. A failing command propagates its exception and no JSON is
    written, so only completed runs are recorded.

    Args:
        cmd: Command passed to ``subprocess.run``.
        output_dir: Output directory holding the predictions (created if absent).
        n_samples: Number of samples generated per molecule.
        params: Generation parameters to keep for provenance (num_beams,
            model_path, ...).
        record_name: File name of the JSON record.
        predictions_pattern: Glob used to count the processed molecules.

    Returns:
        Path of the written JSON file.
    """
    start = time.perf_counter()
    subprocess.run(cmd, check=True)
    elapsed_sec = round(time.perf_counter() - start, 3)

    n_molecules = count_prediction_rows(output_dir, predictions_pattern)
    record = {
        "elapsed_sec": elapsed_sec,
        "n_molecules": n_molecules,
        "n_samples": n_samples,
        "sec_per_molecule": round(elapsed_sec / n_molecules, 3) if n_molecules else None,
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "params": params or {},
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / record_name
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    return json_path
