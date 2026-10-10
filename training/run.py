"""Run original notebook-derived training stages, with explicit outputs/input paths."""
from __future__ import annotations
import argparse, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGES = {
    "event-gather": ("event_classifier", "01_data_gathering.py"),
    "event-preprocess": ("event_classifier", "02_preprocessing.py"),
    "event-train": ("event_classifier", "04_finbert_training_real_only.py"),
    "sentiment-prepare": ("sentiment", "01_fiqa_data_preparation.py"),
    "sentiment-train": ("sentiment", "02_finbert_regression_training.py"),
    "event-synthetic-optional": ("event_classifier", "03_synthetic_generation_optional.py"),
}

def main() -> int:
    p = argparse.ArgumentParser(description="Run RiskPulse training sources without changing app/models")
    p.add_argument("stage", choices=[*STAGES, "check"])
    p.add_argument("--workdir", type=Path, default=None, help="Output dir, also searched for inputs")
    p.add_argument("--input-dir", type=Path, help="Alternative directory containing prepared split files")
    a = p.parse_args()
    if a.stage == "check":
        for _, (folder, filename) in STAGES.items():
            compile((ROOT/folder/"scripts"/filename).read_text(encoding="utf-8"), filename, "exec")
        print("All source files parsed successfully")
        return 0
    folder, filename = STAGES[a.stage]
    script = ROOT / folder / "scripts" / filename
    workdir = (a.workdir or ROOT.parent / "training_runs" / folder).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env["RISK_PULSE_WORKDIR"] = str(workdir)
    env["RISK_PULSE_INPUT_DIR"] = str((a.input_dir or workdir).resolve())
    print("Running", a.stage, "with inputs", env["RISK_PULSE_INPUT_DIR"], "and outputs", workdir, flush=True)
    return subprocess.call([sys.executable, str(script)], env=env, cwd=ROOT.parent)

if __name__ == "__main__":
    raise SystemExit(main())
