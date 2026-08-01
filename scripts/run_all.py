"""Run every module in order, capture clean logs, and print a summary.

    python scripts/run_all.py              # everything
    python scripts/run_all.py --fast       # skip the two hyperparameter searches
    python scripts/run_all.py --keep-noise # do not strip TensorFlow's banner

Why this exists
---------------
Two of TensorFlow's startup lines cannot be silenced from inside Python:

    WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    I0000 ... port.cc:153] oneDNN custom operations are on. ...

They are written straight to file descriptor 2 by the C++ runtime before absl logging
initialises, so TF_CPP_MIN_LOG_LEVEL cannot gate them and swapping sys.stderr does not
see them (see src/quiet_tf.py). The only reliable fix is to strip them from the captured
stream, which is what this script does before writing outputs/logs/.

Only known-noise patterns are removed. Anything unrecognised is kept, so a real error can
never be hidden — and if a run exits non-zero its output is printed in full regardless.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "outputs" / "logs"
PY = ROOT / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")

# (log name, script path, is_slow)
STEPS: list[tuple[str, str, bool]] = [
    ("ch10_validation", "src/ch10_ksb_gulich/ksb_gulich_python.py", False),
    ("ch05_make_dataset", "src/ch05_compressor/make_dataset.py", False),
    ("ch05_pipeline", "src/ch05_compressor/pipeline.py", False),
    ("ch05_tuner", "src/ch05_compressor/tuner.py", True),
    ("ch06_make_dataset", "src/ch06_esp_head/make_dataset.py", False),
    ("ch06_column_transformer_bug", "src/ch06_esp_head/column_transformer_bug.py", False),
    ("ch06_pipeline", "src/ch06_esp_head/pipeline.py", False),
    ("ch06_tuners", "src/ch06_esp_head/tuners.py", True),
    ("classic_fizzbuzz", "classic_problems/fizzbuzz.py", False),
    ("classic_tests", "classic_problems/test_classic.py", False),
]

# Lines matching any of these are TensorFlow/absl chatter, not program output.
NOISE = [
    re.compile(r"^WARNING: All log messages before absl::InitializeLog\(\)"),
    re.compile(r"^I\d{4} \d{2}:\d{2}:\d+\.\d+\s+\d+\s+\S+\.cc:\d+\]"),
    re.compile(r"oneDNN custom operations are on"),
    re.compile(r"^WARNING:tensorflow:"),
    re.compile(r"^WARNING:absl:"),
    re.compile(r"^Instructions for updating:"),
    re.compile(r"^\s*saveable\.load_own_variables\(store\)\s*$"),
    re.compile(r"TF_ENABLE_ONEDNN_OPTS"),
    re.compile(r"^\s*$\n\Z"),  # trailing blank produced by stripping the above
]


def strip_noise(text: str) -> tuple[str, int]:
    kept, removed = [], 0
    for line in text.splitlines():
        if any(p.search(line) for p in NOISE):
            removed += 1
            continue
        kept.append(line)
    # collapse leading blank lines left behind by removed banners
    while kept and not kept[0].strip():
        kept.pop(0)
    return "\n".join(kept) + ("\n" if kept else ""), removed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="skip the two tuner searches")
    ap.add_argument("--keep-noise", action="store_true",
                    help="write logs exactly as produced, banner included")
    args = ap.parse_args()

    if not PY.exists():
        print(f"venv interpreter not found at {PY}\n"
              f"create it first:  python -m venv .venv && "
              f"{PY.parent}/python -m pip install -r requirements.txt")
        return 1

    LOGS.mkdir(parents=True, exist_ok=True)
    steps = [s for s in STEPS if not (args.fast and s[2])]

    print(f"interpreter: {PY}")
    print(f"running {len(steps)} steps\n")
    print(f"{'step':<30}{'exit':>5}{'secs':>7}{'lines':>7}{'stripped':>10}")
    print("-" * 59)

    failures = []
    for name, rel, _slow in steps:
        started = time.time()
        proc = subprocess.run([str(PY), str(ROOT / rel)], capture_output=True,
                              text=True, encoding="utf-8", errors="replace", cwd=ROOT)
        elapsed = time.time() - started

        combined = (proc.stdout or "") + (proc.stderr or "")
        if args.keep_noise:
            cleaned, removed = combined, 0
        else:
            cleaned, removed = strip_noise(combined)

        (LOGS / f"{name}.log").write_text(cleaned, encoding="utf-8")
        n_lines = len(cleaned.splitlines())
        print(f"{name:<30}{proc.returncode:>5}{elapsed:>7.0f}{n_lines:>7}{removed:>10}")

        if proc.returncode != 0:
            failures.append((name, combined))

    print("-" * 59)
    if failures:
        for name, raw in failures:
            print(f"\n=== {name} FAILED — full unfiltered output ===\n{raw}")
        print(f"{len(failures)} step(s) failed")
        return 1

    print(f"all {len(steps)} steps passed; logs in {LOGS.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
