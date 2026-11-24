"""Smoke-test the one wheel in a local build directory, independently of its version."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, default=Path("dist"))
    parser.add_argument("--ui", action="store_true")
    parser.add_argument("--offline", action="store_true", help="Use cached dependencies only.")
    args = parser.parse_args()
    wheels = sorted(args.dist.glob("personal_harness-*.whl"))
    if len(wheels) != 1:
        parser.error("Use a build directory containing exactly one Personal Harness wheel.")
    wheel = str(wheels[0].resolve())
    smoke = "smoke_ui_install.py" if args.ui else "smoke_install.py"
    if args.ui:
        wheel += "[ui]"
    script = Path(__file__).with_name(smoke).resolve()
    env = {key: value for key, value in os.environ.items() if key != "VIRTUAL_ENV"}
    if args.offline:
        env["UV_OFFLINE"] = "1"
    return subprocess.call(
        ["uv", "run", "--isolated", "--no-project", "--with", wheel, "python", str(script)],
        env=env,
    )


if __name__ == "__main__":
    raise SystemExit(main())
