from __future__ import annotations

import os
import subprocess
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def problem_groups(problem_root: Path) -> list[str]:
    return sorted(path.name for path in problem_root.iterdir() if path.is_dir())


def main() -> None:
    root = repo_root()
    problem_root = root / "experiments_conf" / "problem"
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    python_exec = root / ".venv" / "bin" / "python3"
    python_cmd = str(python_exec) if python_exec.exists() else "python3"

    for group in problem_groups(problem_root):
        print(f"=== Running {group} ===", flush=True)
        env = os.environ.copy()
        env["PROBLEM_GROUP"] = group
        env["PYTHON_EXEC"] = python_cmd
        run_log = reports_dir / f"{group}_run.log"
        with run_log.open("w", encoding="utf-8") as log_file:
            subprocess.run(
                ["sh", "./scripts/experiments_recommender_industry.sh"],
                cwd=root,
                env=env,
                check=True,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
        subprocess.run(
            [
                python_cmd,
                "scripts/multirun_summary_mlflow.py",
                "--group",
                group,
            ],
            cwd=root,
            check=True,
        )
        print(f"=== Finished {group} ===", flush=True)


if __name__ == "__main__":
    main()
