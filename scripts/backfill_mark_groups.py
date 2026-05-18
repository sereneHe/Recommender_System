from __future__ import annotations

import os
import subprocess
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    python_cmd = str(root / ".venv" / "bin" / "python3")
    groups = [
        "FRED_9country_monthly",
        "FRED_9country_quarterly",
        "FRED_16country_quarterly",
    ]

    for group in groups:
        env = os.environ.copy()
        env["PYTHON_EXEC"] = python_cmd
        env["PROBLEM_GROUP"] = group
        env["SOLVER_NAME"] = "mark"
        env["EXPERIMENT_NAME"] = "recommender_industry_mark"
        log_path = root / "reports" / f"{group}_mark_run.log"
        print(f"=== Running {group} + mark ===", flush=True)
        with log_path.open("w", encoding="utf-8") as log_file:
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
                "--experiment",
                "recommender_industry_mark",
                "--solver",
                "mark",
                "--output-name",
                f"{group}_mark",
            ],
            cwd=root,
            check=True,
        )
        print(f"=== Finished {group} + mark ===", flush=True)


if __name__ == "__main__":
    main()
