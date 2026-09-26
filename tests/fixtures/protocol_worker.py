"""Real worker protocol, with optimization replaced before the worker runs."""
import asyncio
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
# Install the same no-research-DB/no-external-network guard in this child.
import conftest  # noqa: F401, E402
from scripts import optimize, wfo_worker  # noqa: E402


async def synthetic_optimization(**kwargs):
    assert kwargs["strategy_name"] == "envelope_dca"
    assert kwargs["symbol"] == "BTC/USDT"
    assert kwargs["params_override"]["ma_period"] == [7]
    kwargs["progress_callback"](20., "synthetic IS")
    kwargs["progress_callback"](80., "synthetic OOS")
    return None, 12345


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--db-path", required=True)
    args = parser.parse_args()
    optimize.run_optimization = synthetic_optimization
    raise SystemExit(asyncio.run(wfo_worker._run(args.job_id, args.db_path, "unused")))
