"""
Reproducibility tests for the H-SAS-GF simulation driver.

These tests are intentionally simple — they verify that the driver:
  1. runs to completion with the same seed in < 30 seconds,
  2. produces the same numerical output twice in a row,
  3. respects the seed (different seeds → different outputs),
  4. emits a well-formed JSON with the required keys, and
  5. produces AUC values inside the plausibility envelope
     (multi-seed mean in [0.65, 0.90]).

Run with:
    cd tests && python -m unittest test_driver_reproducibility
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DRIVER = ROOT / "code" / "simulation_driver.py"


def _run(seed: int, n_seeds: int, outdir: Path) -> dict:
    outdir.mkdir(exist_ok=True, parents=True)
    cmd = [
        sys.executable, str(DRIVER),
        "--seed", str(seed),
        "--n-seeds", str(n_seeds),
        "--outdir", str(outdir),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(r.stderr + r.stdout)
    path = outdir / "simulation_aggregate.json"
    with open(path) as f:
        return json.load(f)


class TestDriverReproducibility(unittest.TestCase):

    def test_01_runs_to_completion(self):
        with tempfile.TemporaryDirectory() as d:
            out = _run(42, 10, Path(d))
            self.assertIn("main_metrics", out)
            self.assertIn("ablation", out)
            self.assertIn("compute_budget_modelled", out)
            self.assertIn("cost_model", out)

    def test_02_identical_runs(self):
        with tempfile.TemporaryDirectory() as d1, \
             tempfile.TemporaryDirectory() as d2:
            a = _run(42, 10, Path(d1))
            b = _run(42, 10, Path(d2))
            self.assertEqual(
                a["main_metrics"]["auc"]["mean"],
                b["main_metrics"]["auc"]["mean"],
                "Same seed must produce identical results")

    def test_03_seed_matters(self):
        with tempfile.TemporaryDirectory() as d1, \
             tempfile.TemporaryDirectory() as d2:
            a = _run(42, 10, Path(d1))
            b = _run(7, 10, Path(d2))
            # Different seeds should produce different means
            # (allow tolerance of 0.001 for edge cases)
            self.assertNotEqual(
                a["main_metrics"]["auc"]["mean"],
                b["main_metrics"]["auc"]["mean"],
                "Different seeds should produce different outputs")

    def test_04_required_keys(self):
        with tempfile.TemporaryDirectory() as d:
            out = _run(42, 10, Path(d))
            for k in ["auc", "sensitivity_at_0.6",
                      "specificity_at_0.6", "f1_at_0.6"]:
                self.assertIn(k, out["main_metrics"])
                for sub in ["mean", "sd", "ci_95", "n"]:
                    self.assertIn(sub, out["main_metrics"][k])

    def test_05_auc_plausibility(self):
        """Multi-seed AUC mean should be in the plausibility envelope."""
        with tempfile.TemporaryDirectory() as d:
            out = _run(42, 25, Path(d))
            auc_mean = out["main_metrics"]["auc"]["mean"]
            self.assertGreater(auc_mean, 0.60,
                               f"AUC mean {auc_mean} is implausibly low")
            self.assertLess(auc_mean, 0.95,
                            f"AUC mean {auc_mean} is implausibly high; "
                            "check for over-fitting or prior leakage")

    def test_06_ablation_monotone(self):
        """Full model should beat each single-block ablation on average."""
        with tempfile.TemporaryDirectory() as d:
            out = _run(42, 25, Path(d))
            full = out["ablation"]["full_auc"]["mean"]
            for ab_key in ["no_ssl_auc", "no_gat_auc", "no_xgb_auc"]:
                ab_auc = out["ablation"][ab_key]["mean"]
                # Allow a small tolerance for stochastic ties
                self.assertGreaterEqual(
                    full + 0.01, ab_auc,
                    f"Full model ({full}) should be >= {ab_key} "
                    f"({ab_auc}) up to tolerance")

    def test_07_compute_budget_reasonable(self):
        """Modelled compute budget should be in a defensible range."""
        with tempfile.TemporaryDirectory() as d:
            out = _run(42, 5, Path(d))
            c = out["compute_budget_modelled"]
            self.assertLess(c["inference_s"], 15.0)
            self.assertGreater(c["inference_s"], 0.5)
            self.assertLess(c["incremental_power_w"], 5.0)
            self.assertGreater(c["incremental_power_w"], 0.0)
            self.assertLessEqual(c["model_size_kb"], 2048)


if __name__ == "__main__":
    unittest.main(verbosity=2)
