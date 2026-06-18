import json
import tempfile
import unittest
from pathlib import Path

from tools.run_counter_prior_pilot import (
    DEFAULT_CONFIG,
    build_run_plan,
    load_and_validate_config,
)
from tools.summarize_counter_prior_pilot import summarize


class CounterPriorPilotTest(unittest.TestCase):
    def test_default_plan_has_36_seed_matched_runs(self):
        config = load_and_validate_config(DEFAULT_CONFIG)
        seeds = [42, 123, 456, 789, 1024, 2026]
        runs = build_run_plan(config, seeds, Path("/tmp/pilot"), {"steps": 50})
        self.assertEqual(len(runs), 36)
        for pair_id in ("car", "duck", "kite"):
            pair_runs = [run for run in runs if run["pair_id"] == pair_id]
            self.assertEqual(len(pair_runs), 12)
            for seed in seeds:
                self.assertEqual(sum(int(run["seed"]) == seed for run in pair_runs), 2)

    def test_duplicate_prompt_id_is_rejected(self):
        config = load_and_validate_config(DEFAULT_CONFIG)
        config["pairs"][1]["prompts"][0]["prompt_id"] = config["pairs"][0]["prompts"][0]["prompt_id"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate prompt_id"):
                load_and_validate_config(path)

    def test_gate_uses_only_duck_and_kite_counter_prior_delta(self):
        rows = []
        for pair_id in ("duck", "kite"):
            for condition, failures in (("in_prior", 1), ("counter_prior", 3)):
                for index in range(6):
                    failed = index < failures
                    rows.append(
                        {
                            "pair_id": pair_id,
                            "condition": condition,
                            "target_attribute_correct": str(not failed),
                            "object_identity_preserved": "true",
                            "secondary_attribute_correct": "true",
                        }
                    )
        result = summarize(rows)
        self.assertTrue(result["continuation_gate"]["passed"])


if __name__ == "__main__":
    unittest.main()
