"""CPU controls for parity artifacts. Run directly, with unittest, or pytest."""
import copy
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR))
import cluster_a2_tp_layerwise as lap

SCRIPT = TESTS_DIR / "cluster_a2_tp_layerwise.py"
NUM_LAYERS, HIDDEN, VOCAB, SEQ, TP = 4, 8, 32, 5, 4
CHECKPOINT = "/models/SYNTHETIC-under-test"
PROMPTS = ["alpha prompt", "beta prompt"]
LAYER_TYPES = ["linear_attention"] * 3 + ["full_attention"]


def artifacts():
    """Clone distinct capture streams so each alignment has a positive control."""
    engine_results, hf_results = [], []
    for i, prompt in enumerate(PROMPTS):
        generator = torch.Generator().manual_seed(100 + i)
        streams = {
            key: [torch.randn(SEQ, HIDDEN, generator=generator) for _ in range(count)]
            for key, count in (("layer_states", NUM_LAYERS + 1),
                               ("attn_only", NUM_LAYERS), ("post_ln", NUM_LAYERS))
        }
        logits = torch.randn(VOCAB, generator=generator) * 0.1
        logits[7] = 10.0
        common = {"prompt": prompt, "prompt_ids": [11 + i, 22, 33, 44, 55]}
        engine_results.append({**common, **streams, "state_slot": i, "logits": logits})
        hf_results.append({
            **copy.deepcopy(common),
            "layer_out": [t.clone() for t in streams["layer_states"][1:]],
            "hidden": [t.clone() for t in streams["layer_states"]],
            "mix_out": [t.clone() for t in streams["attn_only"]],
            "mix_in": [t.clone() for t in streams["post_ln"]],
            "logits": logits.clone(),
        })
    common = {
        "format_version": lap.FORMAT_VERSION, "checkpoint": CHECKPOINT,
        "prompts": PROMPTS.copy(), "hidden_size": HIDDEN, "num_layers": NUM_LAYERS,
    }
    engine = {
        **common, "artifact_kind": "layerwise_engine", "tp": TP,
        "dtype": "bfloat16", "layer_types": LAYER_TYPES.copy(), "results": engine_results,
    }
    hf = {**copy.deepcopy(common), "artifact_kind": "layerwise_hf", "results": hf_results}
    return engine, hf


class LayerwiseParityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="lap_neg_")
        self.addCleanup(temporary.cleanup)
        self.cache = Path(temporary.name)

    def write(self, engine, hf, tp=TP):
        torch.save(engine, lap._engine_path(str(self.cache), tp))
        torch.save(hf, lap._hf_path(str(self.cache)))

    def compare(self, tp=TP, cache=None, optimized=False):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1")
        command = [sys.executable] + (["-O"] if optimized else [])
        return subprocess.run(
            command + [str(SCRIPT), "--phase", "compare", "--tp", str(tp),
                       "--cache-dir", str(cache or self.cache),
                       "--csv-dir", str(self.cache / "out")],
            cwd=TESTS_DIR.parent, env=env, capture_output=True, text=True, timeout=600,
        )

    def rejected(self, process, message=None):
        self.assertNotEqual(process.returncode, 0, process.stdout + process.stderr)
        if message:
            self.assertIn(message, process.stdout + process.stderr)

    def test_positive_control(self):
        for tp in (2, 4):
            with self.subTest(tp=tp):
                engine, hf = artifacts()
                engine["tp"] = tp
                # HF's final hidden state is post-norm, not the raw residual.
                for result in hf["results"]:
                    result["hidden"][-1].mul_(10)
                self.write(engine, hf, tp=tp)
                result = self.compare(tp=tp)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("VERDICT", result.stdout)

    def test_invalid_artifacts(self):
        # Names map directly to old negative tests in the refactoring report.
        cases = [
            ("prompt_token_id_mismatch", 0, ("results", 0, "prompt_ids", 2), 999),
            ("hidden_shape_mismatch", 0, ("results", 0, "layer_states", 1),
             torch.zeros(SEQ, HIDDEN - 1)),
            ("token_count_mismatch", 0, ("results", 0, "layer_states", 2),
             torch.zeros(SEQ + 3, HIDDEN)),
            ("different_checkpoint", 0, ("checkpoint",), "/models/OTHER"),
            ("format_version_mismatch", 0, ("format_version",), lap.FORMAT_VERSION + 99),
            ("tp_mismatch", 0, ("tp",), 2),
            ("layer_count_mismatch", 1, ("num_layers",), NUM_LAYERS + 1),
            ("incomplete_layer_hook_list", 1, ("results", 1, "layer_out"), []),
            ("none_entry_in_layer_hook", 1, ("results", 0, "mix_in", 2), None),
            ("prompt_list_reordered", 1, ("prompts",), list(reversed(PROMPTS))),
            ("prompt_text_mismatch", 0, ("results", 0, "prompt"), "wrong prompt"),
            ("artifact_kind_mismatch", 1, ("artifact_kind",), "foreign"),
            ("hidden_size_metadata", 1, ("hidden_size",), HIDDEN + 1),
            ("empty_token_ids", 0, ("results", 0, "prompt_ids"), []),
        ]
        for name, side, path, value in cases:
            with self.subTest(name=name):
                pair = artifacts()
                target = pair[side]
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                self.write(*pair)
                self.rejected(self.compare())
        for boundary in ("empty_results", "shared_wrong_shape", "shared_wrong_prompt"):
            with self.subTest(boundary=boundary):
                engine, hf = artifacts()
                if boundary == "empty_results":
                    engine["results"], hf["results"] = [], []
                    engine["prompts"], hf["prompts"] = [], []
                elif boundary == "shared_wrong_shape":
                    engine["results"][0]["layer_states"][0] = torch.zeros(SEQ, HIDDEN - 1)
                    hf["results"][0]["hidden"][0] = torch.zeros(SEQ, HIDDEN - 1)
                else:
                    engine["results"][0]["prompt"] = hf["results"][0]["prompt"] = "wrong"
                self.write(engine, hf)
                self.rejected(self.compare(), "REFUSING")
        engine, hf = artifacts()
        del hf["results"][0]["layer_out"]
        self.write(engine, hf)
        self.rejected(self.compare(), "missing")

    def test_nonfinite_all_streams(self):
        for side, names in (
            (0, ("layer_states", "attn_only", "post_ln", "logits")),
            (1, ("hidden", "layer_out", "mix_out", "mix_in", "logits")),
        ):
            for name in names:
                for value in (float("nan"), float("inf")):
                    with self.subTest(side=side, stream=name, value=value):
                        pair = artifacts()
                        tensor = pair[side]["results"][0][name]
                        if name != "logits":
                            tensor = tensor[0]
                        tensor.reshape(-1)[0] = value
                        self.write(*pair)
                        self.rejected(self.compare(), "non-finite values")
                        self.assertFalse((self.cache / "out" / "layerwise_results.csv").exists())

    def test_final_logits_rules(self):
        for rule in ("top1", "cosine"):
            with self.subTest(rule=rule):
                engine, hf = artifacts()
                e, h = engine["results"][0]["logits"], hf["results"][0]["logits"]
                if rule == "top1":
                    # Near-tied runners isolate Top-1 failure at cosine > 0.99.
                    h[3], e[3] = 9.99, 10.01
                    self.assertGreater(lap._metrics(e, h)["cosine"], 0.99)
                    self.assertNotEqual(int(e.argmax()), int(h.argmax()))
                else:
                    noise = torch.randn(VOCAB, generator=torch.Generator().manual_seed(5)) * 2
                    noise[7] = 0
                    e.add_(noise)
                    self.assertLess(lap._metrics(e, h)["cosine"], 0.99)
                    self.assertEqual(int(e.argmax()), int(h.argmax()))
                self.write(engine, hf)
                self.rejected(self.compare(), "VERDICT")

    def test_missing_and_corrupt_files(self):
        self.rejected(self.compare(cache=self.cache / "missing"), "missing")
        for corrupt in (False, True):
            with self.subTest(corrupt=corrupt):
                self.write(*artifacts())
                path = Path(lap._hf_path(str(self.cache)))
                if corrupt:
                    path.write_bytes(b"not a torch file at all")
                else:
                    path.unlink()
                self.rejected(self.compare())

    def test_metrics_identical_high_dim_tensor_is_one(self):
        x = (torch.randn(248320, generator=torch.Generator().manual_seed(7)) * 3).bfloat16().float()
        m = lap._metrics(x, x.clone())
        self.assertAlmostEqual(m["cosine"], 1.0, delta=1e-12)
        for key in ("max_abs_err", "mean_abs_err", "rel_l2_err", "nan_count", "inf_count"):
            self.assertEqual(m[key], 0)
        self.assertEqual(m["norm_ratio"], 1.0)

    def test_metrics_known_values(self):
        cases = [
            ([6., 8.], [3., 4.], 1., 2., 4., 3.5, 1.),
            ([1., 0.], [0., 1.], 0., 1., 1., 1., 2 ** 0.5),
            ([1., 2., 3.], [1., 2., 4.], None, None, 1., 1 / 3, 1 / 21 ** 0.5),
        ]
        keys = ("cosine", "norm_ratio", "max_abs_err", "mean_abs_err", "rel_l2_err")
        for a, b, *expected in cases:
            with self.subTest(a=a, b=b):
                metrics = lap._metrics(torch.tensor(a), torch.tensor(b))
                for key, value in zip(keys, expected):
                    if value is not None:
                        self.assertAlmostEqual(metrics[key], value, delta=1e-12)

    def test_metrics_counts_nan_and_inf(self):
        m = lap._metrics(torch.tensor([1., float("nan"), float("inf")]), torch.ones(3))
        self.assertEqual((m["nan_count"], m["inf_count"]), (1, 1))

    def test_metrics_shape_and_optimized_cli(self):
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            lap._metrics(torch.zeros(4), torch.zeros(5))
        engine, hf = artifacts()
        engine["results"][0]["layer_states"][1] = torch.zeros(SEQ, HIDDEN - 1)
        self.write(engine, hf)
        self.rejected(self.compare(optimized=True), "shape mismatch")

    def test_topk_metrics(self):
        ref = torch.zeros(VOCAB)
        ref[:5] = torch.tensor([9., 8., 7., 6., 5.])
        swapped = ref.clone()
        swapped[:2] = torch.tensor([8., 9.])
        disjoint = torch.zeros(VOCAB)
        disjoint[10:15] = ref[:5]
        for engine, expected in ((ref, (1, 1, 5, 1)), (swapped, (0, 1, 5, 1)),
                                 (disjoint, (0, 0, 0, 0))):
            with self.subTest(expected=expected):
                metrics = lap._topk_metrics(ref, engine)
                self.assertEqual(tuple(metrics[k] for k in
                    ("top1_match", "top5_set_match", "top5_overlap", "ref_top1_in_eng_top5")),
                    expected)
                self.assertEqual(metrics["top1_margin"], 1.)

    def test_require_finite(self):
        lap._require_finite("clean", torch.zeros(4))
        with self.assertRaises(SystemExit):
            lap._require_finite("nan", torch.tensor([1., float("nan")]))

    def test_cli_and_independence(self):
        self.assertTrue(lap.DEFAULT_CACHE_DIR.endswith(os.path.join("tests", "_phase_b_cache")))
        self.assertEqual(lap._engine_path("/tmp/x", 4), "/tmp/x/layerwise_engine_tp4.pt")
        self.assertEqual(lap._hf_path("/tmp/x"), "/tmp/x/layerwise_hf.pt")
        self.assertEqual(lap.FINAL_LOGITS_COSINE_THRESHOLD, 0.99)
        source = SCRIPT.read_text()
        for forbidden in ("batch_invariance", "_cluster_day_cache", "_phase6_cache", "subprocess"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
