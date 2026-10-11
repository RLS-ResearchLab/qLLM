"""Negative / self-check tests for tests/cluster_a2_tp_layerwise.py.

WHY THESE EXIST
===============
A numerical parity harness that reports PASS is only worth something if it can
actually detect a wrong answer. These tests mutate SYNTHETIC copies of the
artifact format in a pytest `tmp_path` and assert the comparator refuses them.
The real captures under tests/_phase_b_cache/ are never read or written.

CI-FRIENDLY BY DESIGN
=====================
The comparator is invoked as a SUBPROCESS and only its exit status is
asserted:

  * a correctly-rejected input  -> subprocess exit != 0 -> test PASSES
  * an incorrectly-accepted one -> subprocess exit == 0 -> test FAILS
  * a control case with fully valid, identical artifacts -> exit == 0

So a green test run means "the tool rejected what it should reject", not
"the tool crashed". Nothing here needs a GPU, a checkpoint, or a model: every
call is `--phase compare`, which is CPU-only and reads nothing but the two
artifacts. The synthetic tensors are deliberately tiny (4 layers, hidden 8,
vocab 32) to keep the suite fast.

Run (either works -- this repo does not declare pytest as a dependency, so
the file also carries a tiny built-in runner in __main__):

    python tests/test_cluster_a2_tp_layerwise_negatives.py      # no deps
    python -m pytest tests/test_cluster_a2_tp_layerwise_negatives.py -v
"""
import inspect
import os
import pathlib
import subprocess
import sys
import tempfile
import traceback

import torch

try:
    import pytest
except ImportError:  # pragma: no cover - exercised in this container
    # pytest is not installed and is not in requirements.txt. Rather than add
    # a dependency to run a test, provide the one piece of the API used here.
    class _RaisesCtx:
        def __init__(self, exc, match=None):
            self.exc, self.match = exc, match

        def __enter__(self):
            return self

        def __exit__(self, et, ev, tb):
            if et is None:
                raise AssertionError(f"expected {self.exc.__name__}, nothing was raised")
            if not issubclass(et, self.exc):
                return False          # let the real exception surface
            if self.match and self.match not in str(ev):
                raise AssertionError(
                    f"expected {self.exc.__name__} matching {self.match!r}, got {ev!r}")
            return True               # swallow: the raise was expected

    class _PytestShim:
        @staticmethod
        def raises(exc, match=None):
            return _RaisesCtx(exc, match)

    pytest = _PytestShim()

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TESTS_DIR)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

import cluster_a2_tp_layerwise as lap  # noqa: E402

SCRIPT = os.path.join(TESTS_DIR, "cluster_a2_tp_layerwise.py")

# Synthetic model shape -- small on purpose.
NUM_LAYERS = 4
HIDDEN = 8
VOCAB = 32
SEQ = 5
TP = 4
CHECKPOINT = "/models/SYNTHETIC-under-test"
LAYER_TYPES = ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
PROMPTS = ["alpha prompt", "beta prompt"]


# ---------------------------------------------------------------------------
# synthetic artifact builders
# ---------------------------------------------------------------------------
def _logits(seed: int) -> torch.Tensor:
    """A logits row with a clear, deterministic winner (and therefore a clear
    Top-1 margin) so the 'Top-1 must match' logic is exercised meaningfully."""
    g = torch.Generator().manual_seed(seed)
    t = torch.randn(VOCAB, generator=g) * 0.1
    t[7] = 10.0
    return t


def _engine_result(pi: int) -> dict:
    g = torch.Generator().manual_seed(100 + pi)
    return {
        "prompt": PROMPTS[pi],
        "prompt_ids": [11 + pi, 22 + pi, 33 + pi, 44 + pi, 55 + pi],
        "state_slot": pi,
        "layer_states": [torch.randn(SEQ, HIDDEN, generator=g) for _ in range(NUM_LAYERS + 1)],
        "attn_only": [torch.randn(SEQ, HIDDEN, generator=g) for _ in range(NUM_LAYERS)],
        "post_ln": [torch.randn(SEQ, HIDDEN, generator=g) for _ in range(NUM_LAYERS)],
        "logits": _logits(pi),
    }


def _hf_result(pi: int, engine_result: dict) -> dict:
    """Built FROM the engine result so the baseline is an exact match.

    The three HF capture streams map onto three DIFFERENT engine streams, and
    they must not be conflated:
        hf.layer_out[i] <-> engine.layer_states[i+1]   (post-layer residual)
        hf.mix_out[i]   <-> engine.attn_only[i]        (mixer output)
        hf.mix_in[i]    <-> engine.post_ln[i]          (mixer input / post-LN)
        hf.hidden[0]    <-> engine.layer_states[0]     (embedding)
    """
    num = len(engine_result["attn_only"])
    return {
        "prompt": PROMPTS[pi],
        "prompt_ids": list(engine_result["prompt_ids"]),
        "layer_out": [engine_result["layer_states"][i + 1].clone() for i in range(num)],
        "hidden": ([engine_result["layer_states"][0].clone()]
                   + [engine_result["layer_states"][i + 1].clone() for i in range(num)]),
        "mix_out": [t.clone() for t in engine_result["attn_only"]],
        "mix_in": [t.clone() for t in engine_result["post_ln"]],
        "logits": engine_result["logits"].clone(),
    }


def _engine_payload(tp: int = TP, checkpoint: str = CHECKPOINT,
                    prompts=None, layer_types=None, **overrides) -> dict:
    prompts = PROMPTS if prompts is None else prompts
    layer_types = LAYER_TYPES if layer_types is None else layer_types
    results = [(_engine_result(i)) for i in range(len(prompts))]
    if "results" in overrides:
        results = overrides.pop("results")
    payload = {
        "artifact_kind": "layerwise_engine",
        "format_version": lap.FORMAT_VERSION,
        "tp": tp,
        "dtype": "bfloat16",
        "checkpoint": checkpoint,
        "prompts": list(prompts),
        "layer_types": list(layer_types),
        "hidden_size": HIDDEN,
        "num_layers": len(layer_types),
        "gpu_memory_utilization": 0.60,
        "enforce_eager": True,
        "results": results,
    }
    payload.update(overrides)
    return payload


def _hf_payload(checkpoint: str = CHECKPOINT, prompts=None, layer_types=None,
                **overrides) -> dict:
    prompts = PROMPTS if prompts is None else prompts
    layer_types = LAYER_TYPES if layer_types is None else layer_types
    eng = _engine_payload(prompts=prompts, layer_types=layer_types)
    results = [_hf_result(i, eng["results"][i]) for i in range(len(prompts))]
    if "results" in overrides:
        results = overrides.pop("results")
    payload = {
        "artifact_kind": "layerwise_hf",
        "format_version": lap.FORMAT_VERSION,
        "checkpoint": checkpoint,
        "prompts": list(prompts),
        "mixer_names": ["linear_attn"] * len(layer_types),
        "num_layers": len(layer_types),
        "hidden_size": HIDDEN,
        "results": results,
    }
    payload.update(overrides)
    return payload


def _write(tmp_path, engine=None, hf=None, engine_tp=TP) -> str:
    """Writes the two artifacts into tmp_path and returns the cache dir."""
    cache = str(tmp_path)
    torch.save(_engine_payload() if engine is None else engine,
               os.path.join(cache, f"layerwise_engine_tp{engine_tp}.pt"))
    torch.save(_hf_payload() if hf is None else hf,
               os.path.join(cache, "layerwise_hf.pt"))
    return cache


def _run_compare(cache_dir: str, tp: int = TP):
    """Runs --phase compare as a subprocess. Returns CompletedProcess."""
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""     # compare is CPU-only; don't touch GPUs
    env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, SCRIPT, "--phase", "compare", "--tp", str(tp),
         "--cache-dir", cache_dir, "--csv-dir", os.path.join(cache_dir, "out")],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=600,
    )


def _assert_rejected(proc, why: str) -> None:
    assert proc.returncode != 0, (
        f"EXIT-CODE BUG: the comparator returned {proc.returncode} (success) for "
        f"an input it must reject -- {why}\n"
        f"--- stdout ---\n{proc.stdout[-3000:]}\n--- stderr ---\n{proc.stderr[-3000:]}")


def _assert_accepted(proc, why: str) -> None:
    assert proc.returncode == 0, (
        f"FALSE NEGATIVE: the comparator returned {proc.returncode} for a valid "
        f"input -- {why}\n"
        f"--- stdout ---\n{proc.stdout[-3000:]}\n--- stderr ---\n{proc.stderr[-3000:]}")


# ===========================================================================
# POSITIVE CONTROL -- without this, a comparator that rejects everything
# unconditionally would satisfy every negative test below.
# ===========================================================================
def test_positive_control_valid_artifacts_are_accepted(tmp_path):
    cache = _write(tmp_path)
    proc = _run_compare(cache)
    _assert_accepted(proc, "identical synthetic engine/HF artifacts")
    assert "VERDICT" in proc.stdout


# ===========================================================================
# NEGATIVE CASES
# ===========================================================================
def test_negative_prompt_token_id_mismatch(tmp_path):
    eng = _engine_payload()
    eng["results"][0]["prompt_ids"] = list(eng["results"][0]["prompt_ids"])
    eng["results"][0]["prompt_ids"][2] = 999
    _assert_rejected(_run_compare(_write(tmp_path, engine=eng)),
                     "one prompt token ID was changed")


def test_negative_hidden_shape_mismatch(tmp_path):
    eng = _engine_payload()
    # One layer's captured width is wrong (e.g. an un-reduced TP shard).
    eng["results"][0]["layer_states"][1] = torch.zeros(SEQ, HIDDEN - 1)
    _assert_rejected(_run_compare(_write(tmp_path, engine=eng)),
                     "a layer_states tensor has the wrong hidden width")


def test_negative_token_count_mismatch(tmp_path):
    eng = _engine_payload()
    eng["results"][0]["layer_states"][2] = torch.zeros(SEQ + 3, HIDDEN)
    _assert_rejected(_run_compare(_write(tmp_path, engine=eng)),
                     "a layer_states tensor has the wrong sequence length")


def test_negative_nan_in_layer_tensor(tmp_path):
    eng = _engine_payload()
    eng["results"][0]["layer_states"][2][0, 0] = float("nan")
    _assert_rejected(_run_compare(_write(tmp_path, engine=eng)),
                     "NaN injected into an intermediate layer tensor")


def test_negative_inf_in_logits(tmp_path):
    eng = _engine_payload()
    eng["results"][1]["logits"][3] = float("inf")
    _assert_rejected(_run_compare(_write(tmp_path, engine=eng)),
                     "Inf injected into the final logits")


def test_negative_top1_mismatch(tmp_path):
    eng = _engine_payload()
    # Reference winner is index 7. Nudge a runner-up past it: Top-1 flips while
    # cosine stays ~1.0, so this isolates the Top-1 rule.
    eng["results"][0]["logits"][3] = 10.5
    eng["results"][0]["logits"][7] = 9.5
    _assert_rejected(_run_compare(_write(tmp_path, engine=eng)),
                     "engine Top-1 differs from HF Top-1 (cosine still > 0.99)")


def test_negative_cosine_below_threshold(tmp_path):
    eng = _engine_payload()
    g = torch.Generator().manual_seed(5)
    # Preserve the winner (index 7) but add enough broadband noise to push the
    # final-logits cosine under the author's 0.99 bar. Non-mutating: the sum
    # rebinds the payload entry to a new tensor, so nothing is aliased.
    noise = torch.randn(VOCAB, generator=g) * 2.0
    noise[7] = 0.0
    eng["results"][0]["logits"] = eng["results"][0]["logits"] + noise
    ref = _logits(0)
    cos = float(torch.nn.functional.cosine_similarity(
        eng["results"][0]["logits"].reshape(1, -1).double(),
        ref.reshape(1, -1).double()))
    assert cos < lap.FINAL_LOGITS_COSINE_THRESHOLD, (
        f"test construction error: cosine {cos} is not below the bar")
    assert int(eng["results"][0]["logits"].argmax()) == 7, (
        "test construction error: the winner must be preserved so this case "
        "isolates the cosine rule")
    _assert_rejected(_run_compare(_write(tmp_path, engine=eng)),
                     f"final-logits cosine {cos:.5f} < {lap.FINAL_LOGITS_COSINE_THRESHOLD}")


def test_negative_different_checkpoint(tmp_path):
    eng = _engine_payload(checkpoint="/models/A-DIFFERENT-MODEL")
    _assert_rejected(_run_compare(_write(tmp_path, engine=eng)),
                     "engine artifact came from a different checkpoint than hf")


def test_negative_format_version_mismatch(tmp_path):
    eng = _engine_payload(format_version=lap.FORMAT_VERSION + 99)
    _assert_rejected(_run_compare(_write(tmp_path, engine=eng)),
                     "engine artifact has a stale/unknown format_version")


def test_negative_tp_mismatch(tmp_path):
    # File is named for tp=2 (so the path lookup succeeds) but records tp=4.
    eng = _engine_payload(tp=4)
    _assert_rejected(_run_compare(_write(tmp_path, engine=eng, engine_tp=2), tp=2),
                     "artifact records tp=4 but compare was asked for tp=2")


def test_negative_layer_count_mismatch(tmp_path):
    hf = _hf_payload()
    hf["num_layers"] = NUM_LAYERS + 1
    _assert_rejected(_run_compare(_write(tmp_path, hf=hf)),
                     "hf artifact reports a different layer count")


def test_negative_missing_layer_hook_data(tmp_path):
    hf = _hf_payload()
    del hf["results"][0]["layer_out"]     # hook never fired / pre-hook artifact
    _assert_rejected(_run_compare(_write(tmp_path, hf=hf)),
                     "hf artifact has no authoritative layer_out capture")


def test_negative_incomplete_layer_hook_list(tmp_path):
    hf = _hf_payload()
    hf["results"][1]["layer_out"] = hf["results"][1]["layer_out"][:-1]
    _assert_rejected(_run_compare(_write(tmp_path, hf=hf)),
                     "hf layer_out list is shorter than num_layers (silent skip)")


def test_negative_none_entry_in_layer_hook(tmp_path):
    hf = _hf_payload()
    hf["results"][0]["mix_in"][2] = None
    _assert_rejected(_run_compare(_write(tmp_path, hf=hf)),
                     "an hf per-layer capture slot is None")


def test_negative_missing_artifact_file(tmp_path):
    _write(tmp_path)
    os.remove(os.path.join(str(tmp_path), "layerwise_hf.pt"))
    _assert_rejected(_run_compare(str(tmp_path)),
                     "the hf artifact file is missing entirely")


def test_negative_corrupt_artifact(tmp_path):
    cache = _write(tmp_path)
    with open(os.path.join(cache, "layerwise_hf.pt"), "wb") as f:
        f.write(b"not a torch file at all")
    proc = _run_compare(cache)
    assert proc.returncode != 0, (
        "a corrupt/truncated artifact must not be silently accepted; "
        f"got exit {proc.returncode}")


def test_negative_prompt_list_reordered(tmp_path):
    # Same prompts, different ORDER: zip() would pair unrelated prompts.
    hf = _hf_payload(prompts=list(reversed(PROMPTS)))
    _assert_rejected(_run_compare(_write(tmp_path, hf=hf)),
                     "the two artifacts list the prompts in different orders")


# ===========================================================================
# IN-PROCESS UNIT TESTS for the metric helpers
# ===========================================================================
def test_metrics_identical_high_dim_tensor_is_one():
    """Requirement: identical high-dimensional tensors must score ~1.0 within
    floating-point tolerance. 248320 == the real checkpoint's vocab size, the
    exact case where float32 reduction was measured to lose ~2.8e-5."""
    g = torch.Generator().manual_seed(7)
    x = (torch.randn(248320, generator=g) * 3.0).to(torch.bfloat16).float()
    m = lap._metrics(x, x.clone())
    assert abs(m["cosine"] - 1.0) < 1e-12, f"cosine {m['cosine']} should be exactly 1"
    assert m["max_abs_err"] == 0.0
    assert m["mean_abs_err"] == 0.0
    assert m["rel_l2_err"] == 0.0
    assert m["norm_ratio"] == 1.0
    assert m["nan_count"] == 0 and m["inf_count"] == 0


def test_metrics_known_values():
    # a = 2 * b: perfectly parallel, so cosine is 1 while every magnitude
    # metric still differs. This is the case cosine alone cannot see.
    m = lap._metrics(torch.tensor([6.0, 8.0]), torch.tensor([3.0, 4.0]))
    assert abs(m["cosine"] - 1.0) < 1e-12
    assert m["norm_ratio"] == 2.0                     # ||[6,8]|| / ||[3,4]|| = 10/5
    assert m["max_abs_err"] == 4.0                    # diff = [3,4]
    assert m["mean_abs_err"] == 3.5                   # (3 + 4) / 2
    assert abs(m["rel_l2_err"] - 1.0) < 1e-12         # ||[3,4]|| / ||[3,4]||

    # Orthogonal unit vectors: cosine 0, identical norms.
    orth = lap._metrics(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]))
    assert abs(orth["cosine"]) < 1e-12
    assert orth["norm_ratio"] == 1.0
    assert orth["max_abs_err"] == 1.0
    assert abs(orth["rel_l2_err"] - 2 ** 0.5) < 1e-12  # ||[1,-1]|| / ||[0,1]||

    # A small perturbation: rel_l2 must track the perturbation, not the scale.
    c = lap._metrics(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 2.0, 4.0]))
    assert c["max_abs_err"] == 1.0
    assert abs(c["mean_abs_err"] - 1.0 / 3.0) < 1e-12
    assert abs(c["rel_l2_err"] - 1.0 / (21 ** 0.5)) < 1e-12


def test_metrics_counts_nan_and_inf():
    m = lap._metrics(torch.tensor([1.0, float("nan"), float("inf")]),
                     torch.tensor([1.0, 2.0, 3.0]))
    assert m["nan_count"] == 1 and m["inf_count"] == 1


def test_metrics_raises_on_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        lap._metrics(torch.zeros(4), torch.zeros(5))


def test_metrics_raises_on_shape_mismatch_even_with_O_flag():
    """Guards the `assert`-stripped-under-`python -O` failure mode: the check
    must be a real raise, not an assert."""
    src = open(SCRIPT).read()
    body = src.split("def _metrics", 1)[1].split("def ", 1)[0]
    assert "shape mismatch for comparison" in body
    assert "assert a.shape == b.shape" not in src, (
        "shape validation must not rely on `assert` (stripped under -O)")


def test_topk_metrics_exact_match():
    ref = torch.zeros(VOCAB)
    for i, v in enumerate([9.0, 8.0, 7.0, 6.0, 5.0]):
        ref[i] = v
    tk = lap._topk_metrics(ref, ref.clone(), k=5)
    assert tk["top1_match"] == 1
    assert tk["top5_set_match"] == 1
    assert tk["top5_overlap"] == 5
    assert tk["ref_top1_in_eng_top5"] == 1
    assert abs(tk["top1_margin"] - 1.0) < 1e-6        # 9.0 - 8.0


def test_topk_metrics_detects_disjoint_top1():
    ref = torch.zeros(VOCAB)
    ref[0] = 9.0
    for i, v in enumerate([8.0, 7.0, 6.0, 5.0, 4.0], start=10):
        ref[i] = v
    eng = torch.zeros(VOCAB)
    eng[1] = 9.0                                      # different winner
    for i, v in enumerate([8.0, 7.0, 6.0, 5.0, 4.0], start=20):
        eng[i] = v
    tk = lap._topk_metrics(ref, eng, k=5)
    assert tk["top1_match"] == 0
    assert tk["top5_overlap"] == 0
    assert tk["top5_set_match"] == 0
    assert tk["ref_top1_in_eng_top5"] == 0


def test_topk_metrics_set_match_ignores_order():
    """Top-5 SET equality must be order-insensitive."""
    ref = torch.zeros(VOCAB)
    for i, v in enumerate([9.0, 8.0, 7.0, 6.0, 5.0]):
        ref[i] = v
    eng = ref.clone()
    # Swap ranks 1 and 2 through an explicit temporary. Writing
    # `eng[0], eng[1] = eng[1], eng[0]` would be WRONG: those are live 0-dim
    # views, so the first assignment mutates position 0 before the second
    # reads it, collapsing both entries to the same value.
    first = eng[0].clone()
    eng[0] = eng[1]
    eng[1] = first
    assert eng[0].item() == 8.0 and eng[1].item() == 9.0, "swap construction is wrong"
    tk = lap._topk_metrics(ref, eng, k=5)
    assert tk["top1_match"] == 0                      # argmax changed
    assert tk["top5_set_match"] == 1                  # ... but the set did not
    assert tk["top5_overlap"] == 5


def test_require_finite_raises_on_nan():
    with pytest.raises(SystemExit):
        lap._require_finite("unit-test", torch.tensor([1.0, float("nan")]))


def test_require_finite_accepts_clean_tensor():
    lap._require_finite("unit-test", torch.zeros(4))


# ===========================================================================
# CLI contract
# ===========================================================================
def test_cache_dir_defaults_to_phase_b_cache():
    """--cache-dir must default to the historical location so existing
    invocations keep working, but must be overridable."""
    assert lap.DEFAULT_CACHE_DIR.endswith(os.path.join("tests", "_phase_b_cache"))
    assert lap._engine_path("/tmp/x", 4) == os.path.join("/tmp/x", "layerwise_engine_tp4.pt")
    assert lap._hf_path("/tmp/x") == os.path.join("/tmp/x", "layerwise_hf.pt")


def test_missing_cache_dir_is_rejected(tmp_path):
    _assert_rejected(_run_compare(str(tmp_path / "does-not-exist")),
                     "the cache directory does not exist at all")


def test_exported_threshold_unchanged():
    """The author's acceptance bar must not drift."""
    assert lap.FINAL_LOGITS_COSINE_THRESHOLD == 0.99


def test_finite_gate_is_wired_into_compare():
    """NaN/Inf is rejected twice over (verdict flag AND a fail-fast gate), so
    the NaN subprocess test alone cannot detect the gate's call site being
    removed -- the verdict path still returns non-zero. Pin the wiring
    directly, because losing fail-fast changes WHICH layer gets reported as
    'worst' (NaN compares False against everything, so min() picks arbitrarily)
    even though the overall PASS/FAIL outcome happens to stay correct."""
    src = open(SCRIPT).read()
    body = src.split("def phase_compare", 1)[1]
    assert "_require_finite(" in body, (
        "the NaN/Inf fail-fast gate must be called inside phase_compare before "
        "any metric is computed")
    for stream in ("layer_states", "attn_only", "post_ln", "hidden", "layer_out",
                   "mix_out", "mix_in"):
        assert f'"{stream}"' in body, f"finite gate does not cover the {stream} stream"


def test_module_does_not_import_pr2_batch_invariance():
    """PR 1 must stand alone and must not depend on the PR 2 script, on any
    pre-existing cache, or on outputs/ helper scripts."""
    src = open(SCRIPT).read()
    assert "batch_invariance" not in src, (
        "PR 1 must not depend on the PR 2 batch-invariance script")
    # The only cache it may read is its own; another test's golden data must
    # never be a hidden input.
    assert "_cluster_day_cache" not in src, "PR 1 must not read the A2 golden cache"
    assert "_phase6_cache" not in src, "PR 1 must not read the phase6 cache"
    assert "subprocess" not in src, "PR 1 must not shell out to other scripts"


# ===========================================================================
# Built-in runner (used when pytest is unavailable -- see the header).
# pytest discovers the test_* functions above and ignores this block.
# ===========================================================================
def _run_all() -> int:
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    failures = []
    for name, fn in fns:
        kwargs = {}
        if "tmp_path" in inspect.signature(fn).parameters:
            kwargs["tmp_path"] = pathlib.Path(tempfile.mkdtemp(prefix="lap_neg_"))
        try:
            fn(**kwargs)
            print(f"PASS  {name}")
        except Exception:                                   # noqa: BLE001
            failures.append(name)
            print(f"FAIL  {name}")
            traceback.print_exc()
    print("\n" + "=" * 78)
    print(f"negative-test suite: {len(fns) - len(failures)}/{len(fns)} passed")
    if failures:
        print("FAILED: " + ", ".join(failures))
    print("=" * 78)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_run_all())
