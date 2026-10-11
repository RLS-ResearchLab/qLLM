"""TP-aware LAYERWISE numerical parity against an independent HF reference.

WHY THIS IS A NEW FILE
======================
tests/reference_check_phase6_layerwise.py could not be reused as-is: its
checkpoint path, `tensor_parallel_size=2`, `gpu_memory_utilization=0.9` and its
single prompt are all hardcoded, and its `--phase compare` is built around one
historical bug-hunt (a `--first bad layer` bisect report whose thresholds
assume the -0.109 anti-correlated failure it was written to diagnose). Editing
it in place would change a script whose historical outputs are part of this
project's evidence trail. So this is a purpose-built harness that REUSES what
is reusable:

  * the ENGINE-side capture is ModelRunner.get_prefill_layer_states()
    (engine/model_runner.py:799) -- called exactly as before, NOT modified;
  * the HF-side mixer-identification rule is a local constant here
    (_SKIP_CHILD_NAMES) rather than an import, so this harness carries no
    import-time coupling to that historical script; the rule is unchanged;
  * the HF hook shape (with_kwargs=True, hidden_states read from kwargs) is
    the same trick that file documents, because HF calls the mixer with
    hidden_states as a KEYWORD argument.

The comparison is semantically the same one the original made (engine
residual-stream + attention-only + post-input_layernorm checkpoints vs. HF's
output_hidden_states + per-layer/mixer hooks). What changed is that the
checkpoint, TP size and prompts are configurable instead of hardcoded, the
prompt set is the 5-prompt set cluster_a2_tp_correctness.py uses, and the
numeric battery is wider: cosine, max/mean abs err, relative L2, norm ratio,
NaN/Inf count, shape and layer type, plus -- for the final logits -- Top-1
exact match, Top-5 SET match, Top-5 overlap, whether HF's Top-1 lands inside
the engine's Top-5, and HF's own Top-1-vs-Top-2 margin.

LAST-TOKEN vs FULL-SEQUENCE
===========================
Both sides are captured as FULL SEQUENCE (seq_len, hidden). The primary
comparison is the LAST TOKEN row -- the only row that feeds
final-norm -> lm_head, and so the only one that can influence next-token
prediction. A secondary comparison over the whole (flattened) sequence is
reported too, labelled `full_sequence`, because a full-sequence statistic can
hide a perfect last row (or vice versa). The two are never mixed, and the CSV
always says which one a row is.

THRESHOLDS -- stated BEFORE the numbers, not after
=================================================
The author's A2 golden bar (cluster_a2_tp_correctness.py:181) is unchanged
and remains the acceptance criterion:
    FINAL PREFILL LOGITS: cosine >= 0.99 AND Top-1 argmax matches HF.
Intermediate layer checkpoints have NO pre-existing author-set threshold:
this script RECORDS them and reports the worst observed layer without
declaring PASS/FAIL, so no threshold was invented after seeing results.

Three separate process invocations (HF and the engine cannot coexist in GPU
memory -- the same constraint as reference_check_phase6*.py):

    python tests/cluster_a2_tp_layerwise.py --phase engine --tp 4 \\
        --checkpoint /models/Qwen3.5-35B-A3B --gpu-memory-utilization 0.60
    python tests/cluster_a2_tp_layerwise.py --phase hf \\
        --checkpoint /models/Qwen3.5-35B-A3B
    python tests/cluster_a2_tp_layerwise.py --phase compare --tp 4   # CPU only

Artifacts default to tests/_phase_b_cache/layerwise_{engine_tp<N>|hf}.pt.
Pass --cache-dir to use a fresh directory -- a regression run should not
silently overwrite a previous capture.

STALE-CACHE SAFETY
==================
--phase compare never trusts that the two artifacts it reads belong to the
same run. Before computing a single metric it checks, on both files: the
artifact kind, a FORMAT_VERSION, the checkpoint path, the full ordered PROMPT
list, the layer count, the hidden size, and the completeness of every
per-layer capture list. A mismatch is refused with a non-zero exit instead of
being reported as a PASS/FAIL. Without this, a stale engine artifact from one
checkpoint could be compared against a fresh HF artifact from another and
produce a convincing-looking verdict that means nothing.

EXIT STATUS
===========
  0  matrix printed and the author's final-logits bar met for every prompt
  1  refused (mismatched/incomplete artifacts, non-finite tensors, token-ID
     mismatch) or the bar was not met
"""
import argparse
import csv
import os
import sys
import types

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if "nanovllm" not in sys.modules:
    nanovllm_pkg = types.ModuleType("nanovllm")
    nanovllm_pkg.__path__ = [ROOT]
    nanovllm_pkg.__file__ = os.path.join(ROOT, "__init__.py")
    sys.modules["nanovllm"] = nanovllm_pkg

# Names of a Qwen3.5 decoder layer's NON-mixer children. The mixer is whatever
# child is left, and it is looked up at runtime rather than hardcoded because
# this checkpoint mixes `linear_attn` (GDN) and `self_attn` (full attention)
# depending on layer index.
#
# Deliberately a local constant rather than an import from
# reference_check_phase6_layerwise.py: importing that module for a three-element
# set would drag in its import-time sys.path/module-shim side effects and couple
# this harness to a historical bug-hunt script. The value is identical to
# reference_check_phase6_layerwise.py:66 and the identification rule below is
# unchanged; if an HF checkpoint's decoder layer ever gains a fourth named
# child, both files need the same update.
_SKIP_CHILD_NAMES = {"input_layernorm", "post_attention_layernorm", "mlp"}

DEFAULT_CKPT = os.path.join(ROOT, "qwen35_checkpoint")
DEFAULT_CACHE_DIR = os.path.join(ROOT, "tests", "_phase_b_cache")
MAX_MODEL_LEN = 2048

# Artifact schema version. Bump this whenever the shape or the SEMANTICS of a
# cached artifact changes. --phase compare refuses to read a mismatched
# version rather than silently comparing stale/mismatched captures (see
# _validate_artifacts).
FORMAT_VERSION = 2

# Keys every cached artifact must have. Checked on load so a truncated,
# hand-edited, or foreign .pt fails with a clear message instead of a
# KeyError deep inside the comparison loop.
_ENGINE_KEYS = ("artifact_kind", "format_version", "tp", "dtype", "checkpoint",
                "prompts", "layer_types", "hidden_size", "num_layers", "results")
_HF_KEYS = ("artifact_kind", "format_version", "checkpoint", "prompts",
            "num_layers", "hidden_size", "results")
# Per-prompt keys inside "results".
_ENGINE_RESULT_KEYS = ("prompt", "prompt_ids", "state_slot", "layer_states",
                       "attn_only", "post_ln", "logits")
_HF_RESULT_KEYS = ("prompt", "prompt_ids", "layer_out", "hidden", "mix_out",
                   "mix_in", "logits")

# Same varied set cluster_a2_tp_correctness.py:173 uses -- deliberately reused
# so this layerwise trace is directly comparable to that already-established
# golden reference run.
PROMPTS = [
    "The capital of France is",
    "If John has 3 apples and buys 5 more, he has",
    "2 + 2 =",
    "In a small village, there lived a farmer who had",
    "The quick brown fox jumps over the lazy dog. This sentence is famous because",
]

# The author's FINAL-PREFILL-LOGITS bar (cluster_a2_tp_correctness.py:181).
FINAL_LOGITS_COSINE_THRESHOLD = 0.99


def _engine_path(cache_dir: str, tp: int) -> str:
    return os.path.join(cache_dir, f"layerwise_engine_tp{tp}.pt")


def _hf_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, "layerwise_hf.pt")


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def _metrics(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Full numeric battery for two tensors that are ALREADY known to be
    semantically comparable (same layer, same token position, same meaning).
    Callers are responsible for never passing a sharded tensor against a
    gathered one or a last-row against a full-sequence one.

    COMPUTED IN FLOAT64 ON PURPOSE. Measured on this checkpoint (2026-10-10):
    two BIT-IDENTICAL 248320-dim logit tensors score only 0.999972403 under
    torch.nn.functional.cosine_similarity, which reduces in float32; the same
    pair scores 0.9999999999999994 when reduced in float64. So a float32
    cosine carries ~2.8e-5 of pure reduction noise at this vocab size -- small
    next to the author's 0.99 bar, but large enough to visibly flatten genuine
    differences at the layer level, where most real cosines sit > 0.99.
    Errors are likewise accumulated in float64."""
    a = a.detach().reshape(-1).double()
    b = b.detach().reshape(-1).double()
    # A real raise, not an assert: `assert` is stripped under `python -O`, and
    # a shape mismatch slipping through here would mean cosine() of two
    # structurally different tensors.
    if a.shape != b.shape:
        raise ValueError(
            f"shape mismatch for comparison: engine {tuple(a.shape)} vs "
            f"reference {tuple(b.shape)} -- these are not the same checkpoint "
            f"of the same layer")
    diff = (a - b).abs()
    na, nb = torch.linalg.norm(a), torch.linalg.norm(b)
    return {
        "shape_ref": tuple(b.shape),
        "shape_eng": tuple(a.shape),
        "cosine": (a @ b / torch.clamp(na * nb, min=1e-300)).item(),
        "max_abs_err": diff.max().item(),
        "mean_abs_err": diff.mean().item(),
        "rel_l2_err": (torch.linalg.norm(a - b) / torch.clamp(nb, min=1e-300)).item(),
        "norm_ratio": (na / torch.clamp(nb, min=1e-300)).item(),
        "nan_count": int(torch.isnan(a).sum().item()) + int(torch.isnan(b).sum().item()),
        "inf_count": int(torch.isinf(a).sum().item()) + int(torch.isinf(b).sum().item()),
        "ref_absmax": b.abs().max().item(),
    }


def _atomic_torch_save(obj, path: str) -> None:
    """Same reasoning as cluster_a2_tp_correctness.py's helper: readers only
    test os.path.exists, so never leave a truncated file behind."""
    tmp_path = path + ".tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def _free_diagnostic_seq(llm, seq) -> None:
    """Mirrors cluster_a2_tp_correctness.py's _free_diagnostic_seq: these
    manually-built diagnostic Sequences never go through Scheduler.schedule()
    /postprocess(), so nobody else frees their KV block or GDN state slot.
    Mandatory here because this script allocates one Sequence PER PROMPT --
    without freeing, StateManager.free_slot_ids empties and allocate() raises
    `IndexError: pop from an empty deque` (the exact bug cluster_a2's helper
    docstring records)."""
    llm.scheduler.block_manager.deallocate(seq)
    if llm.model_runner.state_manager is not None:
        llm.model_runner.call("free_state_slot", seq)


# --------------------------------------------------------------------------
# artifact validation
#
# --phase engine / --phase hf / --phase compare run as three SEPARATE
# invocations, so the compare step has no way to know whether the two files
# it reads came from the same run unless the artifacts say so themselves.
# Without these checks a stale engine_tp4.pt from one checkpoint could be
# silently compared against a fresh layerwise_hf.pt from another, producing a
# defensible-looking PASS/FAIL that means nothing.
# --------------------------------------------------------------------------
def _fail(msg: str) -> None:
    """Refuse to continue. SystemExit(str) prints to stderr and exits 1 --
    a real non-zero status, not a printed 'FAIL' that still returns 0."""
    raise SystemExit(f"[layerwise] REFUSING: {msg}")


def _require_finite(label: str, t: torch.Tensor) -> None:
    """Hard NaN/Inf gate, applied BEFORE any metric is computed.

    Deliberately fail-fast rather than just accumulating a flag: with NaNs
    present every derived number (cosine, max_abs, norm_ratio) is NaN too, and
    the downstream `min(local_cos, key=...)` worst-layer selection then
    compares NaN against NaN -- which is always False -- and silently reports
    an arbitrary layer as "worst". Refusing up front keeps the summary honest.
    """
    t = t.detach()
    if torch.isfinite(t).all():
        return
    n_nan = int(torch.isnan(t).sum().item())
    n_inf = int(torch.isinf(t).sum().item())
    _fail(f"non-finite values in {label}: {n_nan} NaN, {n_inf} Inf "
          f"(shape {tuple(t.shape)}, dtype {t.dtype}). Every metric derived "
          f"from this tensor would be NaN; a comparison is meaningless.")


def _require_keys(payload, keys, path: str) -> None:
    if not isinstance(payload, dict):
        _fail(f"{path} did not load as a dict (got {type(payload).__name__}) -- "
              f"not a layerwise artifact.")
    missing = [k for k in keys if k not in payload]
    if missing:
        _fail(f"{path} is missing required key(s) {missing}. It is incomplete, "
              f"hand-edited, or from an older/incompatible version.")


def _validate_artifacts(eng: dict, hf: dict, eng_path: str, hf_path: str, args) -> int:
    """Cross-checks both artifacts agree on everything that would make the
    comparison meaningless if it differed. Returns the shared layer count."""
    _require_keys(eng, _ENGINE_KEYS, eng_path)
    _require_keys(hf, _HF_KEYS, hf_path)

    if eng["artifact_kind"] != "layerwise_engine":
        _fail(f"{eng_path} has artifact_kind={eng['artifact_kind']!r}, expected "
              f"'layerwise_engine'.")
    if hf["artifact_kind"] != "layerwise_hf":
        _fail(f"{hf_path} has artifact_kind={hf['artifact_kind']!r}, expected "
              f"'layerwise_hf'.")

    for label, p, path in (("engine", eng, eng_path), ("hf", hf, hf_path)):
        if p["format_version"] != FORMAT_VERSION:
            _fail(f"{path} has format_version={p['format_version']!r}, this script "
                  f"writes/expects {FORMAT_VERSION}. Re-run --phase "
                  f"{'engine' if label == 'engine' else 'hf'}.")

    # Model identity: a mismatched checkpoint makes every number meaningless.
    if eng["checkpoint"] != hf["checkpoint"]:
        _fail(f"checkpoint mismatch between the two captures:\n"
              f"    engine : {eng['checkpoint']}\n"
              f"    hf     : {hf['checkpoint']}\n"
              f"These were captured from DIFFERENT models; refusing to compare.")

    if eng["tp"] != args.tp:
        _fail(f"{eng_path} was captured with tp={eng['tp']} but --tp {args.tp} was "
              f"requested. Pass --tp {eng['tp']} or re-run --phase engine --tp {args.tp}.")

    # Both keys are mandatory (_ENGINE_KEYS / _HF_KEYS), so no .get() fallback
    # here: a defaulted comparison would silently compare the engine against
    # itself and pass.
    if eng["hidden_size"] != hf["hidden_size"]:
        _fail(f"hidden_size mismatch: engine={eng['hidden_size']} "
              f"hf={hf['hidden_size']} -- the two captures are not of the same "
              f"model.")

    # Prompt set identity, including ORDER: the compare zips the two result
    # lists positionally, so a reordered list would compare prompt A against
    # prompt B. The per-prompt text assert further down cannot catch that on
    # its own if both files were regenerated with a different PROMPTS order.
    if list(eng["prompts"]) != list(hf["prompts"]):
        _fail(f"prompt list mismatch between the two captures:\n"
              f"    engine : {list(eng['prompts'])}\n"
              f"    hf     : {list(hf['prompts'])}")

    num_layers = len(eng["layer_types"])
    if eng["num_layers"] != num_layers:
        _fail(f"{eng_path} is internally inconsistent: num_layers={eng['num_layers']} "
              f"but {num_layers} layer_types.")
    if hf["num_layers"] != num_layers:
        _fail(f"layer count mismatch: engine={num_layers} hf={hf['num_layers']}. "
              f"The two captures are not of the same model.")

    if len(eng["results"]) != len(hf["results"]):
        _fail(f"result count mismatch: engine={len(eng['results'])} "
              f"hf={len(hf['results'])}.")
    if len(eng["results"]) != len(eng["prompts"]):
        _fail(f"{eng_path} has {len(eng['results'])} results for "
              f"{len(eng['prompts'])} prompts.")

    for i, (er, hr) in enumerate(zip(eng["results"], hf["results"]), start=1):
        for k in _ENGINE_RESULT_KEYS:
            if k not in er:
                _fail(f"engine result #{i} is missing key {k!r}.")
        for k in _HF_RESULT_KEYS:
            if k not in hr:
                _fail(f"hf result #{i} is missing key {k!r}.")
        # Per-layer completeness: a hook that silently failed to fire would
        # leave a short list or a None entry, and must be refused rather than
        # skipped.
        for name, seq, expected in (("layer_states", er["layer_states"], num_layers + 1),
                                    ("attn_only", er["attn_only"], num_layers),
                                    ("post_ln", er["post_ln"], num_layers)):
            if len(seq) != expected:
                _fail(f"engine result #{i}: {name} has {len(seq)} entries, "
                      f"expected {expected}.")
            if any(t is None for t in seq):
                _fail(f"engine result #{i}: {name} contains None entry(ies) -- a "
                      f"capture step did not run.")
        for name, seq, expected in (("hidden", hr["hidden"], num_layers + 1),
                                    ("layer_out", hr["layer_out"], num_layers),
                                    ("mix_out", hr["mix_out"], num_layers),
                                    ("mix_in", hr["mix_in"], num_layers)):
            if len(seq) != expected:
                _fail(f"hf result #{i}: {name} has {len(seq)} entries, expected "
                      f"{expected}. The HF capture is incomplete or predates the "
                      f"decoder-layer hooks.")
            if any(t is None for t in seq):
                _fail(f"hf result #{i}: {name} contains None entry(ies) -- a forward "
                      f"hook never fired for some layer.")
    return num_layers


# --------------------------------------------------------------------------
# phase engine
# --------------------------------------------------------------------------
def phase_engine(args):
    os.makedirs(args.cache_dir, exist_ok=True)

    from nanovllm.llm import LLM
    from nanovllm.engine.sequence import Sequence

    print(f"[engine] constructing tensor_parallel_size={args.tp} (== ep_size) "
          f"from {args.checkpoint}, gpu_memory_utilization={args.gpu_memory_utilization}, "
          f"BF16 eager (no w8a8 / fused kernels / cuda graph)")
    llm = LLM(
        args.checkpoint,
        enforce_eager=True,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=max(8, len(PROMPTS)),
        max_model_len=MAX_MODEL_LEN,
    )

    layer_types = list(llm.model_runner.model.model.layer_types)
    n_full = layer_types.count("full_attention")
    n_linear = layer_types.count("linear_attention")
    print(f"[engine] layer_types: {len(layer_types)} layers -> "
          f"{n_full} full_attention, {n_linear} linear_attention (GDN)")

    # The full (un-sharded) hidden size, straight from the engine's own config.
    # Every captured residual-stream tensor must be this wide on rank0: at
    # tp>1 a ColumnParallel shard would be hidden_size/tp wide, which would
    # silently mean the tensor had NOT been through the TP all-reduce and must
    # therefore never be compared against HF's full-width tensor.
    hidden_size = int(llm.model_runner.config.hf_config.hidden_size)
    print(f"[engine] full hidden_size={hidden_size} (tp={args.tp} -> a "
          f"non-reduced shard would be {hidden_size // args.tp})")

    results = []
    for pi, prompt in enumerate(PROMPTS, start=1):
        prompt_ids = llm.tokenizer.encode(prompt)
        seq = Sequence(prompt_ids)
        seq.num_scheduled_tokens = len(prompt_ids)
        llm.scheduler.block_manager.allocate(seq, 0)
        # Dispatch through .call(...) so allocate_state_slot runs on EVERY
        # rank -- at tp>1 each rank owns an independent StateManager, and
        # setting seq.state_slot directly would only zero rank0's copy
        # (reference_check_phase6_layerwise.py:93-102 documents this).
        if llm.model_runner.state_manager is not None:
            llm.model_runner.call("allocate_state_slot", seq)

        captured, attn_only, post_ln, logits = llm.model_runner.call(
            "get_prefill_layer_states", [seq]
        )
        if captured is None:
            raise RuntimeError(
                "get_prefill_layer_states returned None on what should be rank0 "
                "-- the capture is not available to compare against HF")

        slot_used = seq.state_slot
        _free_diagnostic_seq(llm, seq)

        print(f"[engine] prompt {pi}/{len(PROMPTS)} {prompt!r} -> "
              f"{len(prompt_ids)} tokens, slot={slot_used}, "
              f"{len(captured)} residual states (first {tuple(captured[0].shape)}), "
              f"{len(attn_only)} attn-only, {len(post_ln)} post-LN, "
              f"logits {tuple(logits.shape)}")

        # Hard check, not a printout: this is the test's core TP/EP claim.
        # Asserting here means a broken reduction fails loudly at capture time
        # instead of surfacing later as a shape error in --phase compare.
        width = int(captured[0].shape[-1])
        if width != hidden_size:
            _fail(f"rank0 captured residual-stream width {width} != full "
                  f"hidden_size {hidden_size} (tp={args.tp}). The tensors were NOT "
                  f"collected across TP ranks -- at tp={args.tp} a per-rank shard "
                  f"would be {hidden_size // args.tp}. Comparing these against HF "
                  f"would be invalid.")
        for name, seq_tensors in (("layer_states", captured), ("attn_only", attn_only),
                                  ("post_ln", post_ln)):
            for j, t in enumerate(seq_tensors):
                if int(t.shape[-1]) != hidden_size:
                    _fail(f"{name}[{j}] width {int(t.shape[-1])} != hidden_size "
                          f"{hidden_size} -- not TP-reduced.")

        results.append({
            "prompt": prompt,
            "prompt_ids": prompt_ids,
            "state_slot": slot_used,
            "layer_states": [t.float().cpu() for t in captured],
            "attn_only": [t.float().cpu() for t in attn_only],
            "post_ln": [t.float().cpu() for t in post_ln],
            "logits": logits.float().cpu(),
        })

    eng_path = _engine_path(args.cache_dir, args.tp)
    _atomic_torch_save({
        "artifact_kind": "layerwise_engine",
        "format_version": FORMAT_VERSION,
        "tp": args.tp,
        "dtype": "bfloat16",
        "checkpoint": args.checkpoint,
        "prompts": list(PROMPTS),
        "layer_types": layer_types,
        "hidden_size": hidden_size,
        "num_layers": len(layer_types),
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": True,
        "results": results,
    }, eng_path)
    print(f"[engine] saved to {eng_path}")
    print("[engine] PHASE COMPLETE -- exit this process fully before --phase hf")


# --------------------------------------------------------------------------
# phase hf
# --------------------------------------------------------------------------
def phase_hf(args):
    os.makedirs(args.cache_dir, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    print(f"[hf] loading independent HF reference from {args.checkpoint} (device_map='auto') ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    decoder_layers = model.model.layers
    num_layers = len(decoder_layers)
    mixer_names = [
        next(name for name, _ in layer.named_children() if name not in _SKIP_CHILD_NAMES)
        for layer in decoder_layers
    ]
    print(f"[hf] {num_layers} decoder layers; mixer submodule names (first 5): {mixer_names[:5]}")

    handles = []
    # One mutable holder that the hooks write into, with the three buffers
    # swapped out per prompt. Using a dict (rather than rebinding three bare
    # locals that the closures capture) keeps the write path explicit and
    # reviewable: the hooks always index `caps[name]`, so re-ordering the
    # reset below can never leave a hook writing into a stale list.
    caps = {
        "mix_out": [None] * num_layers,   # attention/linear-attn sublayer OUTPUT (full seq)
        "mix_in": [None] * num_layers,    # sublayer INPUT == post-input_layernorm
        # AUTHORITATIVE per-decoder-layer OUTPUT. Do NOT rely on
        # out.hidden_states[-1] for the final layer -- see the long note below.
        "lay_out": [None] * num_layers,
    }

    def _reset_caps() -> None:
        for k in caps:
            caps[k] = [None] * num_layers

    # HF's output_hidden_states IS NOT a pure list of layer outputs for this
    # checkpoint. Measured on 2026-10-10 against this checkpoint:
    #   ||hidden_states[39]|| = 14.25  but  ||hidden_states[40]|| = 110.99
    # while the residual-stream norm grows smoothly (~8 -> 14 over layers
    # 35-39) and the engine's layer-39 raw residual is 15.60. HF's text model
    # applies `hidden_states = self.norm(hidden_states)` INSIDE its inner
    # forward (modeling_qwen3_5_moe.py:1402) before returning, and the last
    # entry of output_hidden_states is that POST-final-normalization tensor.
    # Proof it is post-norm, not layer 39's raw output: the final norm here is
    # the (1+w) variant, whose output norm is bounded near RMS(1+w)*sqrt(d) =
    # 117.1 (measured d=2048, RMS(1+w)=2.5878); the plain variant would cap at
    # RMS(w)*sqrt(d) = 72.1. Observed ||hidden_states[40]|| = 110.99 fits the
    # (1+w) bound and rules out both the plain-norm and raw-residual readings.
    # Comparing the engine's PRE-final-norm residual against it is comparing
    # two different semantic quantities and yields a fake "last layer is
    # catastrophically wrong" reading (norm ratio 0.14, max abs err 26). So
    # each layer's own module output is taken here via hooks instead.
    def _make_layer_hook(idx):
        def _hook(_module, _args, _kwargs, output):
            out = output[0] if isinstance(output, (tuple, list)) else output
            caps["lay_out"][idx] = out[0].detach().float().cpu()
        return _hook

    def _make_hook(idx):
        # with_kwargs=True because HF calls the mixer with hidden_states as a
        # KEYWORD arg -- the default hook signature sees an empty `args`.
        def _hook(module, _args, kwargs, output):
            out = output[0] if isinstance(output, (tuple, list)) else output
            caps["mix_out"][idx] = out[0].detach().float().cpu()          # (seq_len, hidden)
            in_ = kwargs.get("hidden_states", _args[0] if _args else None)
            assert in_ is not None, f"layer {idx}: no hidden_states in hook args/kwargs"
            caps["mix_in"][idx] = in_[0].detach().float().cpu()           # (seq_len, hidden)
        return _hook

    for i, layer in enumerate(decoder_layers):
        handles.append(getattr(layer, mixer_names[i]).register_forward_hook(_make_hook(i), with_kwargs=True))
        handles.append(layer.register_forward_hook(_make_layer_hook(i), with_kwargs=True))

    results = []
    try:
        for pi, prompt in enumerate(PROMPTS, start=1):
            prompt_ids = tokenizer.encode(prompt)
            input_ids = torch.tensor([prompt_ids], dtype=torch.long).to(model.device)
            _reset_caps()
            with torch.no_grad():
                # use_cache=False: this is prefill-only with no past_key_values,
                # so it cannot change any computation; it just stops each
                # successive prompt from populating a cache nobody reads (this
                # script runs several prompts through one loaded model).
                out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)

            if not hasattr(out, "hidden_states") or out.hidden_states is None:
                raise RuntimeError("[hf] output_hidden_states=True produced no .hidden_states")
            assert all(m is not None for m in caps["mix_out"]), "some layers' mixer hook never fired"
            assert all(m is not None for m in caps["mix_in"]), "some layers' mixer input hook never fired"
            assert all(m is not None for m in caps["lay_out"]), "some decoder-layer hook never fired"

            hidden = [hs[0].detach().float().cpu() for hs in out.hidden_states]  # each (seq_len, hidden)
            logits = out.logits[0, -1, :].float().cpu()
            hidden_size = int(hidden[0].shape[-1])
            print(f"[hf] prompt {pi}/{len(PROMPTS)} {prompt!r} -> {len(prompt_ids)} tokens, "
                  f"{len(hidden)} residual states (first {tuple(hidden[0].shape)}), "
                  f"logits {tuple(logits.shape)}")
            results.append({
                "prompt": prompt,
                "prompt_ids": prompt_ids,
                "layer_out": list(caps["lay_out"]),  # authoritative per-layer OUTPUT (pre-final-norm)
                "hidden": hidden,     # index num_layers is POST-final-norm: do not compare raw
                "mix_out": list(caps["mix_out"]),
                "mix_in": list(caps["mix_in"]),
                "logits": logits,
            })
    finally:
        for h in handles:
            h.remove()

    hf_path = _hf_path(args.cache_dir)
    _atomic_torch_save({
        "artifact_kind": "layerwise_hf",
        "format_version": FORMAT_VERSION,
        "checkpoint": args.checkpoint,
        "prompts": list(PROMPTS),
        "mixer_names": mixer_names,
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "results": results,
    }, hf_path)
    print(f"[hf] saved to {hf_path}")
    print("[hf] PHASE COMPLETE")


# --------------------------------------------------------------------------
# phase compare
# --------------------------------------------------------------------------
def _last_row(t: torch.Tensor) -> torch.Tensor:
    return t[-1, :] if t.dim() == 2 else t


def _topk_metrics(ref_logits: torch.Tensor, eng_logits: torch.Tensor, k: int = 5) -> dict:
    r = ref_logits.reshape(-1).float()
    e = eng_logits.reshape(-1).float()
    r_top1 = int(r.argmax())
    e_top1 = int(e.argmax())
    r_topk = r.topk(k).indices.tolist()
    e_topk = e.topk(k).indices.tolist()
    srt = r.topk(2).values
    return {
        "top1_ref": r_top1,
        "top1_eng": e_top1,
        "top1_match": int(r_top1 == e_top1),
        "top5_set_match": int(set(r_topk) == set(e_topk)),
        "top5_overlap": len(set(r_topk) & set(e_topk)),
        "ref_top1_in_eng_top5": int(r_top1 in set(e_topk)),
        "top1_margin": float(srt[0] - srt[1]),
    }


CSV_FIELDS = [
    "prompt_idx", "prompt", "checkpoint_idx", "label", "layer_idx", "layer_type",
    "tensor_kind", "scope", "shape_ref", "shape_eng", "cosine", "max_abs_err",
    "mean_abs_err", "rel_l2_err", "norm_ratio", "nan_count", "inf_count", "ref_absmax",
    "top1_match", "top5_set_match", "top5_overlap", "ref_top1_in_eng_top5", "top1_margin",
]


def _row(pi, prompt, cidx, label, layer_idx, layer_type, kind, scope, m, extra=None):
    r = {
        "prompt_idx": pi, "prompt": prompt, "checkpoint_idx": cidx, "label": label,
        "layer_idx": layer_idx, "layer_type": layer_type, "tensor_kind": kind,
        "scope": scope, "shape_ref": str(m["shape_ref"]), "shape_eng": str(m["shape_eng"]),
        "cosine": f"{m['cosine']:.6f}", "max_abs_err": f"{m['max_abs_err']:.6e}",
        "mean_abs_err": f"{m['mean_abs_err']:.6e}", "rel_l2_err": f"{m['rel_l2_err']:.6e}",
        "norm_ratio": f"{m['norm_ratio']:.6f}", "nan_count": m["nan_count"],
        "inf_count": m["inf_count"], "ref_absmax": f"{m['ref_absmax']:.6f}",
        "top1_match": "", "top5_set_match": "", "top5_overlap": "",
        "ref_top1_in_eng_top5": "", "top1_margin": "",
    }
    if extra:
        for k, v in extra.items():
            r[k] = v
    return r


def _print_metric_row(idx, label, kind, scope, m) -> None:
    """One console row of the comparison matrix. Every call site shares this
    exact column layout, so the format string lives in one place. The CSV is
    written from the dicts, not from this line, so the two cannot drift."""
    print(f"  {idx:>4}  {label:<26} {kind:<15} {scope:<14} "
          f"{m['cosine']:>11.6f} {m['max_abs_err']:>11.3e} {m['mean_abs_err']:>11.3e} "
          f"{m['rel_l2_err']:>11.3e} {m['norm_ratio']:>10.6f} "
          f"{str(m['nan_count']) + '/' + str(m['inf_count']):>8}")


def phase_compare(args):
    eng_path = _engine_path(args.cache_dir, args.tp)
    hf_path = _hf_path(args.cache_dir)
    if not os.path.exists(hf_path):
        _fail(f"missing {hf_path} -- run --phase hf first.")
    if not os.path.exists(eng_path):
        _fail(f"missing {eng_path} -- run --phase engine --tp {args.tp} first.")

    eng_payload = torch.load(eng_path, weights_only=False)
    hf_payload = torch.load(hf_path, weights_only=False)

    # Everything that would make the comparison meaningless if the two files
    # disagreed is checked BEFORE a single metric is computed.
    num_layers = _validate_artifacts(eng_payload, hf_payload, eng_path, hf_path, args)

    eng_results, hf_results = eng_payload["results"], hf_payload["results"]
    layer_types = eng_payload["layer_types"]
    tp = eng_payload["tp"]

    # NaN/Inf gate across every captured tensor, before any metric. Fail-fast
    # so the worst-layer selection below can never rank NaN as "least similar".
    for i, (er, hr) in enumerate(zip(eng_results, hf_results), start=1):
        for name in ("layer_states", "attn_only", "post_ln"):
            for j, t in enumerate(er[name]):
                _require_finite(f"engine prompt#{i} {name}[{j}]", t)
        _require_finite(f"engine prompt#{i} logits", er["logits"])
        for name in ("hidden", "layer_out", "mix_out", "mix_in"):
            for j, t in enumerate(hr[name]):
                _require_finite(f"hf prompt#{i} {name}[{j}]", t)
        _require_finite(f"hf prompt#{i} logits", hr["logits"])

    print("=" * 100)
    print(f"TP=4 LAYERWISE NUMERICAL PARITY -- engine (tp={tp}, ep={tp}) vs independent dense HF reference")
    print(f"engine checkpoint : {eng_payload['checkpoint']}")
    print(f"hf checkpoint     : {hf_payload['checkpoint']}")
    print(f"layers            : {num_layers} ({layer_types.count('full_attention')} full_attention, "
          f"{layer_types.count('linear_attention')} linear_attention/GDN)")
    print(f"ACCEPTANCE (author's bar, FINAL PREFILL LOGITS ONLY): "
          f"cosine >= {FINAL_LOGITS_COSINE_THRESHOLD} AND Top-1 match")
    print(f"Intermediate checkpoints are RECORDED, not pass/failed: the author set no threshold for them.")
    print("=" * 100)

    rows = []
    all_ids_match = True
    all_finite = True
    final_pass_count = 0
    worst_per_prompt = []
    # (raw_cosine, label, tensor_kind, prompt_idx) for every last-token row.
    # Holds the raw float64 cosine, NOT the 6-decimal string that goes into the
    # CSV, so the "worst overall" ranking can never be decided by display
    # rounding.
    worst_last = []

    for pi, (er, hr) in enumerate(zip(eng_results, hf_results), start=1):
        if er["prompt"] != hr["prompt"]:
            raise ValueError(
                f"prompt ordering differs at index {pi}: engine={er['prompt']!r} "
                f"hf={hr['prompt']!r} -- the two artifacts are not positionally "
                f"aligned, so zip() would compare unrelated prompts")
        prompt = er["prompt"]
        ids_match = list(er["prompt_ids"]) == list(hr["prompt_ids"])
        all_ids_match &= ids_match
        print(f"\n{'#' * 100}")
        print(f"# PROMPT {pi}/{len(eng_results)}: {prompt!r}")
        print(f"# token ids: engine={er['prompt_ids']}")
        print(f"#            hf    ={hr['prompt_ids']}")
        print(f"# IDENTICAL={ids_match}" + ("" if ids_match else "   <-- CONFOUNDED: every number below is invalid"))
        print(f"# engine GDN state slot used: {er['state_slot']}")
        print(f"{'#' * 100}")
        if not ids_match:
            _fail(f"prompt #{pi} token IDs differ between engine and HF:\n"
                  f"    engine: {list(er['prompt_ids'])}\n"
                  f"    hf    : {list(hr['prompt_ids'])}\n"
                  f"Comparing logits produced from different token sequences is "
                  f"meaningless -- fix tokenization first.")

        n_engine = len(er["layer_states"])

        hdr = (f"  {'idx':>4}  {'checkpoint':<26} {'kind':<15} {'scope':<14} "
               f"{'cosine':>11} {'maxAbsErr':>11} {'meanAbsErr':>11} {'relL2':>11} "
               f"{'normRatio':>10} {'nan/inf':>8}")
        print(hdr)
        print("  " + "-" * (len(hdr) - 1))

        local_cos = []   # (label, scope, cosine) for worst-layer reporting
        for cidx in range(n_engine):
            e_full = er["layer_states"][cidx]
            # Index 0 (embedding) exists only in `hidden`. Every decoder layer
            # uses `layer_out`, the authoritative per-layer hook capture; the
            # corresponding `hidden[cidx]` entry would be POST-final-norm for
            # cidx == num_layers and must never be compared as a raw residual.
            # _validate_artifacts has already guaranteed layer_out is present
            # and complete, so there is no fallback path here by design.
            h_full = hr["hidden"][0] if cidx == 0 else hr["layer_out"][cidx - 1]
            if cidx == 0:
                label, layer_idx, ltype = "embedding", "", "n/a"
            else:
                layer_idx = cidx - 1
                label = f"layer {layer_idx} residual"
                ltype = layer_types[layer_idx]
            # PRIMARY: last-token row (the row that feeds norm -> lm_head).
            m = _metrics(_last_row(e_full), _last_row(h_full))
            all_finite &= (m["nan_count"] == 0 and m["inf_count"] == 0)
            rows.append(_row(pi, prompt, cidx, label, layer_idx, ltype,
                             "residual_stream", "last_token", m))
            local_cos.append((label, "last_token", m["cosine"]))
            worst_last.append((m["cosine"], label, "residual_stream", pi))
            _print_metric_row(cidx, label, "residual", "last_token", m)
            # SECONDARY: whole sequence, explicitly labelled as a different scope.
            m2 = _metrics(e_full, h_full)
            all_finite &= (m2["nan_count"] == 0 and m2["inf_count"] == 0)
            rows.append(_row(pi, prompt, cidx, label, layer_idx, ltype,
                             "residual_stream", "full_sequence", m2))
            local_cos.append((label, "full_sequence", m2["cosine"]))
            _print_metric_row(cidx, label, "residual", "full_sequence", m2)

        print("  " + "-" * (len(hdr) - 1))
        for li in range(num_layers):
            ltype = layer_types[li]
            # attention / GDN sublayer output only (pre-residual, pre-MoE)
            m = _metrics(_last_row(er["attn_only"][li]), _last_row(hr["mix_out"][li]))
            all_finite &= (m["nan_count"] == 0 and m["inf_count"] == 0)
            rows.append(_row(pi, prompt, li + 1, f"layer {li} mixer out", li, ltype,
                             "mixer_output", "last_token", m))
            local_cos.append((f"layer {li} mixer out", "last_token", m["cosine"]))
            worst_last.append((m["cosine"], f"layer {li} mixer out", "mixer_output", pi))
            _print_metric_row(li + 1, f"layer {li} mixer out", "mixer_out", "last_token", m)
            # post-input_layernorm == the mixer's INPUT (isolates RMSNorm from mixer)
            m = _metrics(_last_row(er["post_ln"][li]), _last_row(hr["mix_in"][li]))
            all_finite &= (m["nan_count"] == 0 and m["inf_count"] == 0)
            rows.append(_row(pi, prompt, li + 1, f"layer {li} post-LN", li, ltype,
                             "post_input_layernorm", "last_token", m))
            local_cos.append((f"layer {li} post-LN", "last_token", m["cosine"]))
            worst_last.append((m["cosine"], f"layer {li} post-LN",
                               "post_input_layernorm", pi))
            _print_metric_row(li + 1, f"layer {li} post-LN", "post_layernorm",
                              "last_token", m)

        # ---------------- final logits (the author's acceptance bar) --------
        m = _metrics(er["logits"].reshape(-1), hr["logits"].reshape(-1))
        all_finite &= (m["nan_count"] == 0 and m["inf_count"] == 0)
        tk = _topk_metrics(hr["logits"], er["logits"], k=5)
        p = m["cosine"] >= FINAL_LOGITS_COSINE_THRESHOLD and bool(tk["top1_match"])
        final_pass_count += int(p)
        rows.append(_row(pi, prompt, n_engine, "final logits (norm+lm_head)", "", "n/a",
                         "final_logits", "last_token", m,
                         extra={"top1_match": tk["top1_match"],
                                "top5_set_match": tk["top5_set_match"],
                                "top5_overlap": tk["top5_overlap"],
                                "ref_top1_in_eng_top5": tk["ref_top1_in_eng_top5"],
                                "top1_margin": f"{tk['top1_margin']:.6f}"}))
        # The final logits ARE a last-token quantity, so they take part in the
        # "worst last-token cosine overall" ranking, exactly as before.
        worst_last.append((m["cosine"], "final logits (norm+lm_head)",
                           "final_logits", pi))
        print("  " + "-" * (len(hdr) - 1))
        _print_metric_row(n_engine, "final logits", "norm+lm_head", "last_token", m)
        print(f"    FINAL LOGITS: cosine={m['cosine']:.6f} (bar >= {FINAL_LOGITS_COSINE_THRESHOLD})  "
              f"top1 ref={tk['top1_ref']} eng={tk['top1_eng']} match={bool(tk['top1_match'])}")
        print(f"                  top5 SET match={bool(tk['top5_set_match'])}  "
              f"top5 overlap={tk['top5_overlap']}/5  ref_top1_in_eng_top5="
              f"{bool(tk['ref_top1_in_eng_top5'])}  ref top1 margin={tk['top1_margin']:.4f}")
        print(f"                  -> {'PASS' if p else 'FAIL'} "
              f"(author's A2 bar: cosine>={FINAL_LOGITS_COSINE_THRESHOLD} AND top1 match)")

        wr = min(local_cos, key=lambda x: x[2])
        worst_per_prompt.append((pi, wr[0], wr[1], wr[2]))
        print(f"    WORST intermediate checkpoint: {wr[0]} [{wr[1]}] cosine={wr[2]:.6f}")

    # ----------------------------- summary --------------------------------
    n = len(eng_results)
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"HF per-layer reference source : decoder-layer forward hooks "
          f"(authoritative, pre-final-norm); hidden_states[num_layers] is "
          f"POST-final-norm and is never compared")
    print(f"prompt token IDs identical    : {all_ids_match} (all {n} prompts)")
    print(f"all tensors finite         : {all_finite}")
    print(f"FINAL LOGITS (author's bar): {final_pass_count}/{n} prompts passed "
          f"(cosine>={FINAL_LOGITS_COSINE_THRESHOLD} AND top1 match)")
    print("\nWorst intermediate checkpoint per prompt (no threshold asserted):")
    print(f"  {'prompt':>7}  {'checkpoint':<32} {'scope':<15} {'cosine':>12}")
    for pi, label, scope, cos in worst_per_prompt:
        print(f"  {pi:>7}  {label:<32} {scope:<15} {cos:>12.6f}")

    if worst_last:
        wcos, wlabel, wkind, wpi = min(worst_last)
        print(f"\nWORST last-token cosine overall: {wcos:.6f} at "
              f"{wlabel} ({wkind}), prompt {wpi}")

    # GDN vs full-attention error behaviour, reported separately as asked.
    if rows:
        print("\nError behaviour by layer type (residual_stream, last_token):")
        print(f"  {'layer_type':<20} {'n':>5} {'min cos':>12} {'median cos':>12} {'mean maxAbsErr':>15}")
        for lt in ("linear_attention", "full_attention"):
            vals = [float(r["cosine"]) for r in rows
                    if r["layer_type"] == lt and r["scope"] == "last_token"
                    and r["tensor_kind"] == "residual_stream"]
            absvals = [float(r["max_abs_err"]) for r in rows
                       if r["layer_type"] == lt and r["scope"] == "last_token"
                       and r["tensor_kind"] == "residual_stream"]
            if not vals:
                continue
            med = sorted(vals)[len(vals) // 2]
            print(f"  {lt:<20} {len(vals):>5} {min(vals):>12.6f} {med:>12.6f} "
                  f"{sum(absvals) / len(absvals):>15.3e}")

    os.makedirs(args.csv_dir, exist_ok=True)
    csv_path = os.path.join(args.csv_dir, "layerwise_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nCSV written: {csv_path}  ({len(rows)} rows)")

    verdict = all_ids_match and all_finite and (final_pass_count == n)
    print("\n" + "=" * 100)
    print(f"LAYERWISE PARITY (tp={tp}) VERDICT on the author's FINAL-LOGITS bar: "
          f"{'PASS' if verdict else 'FAIL'}")
    print("Intermediate-layer numbers above are recorded WITHOUT a pass/fail threshold.")
    print("=" * 100)
    if not verdict:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", required=True, choices=["engine", "hf", "compare"])
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    # Default 2 matches every other script in this repo (cluster_a2's --tp
    # default and phase6_layerwise's hardcoded value); pass --tp 4 explicitly.
    ap.add_argument("--tp", type=int, default=2, help="tensor_parallel_size (== ep_size)")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--csv-dir", default=os.path.join(ROOT, "outputs", "h100_tp4_phase_b"))
    # Where the engine/hf artifacts are read and written. Overridable so a
    # clean regression run can use a fresh directory (never silently
    # overwriting a previous capture) and so tests can point at a throwaway
    # directory instead of the real one.
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR,
                    help="directory holding layerwise_engine_tp<N>.pt / layerwise_hf.pt")
    args = ap.parse_args()

    if args.phase == "engine":
        phase_engine(args)
    elif args.phase == "hf":
        phase_hf(args)
    else:
        phase_compare(args)


if __name__ == "__main__":
    main()
