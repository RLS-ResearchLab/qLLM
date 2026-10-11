"""TP-aware layerwise parity against an independent BF16 HF reference.
Run engine, hf and compare in separate processes (compare is CPU-only):
    python tests/cluster_a2_tp_layerwise.py --phase engine --tp 4 --checkpoint MODEL
    python tests/cluster_a2_tp_layerwise.py --phase hf --checkpoint MODEL
    python tests/cluster_a2_tp_layerwise.py --phase compare --tp 4
Use --cache-dir for fresh captures and --csv-dir for reports. Final logits
must have float64 cosine >= 0.99 AND matching Top-1; intermediate metrics
are diagnostic only. Captures retain full sequences; residuals report both
last-token and full-sequence metrics. Mixer/post-LN report the last token.
Schema v2 requires decoder-layer hooks: HF hidden_states[-1] is post-final-
norm and cannot be compared against the engine's raw last-layer residual.
Cache identity, token IDs, complete hooks, widths and finite values are
validated; invalid inputs or a failed final-logits bar exit nonzero.
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

_SKIP_CHILD_NAMES = {"input_layernorm", "post_attention_layernorm", "mlp"}
DEFAULT_CKPT = os.path.join(ROOT, "qwen35_checkpoint")
DEFAULT_CACHE_DIR = os.path.join(ROOT, "tests", "_phase_b_cache")
MAX_MODEL_LEN = 2048
FORMAT_VERSION = 2
_ENGINE_KEYS = ("artifact_kind", "format_version", "tp", "dtype", "checkpoint",
                "prompts", "layer_types", "hidden_size", "num_layers", "results")
_HF_KEYS = ("artifact_kind", "format_version", "checkpoint", "prompts",
            "num_layers", "hidden_size", "results")
_ENGINE_RESULT_KEYS = ("prompt", "prompt_ids", "state_slot", "layer_states",
                       "attn_only", "post_ln", "logits")
_HF_RESULT_KEYS = ("prompt", "prompt_ids", "layer_out", "hidden", "mix_out",
                   "mix_in", "logits")
PROMPTS = [
    "The capital of France is",
    "If John has 3 apples and buys 5 more, he has",
    "2 + 2 =",
    "In a small village, there lived a farmer who had",
    "The quick brown fox jumps over the lazy dog. This sentence is famous because",
]
FINAL_LOGITS_COSINE_THRESHOLD = 0.99


def _engine_path(cache_dir: str, tp: int) -> str:
    return os.path.join(cache_dir, f"layerwise_engine_tp{tp}.pt")


def _hf_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, "layerwise_hf.pt")


def _metrics(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Reduce in float64: float32 cosine loses precision at vocab=248320."""
    a = a.detach().reshape(-1).double()
    b = b.detach().reshape(-1).double()
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
    tmp_path = path + ".tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def _free_diagnostic_seq(llm, seq) -> None:
    llm.scheduler.block_manager.deallocate(seq)
    if llm.model_runner.state_manager is not None:
        llm.model_runner.call("free_state_slot", seq)


def _fail(msg: str) -> None:
    raise SystemExit(f"[layerwise] REFUSING: {msg}")


def _require_finite(label: str, t: torch.Tensor) -> None:
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
    if eng["checkpoint"] != hf["checkpoint"]:
        _fail(f"checkpoint mismatch between the two captures:\n"
              f"    engine : {eng['checkpoint']}\n"
              f"    hf     : {hf['checkpoint']}\n"
              f"These were captured from DIFFERENT models; refusing to compare.")
    if eng["tp"] != args.tp:
        _fail(f"{eng_path} was captured with tp={eng['tp']} but --tp {args.tp} was "
              f"requested. Pass --tp {eng['tp']} or re-run --phase engine --tp {args.tp}.")
    if eng["hidden_size"] != hf["hidden_size"]:
        _fail(f"hidden_size mismatch: engine={eng['hidden_size']} "
              f"hf={hf['hidden_size']} -- the two captures are not of the same "
              f"model.")
    if list(eng["prompts"]) != list(hf["prompts"]):
        _fail(f"prompt list mismatch between the two captures:\n"
              f"    engine : {list(eng['prompts'])}\n"
              f"    hf     : {list(hf['prompts'])}")
    num_layers = len(eng["layer_types"])
    if num_layers < 1 or not eng["results"] or eng["hidden_size"] < 1 or args.tp < 1:
        _fail("empty capture or invalid layer count, hidden size, or TP size.")
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
        for side, result, keys, streams in (
            ("engine", er, _ENGINE_RESULT_KEYS,
             (("layer_states", num_layers + 1), ("attn_only", num_layers),
              ("post_ln", num_layers))),
            ("hf", hr, _HF_RESULT_KEYS,
             (("hidden", num_layers + 1), ("layer_out", num_layers),
              ("mix_out", num_layers), ("mix_in", num_layers))),
        ):
            _require_keys(result, keys, f"{side} result #{i}")
            if result["prompt"] != eng["prompts"][i - 1] or not result["prompt_ids"]:
                _fail(f"{side} result #{i}: prompt identity mismatch or empty token IDs.")
            for name, expected in streams:
                seq = result[name]
                if len(seq) != expected:
                    _fail(f"{side} result #{i}: {name} has {len(seq)} entries, "
                          f"expected {expected}.")
                if any(t is None for t in seq):
                    _fail(f"{side} result #{i}: {name} contains None entry(ies) "
                          "-- a capture hook or step did not run.")
                expected_shape = (len(result["prompt_ids"]), eng["hidden_size"])
                for t in seq:
                    if not isinstance(t, torch.Tensor) or tuple(t.shape) != expected_shape:
                        _fail(f"{side} result #{i}: {name} shape mismatch; "
                              f"expected full TP hidden shape {expected_shape}.")
    return num_layers


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
    hidden_size = int(llm.model_runner.config.hf_config.hidden_size)
    print(f"[engine] full hidden_size={hidden_size} (tp={args.tp} -> a "
          f"non-reduced shard would be {hidden_size // args.tp})")

    results = []
    for pi, prompt in enumerate(PROMPTS, start=1):
        prompt_ids = llm.tokenizer.encode(prompt)
        seq = Sequence(prompt_ids)
        seq.num_scheduled_tokens = len(prompt_ids)
        llm.scheduler.block_manager.allocate(seq, 0)
        # Dispatch allocation to every TP rank to clear warmup state.
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
    caps = {
        "mix_out": [None] * num_layers,
        "mix_in": [None] * num_layers,
        "lay_out": [None] * num_layers,
    }

    def _reset_caps() -> None:
        for k in caps:
            caps[k] = [None] * num_layers

    # Capture each raw decoder output, including the layer before final norm.
    def _make_layer_hook(idx):
        def _hook(_module, _args, _kwargs, output):
            out = output[0] if isinstance(output, (tuple, list)) else output
            caps["lay_out"][idx] = out[0].detach().float().cpu()
        return _hook

    # with_kwargs=True: HF passes hidden_states as a keyword argument.
    def _make_hook(idx):
        def _hook(module, _args, kwargs, output):
            out = output[0] if isinstance(output, (tuple, list)) else output
            caps["mix_out"][idx] = out[0].detach().float().cpu()
            in_ = kwargs.get("hidden_states", _args[0] if _args else None)
            assert in_ is not None, f"layer {idx}: no hidden_states in hook args/kwargs"
            caps["mix_in"][idx] = in_[0].detach().float().cpu()
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
                out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
            if not hasattr(out, "hidden_states") or out.hidden_states is None:
                raise RuntimeError("[hf] output_hidden_states=True produced no .hidden_states")
            assert all(m is not None for m in caps["mix_out"]), "some layers' mixer hook never fired"
            assert all(m is not None for m in caps["mix_in"]), "some layers' mixer input hook never fired"
            assert all(m is not None for m in caps["lay_out"]), "some decoder-layer hook never fired"
            hidden = [hs[0].detach().float().cpu() for hs in out.hidden_states]
            logits = out.logits[0, -1, :].float().cpu()
            hidden_size = int(hidden[0].shape[-1])
            print(f"[hf] prompt {pi}/{len(PROMPTS)} {prompt!r} -> {len(prompt_ids)} tokens, "
                  f"{len(hidden)} residual states (first {tuple(hidden[0].shape)}), "
                  f"logits {tuple(logits.shape)}")
            results.append({
                "prompt": prompt,
                "prompt_ids": prompt_ids,
                "layer_out": list(caps["lay_out"]),
                "hidden": hidden,
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
    num_layers = _validate_artifacts(eng_payload, hf_payload, eng_path, hf_path, args)
    eng_results, hf_results = eng_payload["results"], hf_payload["results"]
    layer_types = eng_payload["layer_types"]
    tp = eng_payload["tp"]
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
    print(f"LAYERWISE NUMERICAL PARITY -- engine (tp={tp}, ep={tp}) vs independent dense HF reference")
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
        local_cos = []

        def record(e, h, cidx, label, layer_idx, ltype, kind, display, scope):
            nonlocal all_finite
            if scope == "last_token":
                e, h = _last_row(e), _last_row(h)
            m = _metrics(e, h)
            all_finite &= m["nan_count"] == 0 and m["inf_count"] == 0
            rows.append(_row(pi, prompt, cidx, label, layer_idx, ltype, kind, scope, m))
            local_cos.append((label, scope, m["cosine"]))
            if scope == "last_token":
                worst_last.append((m["cosine"], label, kind, pi))
            _print_metric_row(cidx, label, display, scope, m)

        for cidx in range(n_engine):
            e_full = er["layer_states"][cidx]
            # hidden[-1] is post-final-norm; layer_out is the raw residual.
            h_full = hr["hidden"][0] if cidx == 0 else hr["layer_out"][cidx - 1]
            if cidx == 0:
                label, layer_idx, ltype = "embedding", "", "n/a"
            else:
                layer_idx = cidx - 1
                label = f"layer {layer_idx} residual"
                ltype = layer_types[layer_idx]
            for scope in ("last_token", "full_sequence"):
                record(e_full, h_full, cidx, label, layer_idx, ltype,
                       "residual_stream", "residual", scope)
        print("  " + "-" * (len(hdr) - 1))
        for li in range(num_layers):
            for ek, hk, suffix, kind, display in (
                ("attn_only", "mix_out", "mixer out", "mixer_output", "mixer_out"),
                ("post_ln", "mix_in", "post-LN", "post_input_layernorm", "post_layernorm"),
            ):
                record(er[ek][li], hr[hk][li], li + 1, f"layer {li} {suffix}",
                       li, layer_types[li], kind, display, "last_token")

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
    ap.add_argument("--tp", type=int, default=2, help="tensor_parallel_size (== ep_size)")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--csv-dir", default=os.path.join(ROOT, "outputs", "h100_tp4_phase_b"))
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
