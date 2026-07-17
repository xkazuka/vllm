#!/usr/bin/env python3
"""Per-layer GPU timings from a vLLM torch-profiler trace (either engine).

Structure exploited (validated on GLM-5.2 P4D4 smoke traces, 2026-07-16):
  - every model forward is wrapped in a `execute_context_*_generation_*`
    gpu_user_annotation span (cudagraph replay AND eager);
  - the sparse-MLA attention core kernel (fmhaSm100f*TokenSparse*) runs
    exactly once per layer per forward -> 78 anchors segment the layers;
  - the DSA indexer exists on 21 of 78 layers (mqa_logits kernels).

Layer i window = [anchor_i_start, anchor_{i+1}_start); kernels before the
first anchor (embedding, layer-0 indexer/proj) join layer 0, kernels after
the last anchor (tail of layer 77 + final norm/lm_head) join layer 77.
Forwards whose anchor count != num_layers (clipped capture) are skipped.

Usage: parse_layer_timings.py TRACE [TRACE...] [--layers 78] [--json OUT]
"""

import argparse
import gzip
import json
import re
import sys
from collections import Counter, defaultdict

ANCHOR_PATTERNS = ("fmhasm100fkernel", "fmha")
CATEGORY_RULES = [
    ("indexer", ("mqa_logits", "paged_mqa", "indexer", "topk", "top_k",
                 "index_select", "radix")),
    ("comm", ("nccl", "all_reduce", "allreduce", "all_gather", "allgather",
              "reduce_scatter", "custom_ar", "one_shot", "two_shot",
              "cross_device", "lamport")),
    ("attention", ("fmha", "mla", "attn", "attention", "rotary", "rope",
                   "concat_and_cache", "reshape_and_cache", "kv_cache",
                   "softmax")),
    ("moe", ("moe", "expert", "grouped_gemm", "group_gemm", "fused_moe",
             "routing", "finalize", "activation", "silu", "swiglu",
             "act_and_mul")),
    ("gemm", ("gemm", "matmul", "cutlass", "nvjet", "bmm", "mm_")),
    ("quant", ("quant", "e4m3", "fp8", "fp4", "cvt", "convert", "cast")),
    ("norm", ("rms_norm", "layer_norm", "layernorm", "norm")),
]


def kernel_category(name: str) -> str:
    low = name.lower()
    for cat, keys in CATEGORY_RULES:
        if any(k in low for k in keys):
            return cat
    return "other"


def is_anchor(name: str) -> bool:
    low = name.lower()
    return low.startswith("fmhasm100fkernel") or (
        "fmha" in low and "tokensparse" in low
    )


def load(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", errors="replace") as f:
        data = json.load(f)
    return data["traceEvents"] if isinstance(data, dict) else data


def parse_trace(path, num_layers):
    evs = load(path)
    kernels = sorted(
        (e for e in evs if e.get("ph") == "X" and e.get("cat") == "kernel"
         and e.get("dur", 0) >= 0),
        key=lambda e: e["ts"],
    )
    spans = sorted(
        (e for e in evs if e.get("cat") == "gpu_user_annotation"
         and e["name"].startswith("execute_context")),
        key=lambda e: e["ts"],
    )
    forwards = []
    for s in spans:
        lo, hi = s["ts"], s["ts"] + s["dur"]
        inside = [k for k in kernels if lo <= k["ts"] < hi]
        anchors = [i for i, k in enumerate(inside) if is_anchor(k["name"])]
        if len(anchors) != num_layers:
            continue
        m = re.match(r"execute_context_\d+\((\d+)\)_generation_\d+\((\d+)\)",
                     s["name"])
        ctx_tok, gen_tok = (int(m.group(1)), int(m.group(2))) if m else (-1, -1)
        layer_ms = []
        layer_cat = []
        bounds = anchors + [len(inside)]
        start = 0
        for li in range(num_layers):
            end = bounds[li + 1]
            seg = inside[start:end]
            layer_ms.append(sum(k["dur"] for k in seg) / 1e3)
            cats = defaultdict(float)
            for k in seg:
                cats[kernel_category(k["name"])] += k["dur"] / 1e3
            layer_cat.append(dict(cats))
            start = end
        forwards.append({
            "name": s["name"], "ctx_tokens": ctx_tok, "gen_tokens": gen_tok,
            "span_ms": s["dur"] / 1e3,
            "gpu_ms": sum(k["dur"] for k in inside) / 1e3,
            "layer_ms": layer_ms, "layer_cat": layer_cat,
        })
    return forwards, len(kernels), len(spans)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--layers", type=int, default=78)
    ap.add_argument("--json", dest="json_out", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    all_out = []
    for path in args.traces:
        forwards, n_kern, n_spans = parse_trace(path, args.layers)
        out = {"trace": path, "kernels": n_kern, "spans": n_spans,
               "full_forwards": len(forwards)}
        print(f"\n=== {path}")
        print(f"kernels={n_kern} forward-spans={n_spans} "
              f"full({args.layers}-anchor) forwards={len(forwards)}")
        if not forwards:
            print("WARN: no complete forwards; nothing to segment")
            all_out.append(out)
            continue

        by_shape = defaultdict(list)
        for f in forwards:
            by_shape[(f["ctx_tokens"], f["gen_tokens"])].append(f)
        out["shapes"] = {}
        for (ctx, gen), fs in sorted(by_shape.items()):
            n = len(fs)
            step_ms = sum(f["gpu_ms"] for f in fs) / n
            layer_avg = [sum(f["layer_ms"][i] for f in fs) / n
                         for i in range(args.layers)]
            cat_avg = defaultdict(float)
            for f in fs:
                for lc in f["layer_cat"]:
                    for c, ms in lc.items():
                        cat_avg[c] += ms / n
            print(f"\nforward shape ctx={ctx} gen={gen}: {n} steps, "
                  f"avg gpu {step_ms:.3f} ms/step")
            if not args.quiet:
                print(f"{'layer':>5} {'ms':>8}   (avg over {n} steps)")
                for i, ms in enumerate(layer_avg):
                    print(f"{i:>5} {ms:>8.4f}")
            print("  category avg per step: "
                  + "  ".join(f"{c}={ms:.3f}" for c, ms in
                              sorted(cat_avg.items(), key=lambda kv: -kv[1])))
            out["shapes"][f"ctx{ctx}_gen{gen}"] = {
                "steps": n, "step_ms": step_ms, "layer_ms": layer_avg,
                "cat_ms": dict(cat_avg),
            }
        all_out.append(out)

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(all_out, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
