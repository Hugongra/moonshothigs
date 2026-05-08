#!/usr/bin/env python3
"""
One-shot pipeline: RAG retrieval eval + ATLAS case study replication → NeurIPS-style draft PDF/LaTeX.

  python run_neurips_pipeline.py
  python run_neurips_pipeline.py --skip-eval              # only atlas + paper (no Chroma)
  python run_neurips_pipeline.py --skip-atlas             # paper from existing metrics.json
  python run_neurips_pipeline.py --max-rows 8000 --no-compile
  python run_neurips_pipeline.py --ragas --ragas-max-queries -1 \\
      --queries evals/data/rag_queries_500_neurips_mirror.jsonl
  python run_neurips_pipeline.py --reuse-ogts-json evals/ogts/results/ogts_eval_latest.json

Requires: requirements-pipeline.txt + evals/requirements-eval.txt for full eval.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root))

    from evals.load_local_env import load_repo_dotenv

    load_repo_dotenv(root)

    p = argparse.ArgumentParser(
        description="NeurIPS draft: RAG eval methodology + ATLAS Higgs case study replication"
    )
    p.add_argument("--csv", type=Path, default=None, help="ATLAS CSV (default: data/...)")
    p.add_argument("--max-rows", type=int, default=150_000)
    p.add_argument("--full-train", action="store_true")
    p.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Stratified K-fold splits for ATLAS baseline CV metrics (default: 5).",
    )
    p.add_argument(
        "--atlas-output-dir",
        type=Path,
        default=None,
        help="Same as run_atlas_pipeline --output-dir (default: output/atlas_challenge)",
    )
    p.add_argument("--skip-atlas", action="store_true", help="Use existing atlas metrics.json only")
    p.add_argument("--skip-eval", action="store_true", help="Skip retrieval evaluation")
    p.add_argument(
        "--reuse-eval-json",
        type=Path,
        default=None,
        help="Use a previously written evals/results/*.json instead of re-running eval",
    )
    p.add_argument(
        "--reuse-ogts-json",
        type=Path,
        default=None,
        help="Optional evals/ogts/results/*.json to populate OGSR benchmark tables from an executed run.",
    )
    p.add_argument("--rag-db", type=Path, default=None, help="Chroma persist dir for eval")
    p.add_argument(
        "--queries",
        type=Path,
        default=root / "evals" / "data" / "rag_queries.jsonl",
    )
    p.add_argument("--collection", type=str, default=None)
    p.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Embedding model id for eval (must match the indexed Chroma DB: HF id or text-embedding-*).",
    )
    p.add_argument(
        "--enable-ragas",
        action="store_true",
        help="Enable optional RAGAS LLM-judge metrics during retrieval eval (extra deps + API keys).",
    )
    p.add_argument(
        "--ragas",
        action="store_true",
        help="Alias for --enable-ragas (same flag name as evals/run_retrieval_eval.py --ragas).",
    )
    p.add_argument(
        "--ragas-max-queries",
        type=int,
        default=25,
        help="When RAGAS is enabled, score only the first N queries (default: 25). Use -1 for the entire JSONL.",
    )
    p.add_argument("--k-list", type=int, nargs="+", default=[5, 10])
    p.add_argument("--max-k", type=int, default=20)
    p.add_argument(
        "--paper-dir",
        type=Path,
        default=None,
        help="NeurIPS draft output (default: output/neurips_paper)",
    )
    p.add_argument("--no-compile", action="store_true")
    p.add_argument(
        "--clean-output",
        action="store_true",
        help="Delete existing output/neurips_paper/ and output/neurips_overleaf_bundle/ before writing.",
    )
    p.add_argument(
        "--min-tex-bytes",
        type=int,
        default=2000,
        help="Minimum size of the rendered .tex; fail-loud below this threshold.",
    )
    p.add_argument(
        "--sync-overleaf",
        action="store_true",
        help="After rendering, push output/neurips_overleaf_bundle to the Overleaf project via git.",
    )
    p.add_argument(
        "--overleaf-config",
        type=Path,
        default=None,
        help="Overleaf sync config YAML (default: configs/overleaf.local.yaml). "
             "Also honors OVERLEAF_SYNC=1 to auto-enable.",
    )
    p.add_argument(
        "--overleaf-dry-run",
        action="store_true",
        help="Stage files in the Overleaf mirror but do not commit or push.",
    )
    p.add_argument("-q", "--quiet", action="store_true")
    args = p.parse_args()

    def log(msg: str) -> None:
        if not args.quiet:
            print(msg, flush=True)

    from analysis.atlas_challenge.pipeline import run_atlas_baseline
    from analysis.latex_render import render_neurips_rag_atlas_paper, write_neurips_overleaf_bundle
    from analysis.neurips_paper import build_neurips_context, copy_atlas_figures_to_dir
    from analysis.robust_io import (
        RenderValidationError,
        assert_recent,
        atomic_write_text,
        banner,
        summarize,
        validate_rendered_tex,
    )
    from analysis.tex_compile import find_pdflatex, graphicspath_latex, run_pdflatex_twice
    from evals.rag_store import default_collection_name, default_rag_db_path
    from evals.retrieval_eval_lib import MissingEvalDependencies, run_retrieval_evaluation

    pipeline_started_wall = time.time()

    def git_commit_sha(repo: Path) -> str | None:
        try:
            return subprocess.check_output(
                ["git", "-C", str(repo), "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            return None

    atlas_dir = (args.atlas_output_dir or (root / "output" / "atlas_challenge")).resolve()
    metrics_path = atlas_dir / "metrics.json"
    csv_path = args.csv or (root / "data" / "atlas-higgs-challenge-2014-v2.csv")

    t_pipeline = time.perf_counter()

    # --- 1) ATLAS replication ---
    if args.skip_atlas:
        if not metrics_path.is_file():
            print(f"--skip-atlas but missing {metrics_path}", file=sys.stderr)
            return 2
        atlas_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        log(f"[neurips-pipeline] loaded existing atlas metrics from {metrics_path}")
    else:
        if not csv_path.is_file():
            print(f"CSV not found: {csv_path}", file=sys.stderr)
            return 2
        max_rows = None if args.full_train else args.max_rows
        log("[neurips-pipeline] running ATLAS baseline (case study replication)…")
        atlas_metrics = run_atlas_baseline(
            csv_path,
            max_train_rows=max_rows,
            cv_folds=int(args.cv_folds),
            output_dir=atlas_dir,
            verbose=not args.quiet,
        )

    # --- 2) Retrieval eval ---
    eval_aggregate = None
    eval_path: Path | None = None
    eval_skipped = False
    eval_reason = ""
    rag_db = args.rag_db or default_rag_db_path(root)

    if args.reuse_eval_json is not None:
        if not args.reuse_eval_json.is_file():
            print(f"--reuse-eval-json not found: {args.reuse_eval_json}", file=sys.stderr)
            return 2
        eval_aggregate = json.loads(args.reuse_eval_json.read_text(encoding="utf-8"))
        eval_path = args.reuse_eval_json.resolve()
        rag_db_raw = eval_aggregate.get("rag_db")
        if rag_db_raw:
            rag_db = Path(rag_db_raw)
        log(
            f"[neurips-pipeline] reusing eval JSON: {eval_path} "
            f"(n={eval_aggregate.get('n_queries')}, chunks={eval_aggregate.get('rag_chunk_count')})"
        )
    elif args.skip_eval:
        eval_skipped = True
        eval_reason = "Skipped (--skip-eval)."
    elif rag_db is None:
        eval_skipped = True
        eval_reason = "No Chroma DB found; pass --rag-db or index ../rag-ai-scientist/.cursor/rag_db or .cursor/rag_db."
    elif not args.queries.is_file():
        eval_skipped = True
        eval_reason = f"Queries file missing: {args.queries}"
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
        eval_out = root / "evals" / "results" / f"neurips_{ts}.json"
        try:
            collection = args.collection or default_collection_name(root)
            log(f"[neurips-pipeline] running retrieval evaluation (collection='{collection}')…")
            eval_aggregate, eval_path = run_retrieval_evaluation(
                rag_db=rag_db,
                queries_path=args.queries,
                collection_name=collection,
                k_list=list(args.k_list),
                max_k=args.max_k,
                output_path=eval_out,
                embedding_model_id=args.embedding_model,
                enable_ragas=bool(args.enable_ragas or args.ragas),
                ragas_max_queries=(None if int(args.ragas_max_queries) < 0 else int(args.ragas_max_queries)),
            )
            log(
                f"[neurips-pipeline] eval done: n={eval_aggregate.get('n_queries')} queries, "
                f"chunks={eval_aggregate.get('rag_chunk_count')}, "
                f"MRR={eval_aggregate.get('mrr_mean'):.4f}, "
                f"Recall@{max(eval_aggregate.get('k_values', [0]))}="
                f"{eval_aggregate.get('recall_mean', {}).get(str(max(eval_aggregate.get('k_values', [0]))), 0):.4f}"
            )
            log(f"[neurips-pipeline] eval JSON → {eval_path}")
        except MissingEvalDependencies as exc:
            eval_skipped = True
            eval_reason = str(exc).replace("\n", " ")
            log(f"[neurips-pipeline] {eval_reason}")
        except Exception as exc:
            eval_skipped = True
            eval_reason = f"Eval failed: {exc}"
            log(f"[neurips-pipeline] eval failed: {exc}")

    ogts_aggregate: dict | None = None
    ogts_json_display = "---"
    if args.reuse_ogts_json is not None:
        op = args.reuse_ogts_json.resolve()
        if not op.is_file():
            print(f"--reuse-ogts-json not found: {op}", file=sys.stderr)
            return 2
        ogts_aggregate = json.loads(op.read_text(encoding="utf-8"))
        ogts_json_display = str(op)

    rag_display = str(rag_db.resolve()) if rag_db is not None else "(none)"
    # If we reused an eval JSON, prefer the queries path recorded in that artifact so the
    # rendered manuscript matches the evaluated file (not whatever default CLI path was left).
    queries_display = str(args.queries.resolve())
    if isinstance(eval_aggregate, dict):
        qf = eval_aggregate.get("queries_file")
        if isinstance(qf, str) and qf.strip():
            queries_display = qf.strip()

    ctx = build_neurips_context(
        atlas_metrics=atlas_metrics,
        eval_aggregate=eval_aggregate,
        eval_skipped=eval_skipped,
        eval_skip_reason=eval_reason,
        eval_json_path=eval_path,
        rag_db_display=rag_display,
        queries_file_display=queries_display,
        ogts_aggregate=ogts_aggregate,
    )

    # --- 3) LaTeX + figures ---
    paper_root = (args.paper_dir or (root / "output" / "neurips_paper")).resolve()
    overleaf_root = (root / "output" / "neurips_overleaf_bundle").resolve()

    if args.clean_output:
        import shutil

        for victim in (paper_root, overleaf_root):
            if victim.is_dir():
                log(f"[neurips-pipeline] --clean-output: removing {victim}")
                shutil.rmtree(victim, ignore_errors=False)

    fig_dir = paper_root / "figures"
    paper_root.mkdir(parents=True, exist_ok=True)
    copy_atlas_figures_to_dir(atlas_dir, fig_dir)

    gp = graphicspath_latex(paper_root, fig_dir)
    try:
        tex_path = render_neurips_rag_atlas_paper(
            ctx,
            fig_dir,
            paper_root,
            filename="neurips_rag_atlas.tex",
            graphicspath=gp,
        )
        validate_rendered_tex(tex_path, min_bytes=args.min_tex_bytes)
        assert_recent(tex_path, pipeline_started_wall)
    except RenderValidationError as exc:
        print(f"[neurips-pipeline] render validation failed: {exc}", file=sys.stderr)
        return 3
    log(f"[neurips-pipeline] wrote {tex_path}")

    try:
        main_tex = write_neurips_overleaf_bundle(ctx, atlas_dir, overleaf_root)
        validate_rendered_tex(main_tex, min_bytes=args.min_tex_bytes)
        assert_recent(main_tex, pipeline_started_wall)
    except RenderValidationError as exc:
        print(f"[neurips-pipeline] Overleaf bundle validation failed: {exc}", file=sys.stderr)
        return 3
    log(f"[neurips-pipeline] Overleaf bundle → {overleaf_root}/")

    produced = [summarize(tex_path), summarize(main_tex)]
    for stem in ("signal_vs_background_mass_mmc", "roc_validation", "ams_vs_threshold", "feature_importance_top"):
        for base in (fig_dir, overleaf_root / "figures"):
            p = base / f"{stem}.pdf"
            if p.is_file():
                produced.append(summarize(p))

    input_deps = {}
    tpl = root / "paper" / "neurips_rag_atlas.tex.j2"
    if tpl.is_file():
        input_deps["template"] = summarize(tpl).as_dict()
    if eval_path and Path(eval_path).is_file():
        input_deps["eval_json"] = summarize(Path(eval_path)).as_dict()
    if metrics_path.is_file():
        input_deps["atlas_metrics_json"] = summarize(metrics_path).as_dict()
    if args.queries.is_file():
        input_deps["queries_jsonl"] = summarize(args.queries).as_dict()
    if args.reuse_ogts_json is not None and Path(args.reuse_ogts_json).is_file():
        input_deps["ogts_eval_json"] = summarize(Path(args.reuse_ogts_json).resolve()).as_dict()

    manifest = {
        "elapsed_seconds_pipeline": time.perf_counter() - t_pipeline,
        "pipeline_started_iso": datetime.fromtimestamp(pipeline_started_wall, tz=timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "git_commit": git_commit_sha(root),
        "pipeline_argv": sys.argv,
        "queries_path_cli": str(args.queries.resolve()) if args.queries.is_file() else None,
        "ragas_enabled": bool(args.enable_ragas or args.ragas),
        "ragas_max_queries": int(args.ragas_max_queries),
        "reuse_ogts_json": ogts_json_display if ogts_json_display != "---" else None,
        "eval_json_ragas_enabled": (
            bool(eval_aggregate.get("ragas", {}).get("enabled"))
            if isinstance(eval_aggregate, dict)
            else None
        ),
        "atlas_metrics_path": str(metrics_path.resolve()),
        "atlas_output_dir": str(atlas_dir),
        "eval_skipped": eval_skipped,
        "eval_reason": eval_reason,
        "eval_json": str(eval_path) if eval_path else None,
        "eval_available": bool(ctx.get("eval_available")),
        "eval_n_queries": ctx.get("eval_n_queries"),
        "eval_n_failures": ctx.get("eval_n_failures"),
        "eval_mrr": ctx.get("eval_mrr"),
        "neurips_tex": str(tex_path.resolve()),
        "overleaf_bundle": str(overleaf_root.resolve()),
        "rag_db": rag_display,
        "produced": [s.as_dict() for s in produced],
        "inputs": input_deps,
    }
    man_path = paper_root / "pipeline_manifest.json"
    manifest_json = json.dumps(manifest, indent=2)
    atomic_write_text(man_path, manifest_json)
    assert_recent(man_path, pipeline_started_wall)
    log(f"[neurips-pipeline] manifest → {man_path}")

    alt_manifest = root / "output" / "pipeline_manifest.json"
    alt_manifest.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(alt_manifest, manifest_json)
    assert_recent(alt_manifest, pipeline_started_wall)
    log(f"[neurips-pipeline] manifest → {alt_manifest}")

    log("")
    log(banner("neurips-pipeline outputs", produced))

    import os as _os

    sync_requested = args.sync_overleaf or _os.environ.get("OVERLEAF_SYNC") == "1"
    if sync_requested:
        sync_cfg = (args.overleaf_config or (root / "configs" / "overleaf.local.yaml")).resolve()
        log(f"[neurips-pipeline] syncing Overleaf bundle → project (config: {sync_cfg.name})")
        import subprocess as _sub

        cmd = [sys.executable, str(root / "scripts" / "sync_overleaf.py"), "--config", str(sync_cfg)]
        if args.overleaf_dry_run:
            cmd.append("--dry-run")
        if args.quiet:
            cmd.append("--quiet")
        try:
            _sub.run(cmd, check=True)
        except _sub.CalledProcessError as exc:
            print(
                f"[neurips-pipeline] Overleaf sync failed (rc={exc.returncode}). "
                "Paper files are still available in output/neurips_overleaf_bundle/.",
                file=sys.stderr,
            )
            # Do not abort the whole pipeline over Overleaf-only failure.

    if args.no_compile:
        print(
            "\n[neurips-pipeline] PDF skipped (--no-compile). "
            "Use Overleaf or install pdflatex.\n",
            file=sys.stderr,
        )
        return 0

    pdflatex = find_pdflatex()
    if not pdflatex:
        print(
            "\n[neurips-pipeline] pdflatex not found; Overleaf bundle is ready.\n",
            file=sys.stderr,
        )
        return 0

    log(f"[neurips-pipeline] pdflatex: {pdflatex}")
    r = run_pdflatex_twice(tex_path.parent, tex_path.name)
    if r is None or r.returncode != 0:
        if r and r.stdout:
            print(r.stdout[-2500:], file=sys.stderr)
        print("[neurips-pipeline] pdflatex failed on neurips draft", file=sys.stderr)
        return r.returncode if r else 1

    pdf_path = tex_path.with_suffix(".pdf")
    if pdf_path.is_file():
        log(f"[neurips-pipeline] PDF: {pdf_path} ({pdf_path.stat().st_size:,} bytes)")

    r2 = run_pdflatex_twice(main_tex.parent, main_tex.name)
    if r2 and r2.returncode == 0 and main_tex.with_suffix(".pdf").is_file():
        log(f"[neurips-pipeline] PDF (Overleaf layout): {main_tex.with_suffix('.pdf')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
