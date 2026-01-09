"""
awrextractor_clean.py
AWR-Miner text file extractor → pandas DataFrames → CSV → RAG pipeline

Recommended AWR Miner script version: 4.0.0+

Author: Irvansyah (Cunkrink)
License: GPLv3
"""

import re
import os
import sys
import argparse
from typing import List, Tuple, Optional
import pandas as pd
import pyarrow as pa

try:
    import openpyxl
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False


# ============================================================
# Metadata Normalizer
# ============================================================

def normalize_metadata(meta):
    new_meta = {}
    for k, v in meta.items():
        if isinstance(v, list):
            new_meta[k] = [str(x) for x in v]
        elif isinstance(v, (int, float, bool)):
            new_meta[k] = str(v)
        elif v is None:
            new_meta[k] = ""
        else:
            new_meta[k] = v
    return new_meta


# ============================================================
# RAG Pipeline Runner
# ============================================================

def rag_run_all(
    os_info,
    os_memory,
    main_metric,
    aas,
    top_wait,
    top_sql,
    redis_url="redis://localhost:6379",
    index_name="awr_index",
    llm_base_url="http://localhost:1235/v1",
    llm_model="meta-llama-3.1-8b-instruct",
    save_path=None,
    analysis_level="technical",
    recommendation_level="medium",
    language="id",
    style="hybrid",
):
    """
    Full pipeline:
    1. Build all documents
    2. Ingest into Redis
    3. Auto-detect SNAP_ID range
    4. Run RAG range report
    5. Save report
    """

    from awr_rag import (
        create_llm,
        create_embeddings,
        create_redis_vectorstore,
        build_os_info_doc,
        build_snapshot_superdocs_with_time,
        build_hourly_superdocs,
        upsert_documents_to_redis,
        create_range_report_chain,
        normalize_doc,
    )

    print("🔄 Membuat embedding model...")
    embeddings = create_embeddings()

    print("🔄 Membuat vectorstore Redis...")
    vectorstore = create_redis_vectorstore(redis_url, index_name, embeddings)

    print("🔄 Membuat LLM...")
    llm = create_llm(
        base_url=llm_base_url,
        model=llm_model,
        api_key="lm-studio",
        temperature=0.1,
    )

    print("📄 Membuat dokumen superdoc...")
    docs = []

    if os_info is not None:
        docs.append(build_os_info_doc(os_info))

    if all(x is not None for x in [os_info, os_memory, main_metric, aas, top_wait, top_sql]):
        docs.extend(build_snapshot_superdocs_with_time(
            os_info, os_memory, main_metric, aas, top_wait, top_sql
        ))
        docs.extend(build_hourly_superdocs(
            os_info, os_memory, main_metric, aas, top_wait
        ))

    docs = [normalize_doc(d) for d in docs]

    print("🔍 Validating documents...")
    bad_docs = []
    for i, d in enumerate(docs):
        if not isinstance(d, dict):
            bad_docs.append((i, type(d), "not a dict"))
            continue
        if "content" not in d:
            bad_docs.append((i, d.keys(), "missing content"))
        if "metadata" not in d:
            bad_docs.append((i, d.keys(), "missing metadata"))

    if bad_docs:
        print("\n❌ Found invalid documents:")
        for item in bad_docs[:10]:
            print(" - Index:", item[0], "| Keys:", item[1], "| Issue:", item[2])
        raise ValueError("Document structure invalid.")
    else:
        print("✅ All documents valid.")

    for d in docs:
        d["metadata"] = normalize_metadata(d["metadata"])

    print("📥 Ingesting documents into Redis...")
    upsert_documents_to_redis(vectorstore, docs)

    # Auto SNAP range
    snap_col = None
    for c in ["SNAP_ID", "snap", "Snap", "snap_id"]:
        if c in main_metric.columns:
            snap_col = c
            break

    if snap_col is None:
        raise ValueError("main_metric tidak memiliki kolom SNAP_ID atau snap.")

    start_snap = int(main_metric[snap_col].min())
    end_snap = int(main_metric[snap_col].max())

    print(f"📌 Auto SNAP range: {start_snap} → {end_snap}")

    print("🤖 Menjalankan RAG range report...")
    chain = create_range_report_chain(
        llm,
        vectorstore,
        start_snap,
        end_snap,
        analysis_level=analysis_level,
        recommendation_level=recommendation_level,
        language=language,
        style=style,
    )

    start_time = main_metric["start"].min()
    end_time = main_metric["end"].max()

    # Environment summary
    env = {}
    if os_info is not None:
        for _, row in os_info.iterrows():
            env[row["STAT_NAME"]] = row["STAT_VALUE"]

    db_name = env.get("DB_NAME", "Unknown")
    dbid = env.get("DBID", "Unknown")
    platform = env.get("!PLATFORM_NAME", "Unknown")
    version = env.get("VERSION", "Unknown")
    num_cpus = env.get("NUM_CPUS", "Unknown")
    num_cpu_cores = env.get("NUM_CPU_CORES", "Unknown")
    num_cpu_sockets = env.get("NUM_CPU_SOCKETS", "Unknown")
    physical_mem = env.get("PHYSICAL_MEMORY_GB", "Unknown")
    instances = env.get("INSTANCES", "Unknown")

    header_text = f"""
============================================================
AWR PERFORMANCE REPORT
Database: {db_name} (DBID {dbid})
Platform: {platform}
Oracle Version: {version}
CPU: {num_cpu_cores} cores / {num_cpus} threads / {num_cpu_sockets} sockets
Memory: {physical_mem} GB
Instances: {instances}

Waktu: {start_time} → {end_time}
SNAP_ID: {start_snap} → {end_snap}
============================================================
"""

    query_text = (
        f"Provide a detailed AWR performance analysis for SNAP_ID range "
        f"{start_snap} to {end_snap}. Summarize CPU, wait events, AAS, I/O, "
        f"Top SQL, and plan changes."
    )

    report = chain.invoke({"query": query_text})

    final_report = f"""

{header_text}

{report}

============================================================
END OF REPORT
============================================================
"""

    # Output filename
    start_ts = start_time.strftime("%Y%m%d-%H%M")
    end_ts = end_time.strftime("%Y%m%d-%H%M")
    default_filename = f"{db_name}_{dbid}_{start_ts}_{end_ts}_awr_rag_output.txt"

    if save_path:
        if os.path.isdir(save_path):
            save_path = os.path.join(save_path, default_filename)
        else:
            folder = os.path.dirname(save_path)
            if folder == "":
                save_path = os.path.join(".", default_filename)

        folder = os.path.dirname(save_path)
        if folder:
            os.makedirs(folder, exist_ok=True)

        with open(save_path, "w", encoding="utf-8") as f:
            f.write(final_report)

        print(f"💾 Report saved to: {save_path}")

    print(final_report)
    return

# ============================================================
# BEGIN/END Block Detection
# ============================================================

def find_blocks(lines: List[str]) -> List[Tuple[str, int, int]]:
    """
    Return list of (section_name, begin_idx, end_idx).
    begin_idx = line after BEGIN marker
    end_idx   = line of END marker (exclusive)
    """
    blocks = []
    begin_re = re.compile(r"~~BEGIN-(.+?)~~")
    end_re = re.compile(r"~~END-(.+?)~~")
    current = None

    for i, ln in enumerate(lines):
        m = begin_re.search(ln)
        if m:
            name = m.group(1).strip()
            current = (name, i + 1)
            continue

        m2 = end_re.search(ln)
        if m2 and current is not None:
            begin_name, begin_idx = current
            blocks.append((begin_name, begin_idx, i))
            current = None

    return blocks


# ============================================================
# Dash-Line Column Slice Detection
# ============================================================

def compute_slices_from_dash_line(dash_line: str) -> List[Tuple[int, int]]:
    matches = list(re.finditer(r"-+", dash_line))
    if not matches:
        return []

    slices = []
    for idx, m in enumerate(matches):
        start = m.start()
        if idx + 1 < len(matches):
            end = matches[idx + 1].start()
        else:
            end = len(dash_line)
        slices.append((start, end))

    return slices


def slice_line_by_slices(line: str, slices: List[Tuple[int, int]]) -> List[str]:
    return [line[s:e].strip() for (s, e) in slices]


# ============================================================
# Block → DataFrame Parser
# ============================================================

def parse_block_to_df(block_lines: List[str]) -> Optional[pd.DataFrame]:
    """
    Parse a single block (between BEGIN/END).
    Supports dash-delimited column boundaries.
    """
    start_idx = 0
    n = len(block_lines)

    while start_idx < n and block_lines[start_idx].strip() == "":
        start_idx += 1
    if start_idx >= n:
        return None

    dash_idx = None
    for j in range(start_idx + 1, n):
        if re.search(r"-{2,}", block_lines[j]):
            dash_idx = j
            break

    if dash_idx:
        header_block = block_lines[start_idx:dash_idx]
        dash_line = block_lines[dash_idx].rstrip("\n")
        data_start = dash_idx + 1
    else:
        header_block = [block_lines[start_idx].rstrip("\n")]
        dash_line = None
        data_start = start_idx + 1

    header_line = " ".join([ln.rstrip("\n") for ln in header_block])

    data_lines = []
    for j in range(data_start, n):
        ln = block_lines[j].rstrip("\n")
        if ln.strip():
            data_lines.append(ln)

    if dash_line:
        slices = compute_slices_from_dash_line(dash_line)
        if slices:
            headers = [h.strip() for h in slice_line_by_slices(header_line, slices)]
            rows = []
            for ln in data_lines:
                cells = slice_line_by_slices(ln, slices)
                if len(cells) != len(headers):
                    cells = [c for c in re.split(r"\s+", ln.strip()) if c]
                rows.append(cells)

            rows = [r for r in rows if any(x != "" for x in r)]
            try:
                df = pd.DataFrame(rows, columns=headers)
            except Exception:
                df = pd.DataFrame(rows)
            return df

    headers = [h for h in re.split(r"\s+", header_line.strip()) if h]
    rows = []
    for ln in data_lines:
        cells = [c for c in re.split(r"\s+", ln.strip()) if c]
        rows.append(cells)

    if not rows:
        return None

    if all(len(r) == len(headers) for r in rows):
        return pd.DataFrame(rows, columns=headers)
    else:
        return pd.DataFrame(rows)


# ============================================================
# File Parser
# ============================================================

def parse_file_to_dfs(filename: str) -> dict:
    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    blocks = find_blocks(lines)
    result = {}

    for (name, begin_idx, end_idx) in blocks:
        block_lines = [ln for ln in lines[begin_idx:end_idx]]
        df = parse_block_to_df(block_lines)
        result[name.strip()] = df

    return result


# ============================================================
# Name Sanitizer
# ============================================================

def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "-", name)

# ============================================================
# Main CLI Entry Point
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", metavar="INPUT",
                   help="Input AWR Miner text file")
    p.add_argument("--section", "-s", help="Only extract this named section")
    p.add_argument("--outdir", "-o", default="out_sections")
    p.add_argument("--csv", action="store_true", help="Write CSV for extracted sections")
    p.add_argument("--csv-all", action="store_true", help="Write CSV for all sections")
    p.add_argument("--excel", action="store_true", help="Write all sections to a single Excel file")
    p.add_argument("--excel-filename", default="awr_extracted_sections.xlsx")
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument("--rag-ingest", action="store_true")
    p.add_argument("--rag-report-snap", nargs=1)
    p.add_argument("--instance", type=int)
    p.add_argument("--rag-report-range", nargs=2)
    p.add_argument("--rag-ask", nargs="+")
    p.add_argument("--rag-report-all", action="store_true")
    p.add_argument("--rag-run-all", action="store_true")
    p.add_argument("--save", default="out_sections")
    p.add_argument("--analysis-level", default="technical",
                   choices=["executive", "technical", "deepdive"])
    p.add_argument("--recommendation-level", default="medium",
                   choices=["high", "medium", "expert"])
    p.add_argument("--language", default="id", choices=["id", "en"])
    p.add_argument("--style", default="hybrid", choices=["hybrid"])
    p.add_argument("--preset",
                   choices=["manager", "dba", "expert", "balanced", "english"])

    args = p.parse_args()

    analysis_level = args.analysis_level
    recommendation_level = args.recommendation_level
    language = args.language
    style = args.style

    # Apply preset
    if args.preset == "manager":
        analysis_level = "executive"
        recommendation_level = "high"
        language = "id"
        style = "hybrid"
    elif args.preset == "dba":
        analysis_level = "technical"
        recommendation_level = "medium"
        language = "id"
        style = "hybrid"
    elif args.preset == "expert":
        analysis_level = "deepdive"
        recommendation_level = "expert"
        language = "id"
        style = "hybrid"
    elif args.preset == "balanced":
        analysis_level = "technical"
        recommendation_level = "medium"
        language = "id"
        style = "hybrid"
    elif args.preset == "english":
        language = "en"

    # Resolve input path
    raw_input = os.path.expanduser(args.input)
    if os.path.isabs(raw_input) or os.path.dirname(raw_input):
        input_path = os.path.abspath(raw_input)
    else:
        input_path = os.path.join(os.getcwd(), raw_input)

    # Parse file
    try:
        dfs = parse_file_to_dfs(input_path)
        os_info = dfs.get("OS-INFORMATION")
        os_memory = dfs.get("MEMORY")
        main_metric = dfs.get("MAIN-METRICS")
        aas = dfs.get("AVERAGE-ACTIVE-SESSIONS")
        top_wait = dfs.get("TOP-N-TIMED-EVENTS")
        top_sql = dfs.get("TOP-SQL-BY-SNAPID")

        # MAIN-METRICS type conversion
        if main_metric is not None:
            col_map = {
                "SNAP_ID": "snap",
                "snap": "snap",
                "DUR_M": "dur_m",
                "dur_m": "dur_m",
                "END": "end",
                "end": "end",
            }
            main_metric.rename(
                columns={k: v for k, v in col_map.items() if k in main_metric.columns},
                inplace=True,
            )

            main_metric["snap"] = pd.to_numeric(main_metric["snap"], errors="coerce").astype("Int64")
            main_metric["dur_m"] = pd.to_numeric(main_metric["dur_m"], errors="coerce").astype("Int64")

            main_metric["end"] = pd.to_datetime(
                main_metric["end"],
                format="%y/%m/%d %H:%M",
                errors="coerce",
            )

            main_metric["start"] = main_metric["end"] - pd.to_timedelta(main_metric["dur_m"], unit="m")

    except FileNotFoundError:
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"Error reading input file '{input_path}': {e}", file=sys.stderr)
        sys.exit(1)

    if not dfs:
        print(f"Error: No sections found in file: {input_path}", file=sys.stderr)
        sys.exit(3)

    if args.section:
        keys = [k for k in dfs.keys() if k.upper() == args.section.upper()]
        if not keys:
            print(f"Section not found: {args.section}")
            return
    else:
        keys = list(dfs.keys())

    os.makedirs(args.outdir, exist_ok=True)

    # Export loop
    for name in keys:
        df = dfs.get(name)
        if df is None:
            continue

        if args.verbose:
            print("\n" + "=" * 60)
            print(f"Section: {name}  (rows={df.shape[0] if df is not None else 0})")
            with pd.option_context("display.max_rows", 10, "display.max_columns", 20):
                print(df.head(10).to_string(index=False))

        if args.csv or args.csv_all:
            try:
                outname = os.path.join(args.outdir, "section_" + sanitize_name(name) + ".csv")
                df.to_csv(outname, index=False)
                print(f"Wrote CSV: {outname}")
            except Exception as e:
                print(f"Failed to write CSV for {name}: {e}")

    # RAG ingestion only
    if args.rag_ingest:
        from awr_rag import (
            create_embeddings,
            create_redis_vectorstore,
            build_os_info_doc,
            build_snapshot_superdocs_with_time,
            build_hourly_superdocs,
            upsert_documents_to_redis,
        )

        emb = create_embeddings()
        vs = create_redis_vectorstore("redis://localhost:6379", "awr_index", emb)

        docs = []
        if os_info is not None:
            docs.append(build_os_info_doc(os_info))
        if all(x is not None for x in [os_info, os_memory, main_metric, aas, top_wait, top_sql]):
            docs.extend(build_snapshot_superdocs_with_time(os_info, os_memory, main_metric, aas, top_wait, top_sql))
            docs.extend(build_hourly_superdocs(os_info, os_memory, main_metric, aas, top_wait))

        print(f"Ingesting {len(docs)} documents into Redis...")
        upsert_documents_to_redis(vs, docs)
        print("RAG ingestion completed.")
        return

    # Full RAG pipeline
    if args.rag_run_all:
        rag_run_all(
            os_info,
            os_memory,
            main_metric,
            aas,
            top_wait,
            top_sql,
            save_path=args.save,
            analysis_level=analysis_level,
            recommendation_level=recommendation_level,
            language=language,
            style=style,
        )
        return


if __name__ == "__main__":
    main()