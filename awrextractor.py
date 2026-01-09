"""
awrtest2.py - AWR-Miner text file extractor to pandas DataFrames and CSVs
recomended awr miner script version 4.0.0+

Copyright (C) 2025 Irvansyah(Cunkrink)

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of  MERCHANTABILITY or FITNESS FOR
A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program.  If not, see <http://www.gnu.org/licenses/>.


======================================================================================================
Extract every ~~BEGIN-...~~ / ~~END-...~~ block from an AWR-like text file
and convert blocks that contain dash-delimited column lines into
pandas.DataFrame objects. The first dash row is used to compute column
boundaries and then removed from data.

Usage examples:
    py .\awrtest2.py awr-hist-1738933432-NAKULA-3366-3564.out --section SGA --outdir out_all --csv
    py .\awrtest2.py awr-hist-1738933432-NAKULA-3366-3564.out --csv-all --outdir out_all

This script follows the spirit of `getdata` in `pandas_test.py` but with
more robust dash-line column boundary detection and CSV export support.

Requirements:
Python 3.13.5+
Pandas 2.2.4+
redis-py 4.5.5+
PyArrow 12.0.0+
Orange3 3.34.0+
openpyxl 3.1.2+ (only if --excel flag is used)

todo:
- populate data for analysis databese performance tuning:
   * Cpu usage
   * average active sessions
   * db time
   * top sql by cpu
   * Wait events
   * I/O stats
   * logswitches
   * Memory usage
   * capacity planning
======================================================================================================
"""

from email import parser
from itertools import chain
import re
import os
import sys
import argparse
from typing import List, Tuple, Optional
import pandas as pd
import pyarrow as pa
import torch
try:
    import openpyxl
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False


def normalize_metadata(meta):
    new_meta = {}
    for k, v in meta.items():
        # Convert list → list of strings
        if isinstance(v, list):
            new_meta[k] = [str(x) for x in v]
        # Convert int/float/bool → string
        elif isinstance(v, (int, float, bool)):
            new_meta[k] = str(v)
        # None → empty string
        elif v is None:
            new_meta[k] = ""
        else:
            new_meta[k] = v
    return new_meta

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
    2. Ingest into Redis (with progress bar)
    3. Auto-detect SNAP_ID range
    4. Run RAG range report
    5. Optionally save report to file
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
        upsert_documents_to_redis_parallel,
        normalize_doc
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
    #print("DEBUG LLM:", llm)


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
        for item in bad_docs[:10]:  # tampilkan 10 dulu
            print(" - Index:", item[0], "| Keys:", item[1], "| Issue:", item[2])
        raise ValueError("Document structure invalid. Fix required.")
    else:
        print("✅ All documents valid.")

    for d in docs:
        d["metadata"] = normalize_metadata(d["metadata"])
    
    #for d in docs[:20]:
    #    print("DEBUG metadata:", d["metadata"])


    # INGESTION WITH PROGRESS BAR
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    upsert_documents_to_redis(vectorstore, docs)
    
    
    #parallel ingestion (faster?)
    #upsert_documents_to_redis_parallel(
    #vectorstore,
    #docs,
    #embeddings,
    #workers=4,        # jumlah CPU worker
    #batch_size=5     # batch kecil = lebih cepat
    #)

    # Auto SNAP range
    
    snap_col = None
    for c in ["SNAP_ID", "snap", "Snap", "snap_id"]:
        if c in main_metric.columns:
            snap_col = c
            break

    if snap_col is None:
        raise ValueError(f"main_metric tidak memiliki kolom SNAP_ID atau snap. Kolom tersedia: {main_metric.columns.tolist()}")
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
    #report = chain.invoke(None)
    # ============================
    # 🔧 HEADER LAPORAN DENGAN TIMESTAMP
    # ============================

    start_time = main_metric["start"].min()
    end_time = main_metric["end"].max()
    # ============================
    # 🔧 ENVIRONMENT SUMMARY (OS-INFORMATION)
    # ============================
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
    #print(header_text)
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



    # SAVE TO FILE IF REQUESTED
    # ============================
    # 🔧 BUILD OUTPUT FILENAME (CUSTOM FORMAT)
    # ============================

    # Ambil environment info dari OS-INFORMATION
    env = {}
    if os_info is not None:
        for _, row in os_info.iterrows():
            env[row["STAT_NAME"]] = row["STAT_VALUE"]

    db_name = env.get("DB_NAME", "UNKNOWN")
    dbid = env.get("DBID", "UNKNOWN")

    # Format timestamp untuk nama file
    start_ts = start_time.strftime("%Y%m%d-%H%M")
    end_ts = end_time.strftime("%Y%m%d-%H%M")

    # Nama file sesuai preferensi user
    default_filename = f"{db_name}_{dbid}_{start_ts}_{end_ts}_awr_rag_output.txt"
    if save_path:
        # Jika save_path adalah folder → gunakan default_filename
        if os.path.isdir(save_path):
            save_path = os.path.join(save_path, default_filename)

        # Jika save_path adalah file → gunakan apa adanya
        else:
            folder = os.path.dirname(save_path)
            if folder == "":
                save_path = os.path.join(".", default_filename)

        # Pastikan folder ada
        folder = os.path.dirname(save_path)
        if folder:
            os.makedirs(folder, exist_ok=True)

        with open(save_path, "w", encoding="utf-8") as f:
            f.write(final_report)

        print(f"💾 Report saved to: {save_path}")

    #print("\n================= REPORT OUTPUT =================\n")
    print(final_report)
    #print("\n=================================================\n")

    # Jangan return report (supaya tidak muncul None)
    return



def find_blocks(lines: List[str]) -> List[Tuple[str, int, int]]:
    """Return list of (section_name, begin_idx, end_idx) in lines.
    begin_idx points to the line after the BEGIN marker, end_idx points to
    the line of the END marker (exclusive).
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
            name2 = m2.group(1).strip()
            # Only close block if names match (very likely), otherwise still close
            begin_name, begin_idx = current
            blocks.append((begin_name, begin_idx, i))
            current = None
    return blocks


def compute_slices_from_dash_line(dash_line: str) -> List[Tuple[int, int]]:
    """Compute column slices (start,end) using runs of '-' in dash_line.
    Return list of (start, end) pairs suitable for slicing text lines.
    """
    matches = list(re.finditer(r"-+", dash_line))
    if not matches:
        return []
    starts = [m.start() for m in matches]
    # make ends go to next start (so we include the whitespace gap)
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


def parse_block_to_df(block_lines: List[str]) -> Optional[pd.DataFrame]:
    """Parse a single block (lines between BEGIN and END).

    Expect the first non-empty line to be header, the next non-empty line may
    be a dash-line which we use to compute column boundaries. If a dash-line
    exists, we remove it from output and parse subsequent lines using the
    computed slices. If no dash-line is found, fall back to whitespace split.
    """
    # strip leading empty lines and find start
    start_idx = 0
    n = len(block_lines)
    while start_idx < n and block_lines[start_idx].strip() == "":
        start_idx += 1
    if start_idx >= n:
        return None

    # find dash line index (if any) after the first non-empty line
    dash_idx = None
    for j in range(start_idx + 1, n):
        if re.search(r"-{2,}", block_lines[j]):
            dash_idx = j
            break

    # Support multi-line headers: header area is from start_idx up to dash_idx (if present)
    if dash_idx:
        header_block = block_lines[start_idx:dash_idx]
        dash_line = block_lines[dash_idx].rstrip('\n')
        data_start = dash_idx + 1
    else:
        # header is the first non-empty line only
        header_block = [block_lines[start_idx].rstrip('\n')]
        dash_line = None
        data_start = start_idx + 1

    # build a single header line by joining header block lines (preserves multi-line headers)
    header_line = " ".join([ln.rstrip('\n') for ln in header_block])

    data_lines = []
    for j in range(data_start, n):
        ln = block_lines[j].rstrip('\n')
        if ln.strip() == "":
            continue
        data_lines.append(ln)

    # If we have a dash line, compute slices
    if dash_line:
        slices = compute_slices_from_dash_line(dash_line)
        if slices:
            # Derive header names by slicing the header_line with same slices
            headers = [h.strip() for h in slice_line_by_slices(header_line, slices)]
            rows = []
            for ln in data_lines:
                cells = slice_line_by_slices(ln, slices)
                # if number of cells differs from headers, fallback to whitespace
                if len(cells) != len(headers):
                    cells = [c for c in re.split(r"\s+", ln.strip()) if c != ""]
                rows.append(cells)
            # remove rows that are completely empty
            rows = [r for r in rows if any(x != "" for x in r)]
            try:
                df = pd.DataFrame(rows, columns=headers)
            except Exception:
                # If columns mismatch, create DF without headers and return raw
                df = pd.DataFrame(rows)
            # Try to coerce numeric columns conservatively
            df = coerce_column_types(df)
            return df

    # No dash line -> fallback: header and whitespace-split data
    headers = [h for h in re.split(r"\s+", header_line.strip()) if h != ""]
    rows = []
    for ln in data_lines:
        cells = [c for c in re.split(r"\s+", ln.strip()) if c != ""]
        rows.append(cells)
    if not rows:
        return None
    # If row lengths match headers, use them; otherwise return DataFrame without header
    if all(len(r) == len(headers) for r in rows):
        df = pd.DataFrame(rows, columns=headers)
        df = coerce_column_types(df)
        return df
    else:
        df = pd.DataFrame(rows)
        df = coerce_column_types(df)
        return df


def coerce_column_types(df: pd.DataFrame) -> pd.DataFrame:
    """Conservatively coerce columns to numeric when a majority of values
    successfully convert. Cleans common thousand separators and percent signs.
    Returns the DataFrame with converted dtypes where appropriate.
    """
    if df is None or df.shape[0] == 0:
        return df
    for col in list(df.columns):
        # operate on stringified values
        ser = df[col].astype(str).str.strip()
        # normalize common thousand separators and percent
        cleaned = ser.str.replace(r",", "", regex=True).str.replace(r"%", "", regex=True)
        # treat empty strings as NaN
        cleaned = cleaned.replace({'': None})
        coerced = pd.to_numeric(cleaned, errors='coerce')
        non_na = coerced.notna().sum()
        # if more than half of rows convert to numeric, keep numeric type
        #if non_na >= max(1, int(0.5 * len(df))):
        #    df[col] = coerced
    return df


def parse_file_to_dfs(filename: str) -> dict:
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    blocks = find_blocks(lines)
    result = {}
    for (name, begin_idx, end_idx) in blocks:
        block_lines = [ln for ln in lines[begin_idx:end_idx]]
        df = parse_block_to_df(block_lines)
        key = name.strip()
        result[key] = df
    return result


def try_parse_first_col_datetime(df: pd.DataFrame):
    if df is None or df.shape[0] == 0:
        return df
    # Try to find first column name and parse it
    first_col = df.columns[0]
    # Only attempt if values look like datetimes (contain '/' or ':' etc)
    sample = df[first_col].astype(str).head(10).str.strip()
    if sample.str.contains(r"\d{1,2}/\d{1,2}/\d{2,4}").any() or sample.str.contains(":").any():
        raw_col = f"{first_col}_raw"
        df[raw_col] = df[first_col].astype(str)
        df[first_col] = pd.to_datetime(df[raw_col], format='%y/%m/%d %H:%M', errors='coerce')
        mask = df[first_col].isna()
        if mask.any():
            df.loc[mask, first_col] = pd.to_datetime(df.loc[mask, raw_col], errors='coerce')
    return df


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "-", name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('input', metavar='INPUT',
                   help='Input file path or name (required). If a path is provided it will be used; otherwise the filename is resolved relative to the current working directory.')
    p.add_argument('--section', '-s', help='Only extract this named section')
    p.add_argument('--outdir', '-o', default='out_sections')
    p.add_argument('--csv', action='store_true', help='Write CSV for extracted sections')
    p.add_argument('--csv-all', action='store_true', help='Write CSV for all sections')
    p.add_argument('--excel', action='store_true', help='Write all sections to a single Excel file')
    p.add_argument('--excel-filename', default='awr_extracted_sections.xlsx', help='Excel output filename (default: awr_extracted_sections.xlsx)')
    p.add_argument('--verbose', '-v',  action='store_true', help='More detialed output')
    p.add_argument('--rag-ingest', action='store_true', help='Ingest parsed DataFrames into Redis for RAG')
    p.add_argument('--rag-report-snap', nargs=1, help='Generate RAG report for a specific SNAP_ID (after ingestion)')
    p.add_argument('--instance', type=int, help='Instance number for snapshot report')
    p.add_argument('--rag-report-range', nargs=2, help='Generate RAG report for a range of SNAP_ID (after ingestion)')
    p.add_argument('--rag-ask', nargs='+', help='Ask any question to the RAG system (after ingestion)')
    p.add_argument('--rag-report-all', action='store_true',
              help='Generate RAG report for the entire file (auto min/max SNAP_ID) (after ingestion)')
    p.add_argument('--rag-run-all', action='store_true',
              help='Parse → ingest → generate full-range RAG report in one execution')
    p.add_argument('--save', default='out_sections', help='Save RAG output to a text file')
    p.add_argument("--analysis-level", default="technical",
                        choices=["executive", "technical", "deepdive"])
    p.add_argument("--recommendation-level", default="medium",
                        choices=["high", "medium", "expert"])
    p.add_argument("--language", default="id",
                        choices=["id", "en"])
    p.add_argument("--style", default="hybrid",
                    choices=["hybrid"])
    p.add_argument(
        "--preset",
        choices=["manager", "dba", "expert", "balanced", "english"],
        help="Gunakan preset konfigurasi prompting"
    )


    args = p.parse_args()
    analysis_level = args.analysis_level
    recommendation_level = args.recommendation_level
    language = args.language
    style = args.style
    # Apply preset if provided
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

    pd.set_option('future.no_silent_downcasting', True)
    # Accept either a full/relative path or a filename. Expand user and
    # resolve to absolute path. If a plain filename is provided, resolve
    # it relative to the current working directory.
    raw_input = os.path.expanduser(args.input)
    if os.path.isabs(raw_input) or os.path.dirname(raw_input):
        input_path = os.path.abspath(raw_input)
    else:
        input_path = os.path.join(os.getcwd(), raw_input)

    # Read and validate input file
    try:
        dfs = parse_file_to_dfs(input_path)
        os_info   = dfs.get("OS-INFORMATION")
        os_memory = dfs.get("MEMORY")
        main_metric = dfs.get("MAIN-METRICS")
        aas = dfs.get("AVERAGE-ACTIVE-SESSIONS")
        top_wait = dfs.get("TOP-N-TIMED-EVENTS")
        top_sql = dfs.get("TOP-SQL-BY-SNAPID")
        # ============================
        # 🔧 KONVERSI TIPE DATA MAIN-METRICS
        # ============================

        if main_metric is not None:
        
            # Normalisasi nama kolom
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
            # Konversi tipe data
            main_metric["snap"] = pd.to_numeric(main_metric["snap"], errors="coerce").astype("Int64")
            main_metric["dur_m"] = pd.to_numeric(main_metric["dur_m"], errors="coerce").astype("Int64")

            # Konversi end → datetime
            main_metric["end"] = pd.to_datetime(
                main_metric["end"],
                format="%y/%m/%d %H:%M",
                errors="coerce",
            )
            # Hitung start_time
            main_metric["start"] = main_metric["end"] - pd.to_timedelta(main_metric["dur_m"], unit="m")
    except FileNotFoundError:
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"Error reading input file '{input_path}': {e}", file=sys.stderr)
        sys.exit(1)

    # If parser found no BEGIN/END blocks, fail fast with a clear message
    if not dfs:
        print(f"Error: No sections found in file: {input_path}", file=sys.stderr)
        sys.exit(3)
    if args.section:
        keys = [k for k in dfs.keys() if k.upper() == args.section.upper()]
        if not keys:
            print(f"Section not found: {args.section}")
            return
        keys = keys
    else:
        keys = list(dfs.keys())

    os.makedirs(args.outdir, exist_ok=True)
    #outname = os.path.join(args.outdir, 'awr_extracted_sections.csv')    
      
    # export loop
    for name in keys:
        df = dfs.get(name)
        if df is None:
            print("(no parsed table for this section)")
            continue

        # Try parsing first column datetimes conservatively
        #df = try_parse_first_col_datetime(df)

        # ====get dbname and db id and number of cpus
        if name == "OS-INFORMATION":
            #dbname = df.loc[10, 'STAT_VALUE']
            dbname = df[df['STAT_NAME'] == 'DB_NAME']['STAT_VALUE'].values[0]
            dbid = df[df['STAT_NAME'] == 'DBID']['STAT_VALUE'].values[0]
            num_core = df[df['STAT_NAME'] == '!CPU_COUNT']['STAT_VALUE'].values[0]
            num_cpus = df[df['STAT_NAME'] == 'NUM_CPUS']['STAT_VALUE'].values[0]
            print(f"Number of CPUs (Core/Thread): {num_core}/{num_cpus}")
            print(f"Database Name/ID: {dbname}/{dbid}")
    
        # =====Show a small preview
        if args.verbose:
            print('\n' + '='*60)
            print(f"Section: {name}  (rows={(0 if df is None else df.shape[0])})")
            with pd.option_context('display.max_rows', 10, 'display.max_columns', 20):
                print(df.head(10).to_string(index=False))


        # =====Write CSVs
        if args.csv or args.csv_all:
            
            try:
                outname = os.path.join(args.outdir, 'section_' + sanitize_name(name) + '.csv')
                df.to_csv(outname, index=False)                
                print(f"Wrote CSV: {outname}")

            except Exception as e:
                print(f"Failed to write CSV for {name}: {e}")
    
    # ===== end of export loop

    # ======RAG ingestion (if requested)
    if args.rag_ingest:
        from awr_rag import (
            create_llm, create_embeddings, create_redis_vectorstore,
            build_os_info_doc, build_snapshot_superdocs_with_time,
            build_hourly_superdocs, upsert_documents_to_redis
        )

        emb = create_embeddings()
        vs = create_redis_vectorstore("redis://localhost:6379", "awr_index", emb)

        docs = []
        if os_info is not None:
            docs.append(build_os_info_doc(os_info))
        if all(x is not None for x in [os_info, os_memory, main_metric, aas, top_wait, top_sql]):
            snap_docs = build_snapshot_superdocs_with_time(os_info, os_memory, main_metric, aas, top_wait, top_sql)
            docs.extend(snap_docs)
            hour_docs = build_hourly_superdocs(os_info, os_memory, main_metric, aas, top_wait)
            docs.extend(hour_docs)

        print(f"Ingesting {len(docs)} documents into Redis...")
        upsert_documents_to_redis(vs, docs)
        print("RAG ingestion completed.")
        return
    # end RAG ingestion

    # ======RAG question report snap (if requested)
    if args.rag_report_snap:
        from awr_rag import create_llm, create_embeddings, create_redis_vectorstore, create_snapshot_report_chain

        snap_id = int(args.rag_report_snap[0])
        instance = args.instance

        llm = create_llm()
        emb = create_embeddings()
        vs = create_redis_vectorstore("redis://localhost:6379", "awr_index", emb)

        chain = create_snapshot_report_chain(llm, vs, snap_id, instance)
        report = chain.invoke(snap_id)
        print(report)
        return
    # end RAG report snap

    # ======RAG report range (if requested)
    if args.rag_report_range:
        from awr_rag import create_llm, create_embeddings, create_redis_vectorstore, create_range_report_chain

        start_snap = int(args.rag_report_range[0])
        end_snap = int(args.rag_report_range[1])

        llm = create_llm()
        emb = create_embeddings()
        vs = create_redis_vectorstore("redis://localhost:6379", "awr_index", emb)

        chain = create_range_report_chain(llm, vs, start_snap, end_snap)
        report = chain.invoke(None)
        print(report)
        return
    # end RAG report range

    # ======RAG ask question (if requested)
    if args.rag_ask:
        from awr_rag import create_llm, create_embeddings, create_redis_vectorstore, create_qa_rag_chain
        question = " ".join(args.rag_ask)

        llm = create_llm()
        emb = create_embeddings()
        vs = create_redis_vectorstore("redis://localhost:6379", "awr_index", emb)

        chain = create_qa_rag_chain(llm, vs)
        answer = chain.invoke(question)
        print(answer)
        return
    # end RAG ask question

    # ======RAG report all (if requested)
    if args.rag_report_all:
        from awr_rag import (
            create_llm,
            create_embeddings,
            create_redis_vectorstore,
            create_range_report_chain
        )

        if main_metric is None:
            print("MAIN-METRICS section not found, cannot determine SNAP_ID range.")
            return

        min_snap = int(main_metric["SNAP_ID"].min())
        max_snap = int(main_metric["SNAP_ID"].max())

        print(f"Auto-detected SNAP_ID range: {min_snap} to {max_snap}")

        llm = create_llm()
        emb = create_embeddings()
        vs = create_redis_vectorstore("redis://localhost:6379", "awr_index", emb)

        chain = create_range_report_chain(llm, vs, min_snap, max_snap)
        report = chain.invoke(None)

        print("\n================ RAG REPORT (ALL SNAPSHOTS) ================\n")
        print(report)
        return
    # end RAG report all

    # ======RAG RUN ALL (Parse → Ingest → Report ALL)
    if args.rag_run_all:
        print("🚀 Menjalankan full pipeline RAG (parse → ingest → report all)...")
    
        save_path = args.save if args.save else None
    
        report = rag_run_all(
            os_info=os_info,
            os_memory=os_memory,
            main_metric=main_metric,
            aas=aas,
            top_wait=top_wait,
            top_sql=top_sql,
            save_path=save_path,
            analysis_level=analysis_level,
            recommendation_level=recommendation_level,
            language=language,
            style=style,
        )

    
        print(report)
        return
    # end RAG RUN ALL
    
    # ======Excel export (if requested)
    if args.excel:
        print("\n" + "="*60 + "\n" + "Writing Excel file...")
        if not OPENPYXL_AVAILABLE:
            print("Error: --excel flag requires openpyxl. Install with: pip install openpyxl", file=sys.stderr)
            sys.exit(4)
        excel_path = os.path.join(args.outdir, args.excel_filename)
        try:
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                for name in keys:
                    df = dfs.get(name)
                    if df is None or df.shape[0] == 0:
                        continue
                    # Sanitize sheet name (max 31 chars, no special chars)
                    sheet_name = sanitize_name(name)[:31]
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                    if args.verbose:
                        print(f"Wrote sheet: {sheet_name}")
            print(f"Excel file created: {excel_path}")
        except Exception as e:
            print(f"Failed to write Excel file: {e}", file=sys.stderr)
            sys.exit(5)
    
    # =====end Excel export        
    


if __name__ == '__main__':
    main()



