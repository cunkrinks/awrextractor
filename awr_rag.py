"""
awr_rag.py
RAG pipeline for Oracle AWR Miner:
- Ingestion from pandas DataFrames
- Snapshot Super-Document builder
- Hourly Super-Document builder
- Top SQL integration
- SQL trend analysis
- SQL plan change detection
- SQL bottleneck classifier
- Redis VectorStore integration
- RAG chains for LM Studio (OpenAI-compatible)

Author: Irvansyah (Cunkrink) + Copilot
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
import pandas as pd

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Redis
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.documents import Document


# ============================================================
# 1. LLM & Embeddings (LM Studio)
# ============================================================

def create_llm(
    base_url: str = "http://localhost:1234/v1",
    model: str = "lmstudio-oracle-dba",
    api_key: str = "lm-studio",
    temperature: float = 0.1,
) -> ChatOpenAI:
    """
    Create LM Studio LLM instance (OpenAI-compatible).
    """
    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
    )


def create_embeddings(
    base_url: str = "http://localhost:1234/v1",
    model: str = "lmstudio-embedding",
    api_key: str = "lm-studio",
) -> OpenAIEmbeddings:
    """
    Embedding model for vectorstore.
    """
    return OpenAIEmbeddings(
        base_url=base_url,
        api_key=api_key,
        model=model,
    )


# ============================================================
# 2. VectorStore Redis
# ============================================================

def create_redis_vectorstore(
    redis_url: str,
    index_name: str,
    embeddings: OpenAIEmbeddings,
) -> Redis:
    """
    Initialize Redis VectorStore.
    """
    return Redis(
        redis_url=redis_url,
        index_name=index_name,
        embedding=embeddings,
    )


# ============================================================
# 3. Ingestion: DataFrame → Documents
# ============================================================

# ---------- OS Info ----------

def build_os_info_doc(os_info: pd.DataFrame) -> Dict[str, Any]:
    """
    Build OS Info document.
    """
    d = {row["STAT_NAME"]: row["STAT_VALUE"] for _, row in os_info.iterrows()}

    text = f"""
Environment Summary (OS Info)

Database: {d.get("DB_NAME")}
Version: {d.get("VERSION")}
Instances: {d.get("INSTANCES")}
Hosts: {d.get("HOSTS")}

Hardware:
- CPUs: {d.get("NUM_CPUS")}
- CPU Cores: {d.get("NUM_CPU_CORES")}
- CPU Sockets: {d.get("NUM_CPU_SOCKETS")}
- Physical Memory: {d.get("PHYSICAL_MEMORY_GB")} GB

Platform:
- OS: {d.get("!PLATFORM_NAME")}
""".strip()

    return {
        "text": text,
        "metadata": {
            "type": "os_info",
            "db_name": d.get("DB_NAME"),
        },
    }

# ---------- Top SQL section (per snapshot) ----------

def build_top_sql_section(top_sql_df: pd.DataFrame, snap_id: int) -> str:
    """
    Build Top SQL section text for a given SNAP_ID.
    Expected columns:
    SNAP_ID, PARSING_SCHEMA_NAME, PLAN_HASH, MODULE, ACTION, SQL_ID,
    OPTIMIZER_COST, COMMAND_NAME, EXECS, BUFFER_GETS, ROWS_PROC,
    CPU_T_S, ELAP_S, READ_MB, IO_WAIT, ELAP_RANK, PLAN_CHANGE, PLANS,
    PHY_READ_GB, PX_SERVERS_EXECS, DIRECT_W_GB, IOWAIT_TIME, PIO
    """

    if top_sql_df is None or top_sql_df.empty:
        return "(No SQL data available for this snapshot)"

    df = top_sql_df[top_sql_df["SNAP_ID"] == snap_id]
    if df.empty:
        return "(No SQL data available for this snapshot)"

    # Sort by elapsed rank
    if "ELAP_RANK" in df.columns:
        df = df.sort_values("ELAP_RANK")
    elif "ELAP_S" in df.columns:
        df = df.sort_values("ELAP_S", ascending=False)

    lines: List[str] = []

    for _, row in df.iterrows():
        lines.append(f"""
{row.get('ELAP_RANK', '')}. SQL_ID {row['SQL_ID']}
   - Schema: {row.get('PARSING_SCHEMA_NAME', '')}
   - Module: {row.get('MODULE', '')}
   - Command: {row.get('COMMAND_NAME', '')}
   - Execs: {row.get('EXECS', '')}
   - Buffer Gets: {row.get('BUFFER_GETS', '')}
   - Rows Processed: {row.get('ROWS_PROC', '')}
   - CPU Time (s): {row.get('CPU_T_S', '')}
   - Elapsed Time (s): {row.get('ELAP_S', '')}
   - Read MB: {row.get('READ_MB', '')}
   - IO Wait (s): {row.get('IO_WAIT', '')}
   - Physical Read (GB): {row.get('PHY_READ_GB', '')}
   - Direct Write (GB): {row.get('DIRECT_W_GB', '')}
   - PX Servers: {row.get('PX_SERVERS_EXECS', '')}
   - Plan Hash: {row.get('PLAN_HASH', '')}
   - Plan Changes: {row.get('PLAN_CHANGE', '')}
   - Plans Seen: {row.get('PLANS', '')}
""".rstrip())

    return "\n\n".join(lines) if lines else "(No SQL rows for this snapshot)"


def summarize_top_sql_for_snapshot(top_sql_df: pd.DataFrame, snap_id: int) -> str:
    """
    Short verbal summary of Top SQL for a given SNAP_ID.
    """
    if top_sql_df is None or top_sql_df.empty:
        return "Tidak ada data SQL untuk snapshot ini."

    df = top_sql_df[top_sql_df["SNAP_ID"] == snap_id]
    if df.empty:
        return "Tidak ada data SQL untuk snapshot ini."

    if "ELAP_RANK" in df.columns:
        df = df.sort_values("ELAP_RANK")
    else:
        df = df.sort_values("ELAP_S", ascending=False)

    top1 = df.iloc[0]

    total_elapsed = df["ELAP_S"].sum() if "ELAP_S" in df.columns else 0
    total_cpu = df["CPU_T_S"].sum() if "CPU_T_S" in df.columns else 0
    total_execs = df["EXECS"].sum() if "EXECS" in df.columns else 0
    unique_sql = df["SQL_ID"].nunique()
    unique_schema = df["PARSING_SCHEMA_NAME"].nunique()
    top_module = (
        df["MODULE"].mode().iat[0]
        if "MODULE" in df.columns and not df["MODULE"].isna().all()
        else "N/A"
    )

    lines = []

    lines.append(
        f"Total {unique_sql} SQL di Top SQL untuk SNAP_ID {snap_id} "
        f"dengan total elapsed time {total_elapsed:.1f} s dan total CPU {total_cpu:.1f} s."
    )
    lines.append(
        f"Total executions: {int(total_execs)}; schema dominan: {unique_schema} schema; "
        f"module paling sering muncul: {top_module}."
    )

    lines.append(
        f"SQL paling berat berdasarkan elapsed time adalah SQL_ID {top1['SQL_ID']} "
        f"dengan elapsed {top1['ELAP_S']:.1f} s, CPU {top1['CPU_T_S']:.1f} s, "
        f"execs {top1['EXECS']}, buffer gets {top1['BUFFER_GETS']}."
    )

    plan_change_sql = df[df.get("PLAN_CHANGE", 0) > 0] if "PLAN_CHANGE" in df.columns else pd.DataFrame()
    if not plan_change_sql.empty:
        lines.append(
            f"Terdapat {len(plan_change_sql)} SQL dengan plan change dalam snapshot ini."
        )

    return "\n".join(lines)


# ---------- Snapshot Super-Document (per SNAP_ID + instance) ----------

def build_snapshot_superdocs_with_time(
    os_info: pd.DataFrame,
    os_memory: pd.DataFrame,
    main_metric: pd.DataFrame,
    aas: pd.DataFrame,
    top_wait: pd.DataFrame,
    top_sql: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """
    Build super-documents per SNAP_ID per instance.
    Uses:
    - OS-INFORMATION
    - MEMORY
    - MAIN-METRICS
    - AVERAGE-ACTIVE-SESSIONS
    - TOP-N-TIMED-EVENTS
    - TOP-SQL-BY-SNAPID
    """
    docs: List[Dict[str, Any]] = []

    os_dict = {row["STAT_NAME"]: row["STAT_VALUE"] for _, row in os_info.iterrows()}
    db_name = os_dict.get("DB_NAME")
    platform = os_dict.get("!PLATFORM_NAME")

    mm = main_metric.copy()
    # Expect columns: snap, end, dur_m, inst, ...
    # Normalize column names if needed
    col_map = {
        "SNAP_ID": "snap",
        "END": "end",
        "DUR_M": "dur_m",
        "INSTANCE_NUMBER": "inst",
    }
    for old, new in col_map.items():
        if old in mm.columns and new not in mm.columns:
            mm.rename(columns={old: new}, inplace=True)

    if not pd.api.types.is_datetime64_any_dtype(mm["end"]):
        mm["end"] = pd.to_datetime(mm["end"], errors="coerce")

    for snap_id in sorted(mm["snap"].unique()):
        mm_df = mm[mm["snap"] == snap_id]

        for _, row in mm_df.iterrows():
            instance = int(row["inst"])
            end_time = row["end"]
            duration_min = float(row["dur_m"])
            start_time = end_time - pd.Timedelta(minutes=duration_min)

            # Memory
            mem_df = os_memory[
                (os_memory["SNAP_ID"] == snap_id)
                & (os_memory["INSTANCE_NUMBER"] == instance)
            ]
            if len(mem_df) > 0:
                mrow = mem_df.iloc[0]
                mem_text = (
                    f"Instance {instance}:\n"
                    f"- SGA: {mrow['SGA']} GB\n"
                    f"- PGA: {mrow['PGA']} GB\n"
                    f"- Total: {mrow['TOTAL']} GB\n"
                )
            else:
                mem_text = f"Instance {instance}: (no memory data available)\n"

            # AAS
            aas_df = aas[aas["SNAP_ID"] == snap_id]
            aas_lines = [
                f"- {arow['WAIT_CLASS']}: {arow['AVG_SESS']}"
                for _, arow in aas_df.iterrows()
            ]
            aas_text = "\n".join(aas_lines) if aas_lines else "(no AAS data)"

            # Top Wait
            tw_df = top_wait[top_wait["SNAP_ID"] == snap_id]
            tw_lines = []
            for _, trow in tw_df.iterrows():
                tw_lines.append(
                    f"""{trow['EVENT_NAME']}
- Wait Class: {trow['WAIT_CLASS']}
- %DB Time: {trow['PCTDBT']}
- Total Wait Time: {trow['TOTAL_TIME_S']}"""
                )
            tw_text = "\n\n".join(tw_lines) if tw_lines else "(no top wait data)"

            # Top SQL
            top_sql_text = build_top_sql_section(top_sql, snap_id)
            top_sql_summary = summarize_top_sql_for_snapshot(top_sql, snap_id)

            text = f"""
Snapshot Super-Document — SNAP_ID {snap_id} (Instance {instance})

============================================================
1. Snapshot Info
============================================================
Database: {db_name}
Instance: {instance}
Platform: {platform}

SNAP_ID: {snap_id}
Start Time: {start_time}
End Time: {end_time}
Duration: {duration_min} minutes

============================================================
2. CPU & Workload Summary
============================================================
- OS CPU Usage: {row.get('os_cpu', row.get('OS_CPU', 'N/A'))}% (max {row.get('os_cpu_max', row.get('OS_CPU_MAX', 'N/A'))}%)
- DB CPU Ratio: {row.get('db_cpu_ratio', row.get('DB_CPU_RATIO', 'N/A'))}%
- DB Wait Ratio: {row.get('db_wait_ratio', row.get('DB_WAIT_RATIO', 'N/A'))}%
- AAS: {row.get('aas', row.get('AAS', 'N/A'))}
- Executions/s: {row.get('exec_s', row.get('EXEC_S', 'N/A'))}
- Logons/s: {row.get('logons_s', row.get('LOGONS_S', 'N/A'))}
- SQL Response Time: {row.get('sql_res_t_cs', row.get('SQL_RES_T_CS', 'N/A'))} cs

============================================================
3. Memory Summary
============================================================
{mem_text.strip()}

============================================================
4. Average Active Sessions (AAS)
============================================================
{aas_text}

============================================================
5. Top Wait Events
============================================================
{tw_text}

============================================================
6. Top SQL (Ranked by Elapsed Time)
============================================================
{top_sql_text}

Ringkasan Top SQL:
{top_sql_summary}

============================================================
7. I/O Summary
============================================================
- Read MB/s: {row.get('read_mb_s', row.get('READ_MB_S', 'N/A'))}
- Write MB/s: {row.get('write_mb_s', row.get('WRITE_MB_S', 'N/A'))}
- Read IOPS: {row.get('read_iops', row.get('READ_IOPS', 'N/A'))}
- Write IOPS: {row.get('write_iops', row.get('WRITE_IOPS', 'N/A'))}
- Redo MB/s: {row.get('redo_mb_s', row.get('REDO_MB_S', 'N/A'))}

============================================================
8. RAC / Global Cache
============================================================
- gc cr rec/s: {row.get('gc_cr_rec_s', row.get('GC_CR_REC_S', 'N/A'))}
- gc cu rec/s: {row.get('gc_cu_rec_s', row.get('GC_CU_REC_S', 'N/A'))}

============================================================
9. Overall Interpretation
============================================================
- Combine CPU, wait events, AAS, memory, RAC, and Top SQL metrics to identify bottlenecks.
""".strip()

            docs.append(
                {
                    "text": text,
                    "metadata": {
                        "type": "snapshot_superdoc",
                        "snap_id": int(snap_id),
                        "instance": instance,
                        "start_time": str(start_time),
                        "end_time": str(end_time),
                        "duration_min": duration_min,
                        "db_name": db_name,
                    },
                }
            )

    return docs


# ---------- Hourly Super-Document ----------

def build_hourly_superdocs(
    os_info: pd.DataFrame,
    os_memory: pd.DataFrame,
    main_metric: pd.DataFrame,
    aas: pd.DataFrame,
    top_wait: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """
    Build hourly super-documents using 'end' column in MAIN-METRICS.
    """
    docs: List[Dict[str, Any]] = []

    os_dict = {row["STAT_NAME"]: row["STAT_VALUE"] for _, row in os_info.iterrows()}
    db_name = os_dict.get("DB_NAME")
    platform = os_dict.get("!PLATFORM_NAME")

    mm = main_metric.copy()
    col_map = {
        "SNAP_ID": "snap",
        "END": "end",
        "INSTANCE_NUMBER": "inst",
    }
    for old, new in col_map.items():
        if old in mm.columns and new not in mm.columns:
            mm.rename(columns={old: new}, inplace=True)

    if not pd.api.types.is_datetime64_any_dtype(mm["end"]):
        mm["end"] = pd.to_datetime(mm["end"], errors="coerce")

    for hour in sorted(mm["end"].dt.hour.unique()):
        mm_hour = mm[mm["end"].dt.hour == hour]
        snap_ids = mm_hour["snap"].unique()
        if len(mm_hour) == 0:
            continue

        mem_hour = os_memory[os_memory["SNAP_ID"].isin(snap_ids)]
        aas_hour = aas[aas["SNAP_ID"].isin(snap_ids)]
        tw_hour = top_wait[top_wait["SNAP_ID"].isin(snap_ids)]

        cpu_avg = mm_hour["os_cpu"].mean() if "os_cpu" in mm_hour.columns else mm_hour["OS_CPU"].mean()
        cpu_max = mm_hour["os_cpu_max"].max() if "os_cpu_max" in mm_hour.columns else mm_hour["OS_CPU_MAX"].max()

        cpu_lines = []
        for _, row in mm_hour.iterrows():
            cpu_lines.append(
                f"""Snapshot {row['snap']} (Instance {row['inst']}):
- OS CPU: {row.get('os_cpu', row.get('OS_CPU', 'N/A'))}% (max {row.get('os_cpu_max', row.get('OS_CPU_MAX', 'N/A'))}%)
- DB CPU Ratio: {row.get('db_cpu_ratio', row.get('DB_CPU_RATIO', 'N/A'))}%
- DB Wait Ratio: {row.get('db_wait_ratio', row.get('DB_WAIT_RATIO', 'N/A'))}%
- AAS: {row.get('aas', row.get('AAS', 'N/A'))}
- Exec/s: {row.get('exec_s', row.get('EXEC_S', 'N/A'))}
- Logons/s: {row.get('logons_s', row.get('LOGONS_S', 'N/A'))}"""
            )

        mem_lines = []
        for inst in sorted(mem_hour["INSTANCE_NUMBER"].unique()):
            df = mem_hour[mem_hour["INSTANCE_NUMBER"] == inst]
            mem_lines.append(
                f"Instance {inst}: SGA={df['SGA'].mean()} GB, PGA={df['PGA'].mean()} GB"
            )

        aas_lines = []
        for wc in aas_hour["WAIT_CLASS"].unique():
            avg = aas_hour[aas_hour["WAIT_CLASS"] == wc]["AVG_SESS"].mean()
            aas_lines.append(f"- {wc}: {avg}")

        tw_lines = []
        for _, row in tw_hour.iterrows():
            tw_lines.append(
                f"""{row['EVENT_NAME']}
- Wait Class: {row['WAIT_CLASS']}
- %DB Time: {row['PCTDBT']}
- Total Wait Time: {row['TOTAL_TIME_S']}"""
            )

        text = f"""
Hourly Super-Document — {hour:02d}:00–{hour:02d}:59

Database: {db_name}
Platform: {platform}
Snapshots included: {', '.join(str(s) for s in snap_ids)}

============================================================
1. CPU & Workload Summary (Aggregated)
============================================================
Average OS CPU: {cpu_avg}%
Max OS CPU: {cpu_max}%

{chr(10).join(cpu_lines)}

============================================================
2. Memory Summary (Avg per Instance)
============================================================
{chr(10).join(mem_lines)}

============================================================
3. AAS Breakdown (Aggregated)
============================================================
{chr(10).join(aas_lines)}

============================================================
4. Top Wait Events (Aggregated)
============================================================
{chr(10).join(tw_lines)}

============================================================
5. I/O Summary (Aggregated)
============================================================
Read MB/s: {mm_hour.get('read_mb_s', mm_hour.get('READ_MB_S')).mean()}
Write MB/s: {mm_hour.get('write_mb_s', mm_hour.get('WRITE_MB_S')).mean()}
Read IOPS: {mm_hour.get('read_iops', mm_hour.get('READ_IOPS')).mean()}
Write IOPS: {mm_hour.get('write_iops', mm_hour.get('WRITE_IOPS')).mean()}
Redo MB/s: {mm_hour.get('redo_mb_s', mm_hour.get('REDO_MB_S')).mean()}

============================================================
6. RAC / Global Cache Summary
============================================================
gc cr rec/s: {mm_hour.get('gc_cr_rec_s', mm_hour.get('GC_CR_REC_S')).mean()}
gc cu rec/s: {mm_hour.get('gc_cu_rec_s', mm_hour.get('GC_CU_REC_S')).mean()}

============================================================
7. Overall Interpretation
============================================================
- Use CPU, wait, AAS, I/O, and RAC trends to identify hourly bottlenecks.
""".strip()

        docs.append(
            {
                "text": text,
                "metadata": {
                    "type": "hourly_superdoc",
                    "hour": int(hour),
                    "snap_ids": [int(s) for s in snap_ids],
                    "db_name": db_name,
                },
            }
        )

    return docs

# ============================================================
# 4. SQL Trend, Plan Change, Bottleneck Classification
# ============================================================

def sql_trend_over_range(
    top_sql_df: pd.DataFrame,
    start_snap: int,
    end_snap: int,
) -> pd.DataFrame:
    """
    Aggregate SQL metrics over a range of SNAP_ID.
    Returns DataFrame aggregated per SQL_ID.
    """
    if top_sql_df is None or top_sql_df.empty:
        return pd.DataFrame()

    df = top_sql_df[
        (top_sql_df["SNAP_ID"] >= start_snap)
        & (top_sql_df["SNAP_ID"] <= end_snap)
    ]

    if df.empty:
        return pd.DataFrame()

    grouped = (
        df.groupby("SQL_ID")
        .agg(
            schema=("PARSING_SCHEMA_NAME", "first"),
            module=("MODULE", "first"),
            command=("COMMAND_NAME", "first"),
            snaps=("SNAP_ID", "nunique"),
            total_execs=("EXECS", "sum"),
            total_cpu_s=("CPU_T_S", "sum"),
            total_elapsed_s=("ELAP_S", "sum"),
            total_buffer_gets=("BUFFER_GETS", "sum"),
            total_read_mb=("READ_MB", "sum"),
            total_io_wait_s=("IO_WAIT", "sum"),
            total_physical_read_gb=("PHY_READ_GB", "sum"),
            total_direct_w_gb=("DIRECT_W_GB", "sum"),
            max_plan_changes=("PLAN_CHANGE", "max"),
            plans_seen=("PLANS", "max"),
        )
        .reset_index()
    )

    grouped = grouped.sort_values("total_elapsed_s", ascending=False)

    return grouped


def summarize_sql_trend_text(
    agg_df: pd.DataFrame,
    start_snap: int,
    end_snap: int,
    top_n: int = 5,
) -> str:
    """
    Build textual summary for SQL trend over a range.
    """
    if agg_df is None or agg_df.empty:
        return f"Tidak ada data SQL untuk SNAP_ID {start_snap} sampai {end_snap}."

    top = agg_df.head(top_n)

    lines = [
        f"Analisa SQL dalam range SNAP_ID {start_snap} sampai {end_snap}:",
        f"Total SQL unik: {agg_df['SQL_ID'].nunique()}, "
        f"jumlah entri agregat: {len(agg_df)}.",
    ]

    for _, row in top.iterrows():
        lines.append(
            f"""
- SQL_ID {row['SQL_ID']} (schema {row['schema']}, module {row['module']}, command {row['command']}):
  - Muncul di {row['snaps']} snapshot
  - Total elapsed: {row['total_elapsed_s']:.1f} s
  - Total CPU: {row['total_cpu_s']:.1f} s
  - Total execs: {int(row['total_execs'])}
  - Total buffer gets: {int(row['total_buffer_gets'])}
  - Total read MB: {row['total_read_mb']:.1f}
  - IO wait total: {row['total_io_wait_s']:.1f} s
  - Physical read: {row['total_physical_read_gb']:.1f} GB
  - Direct write: {row['total_direct_w_gb']:.1f} GB
  - Plan changes max: {row['max_plan_changes']}, plans seen: {row['plans_seen']}
""".rstrip()
        )

    return "\n".join(lines)


def detect_sql_plan_changes(
    top_sql_df: pd.DataFrame,
    start_snap: int,
    end_snap: int,
) -> pd.DataFrame:
    """
    Detect SQL with plan changes over a range of SNAP_ID.
    Criteria:
    - PLANS > 1 or PLAN_CHANGE > 0 or more than one PLAN_HASH
    """
    if top_sql_df is None or top_sql_df.empty:
        return pd.DataFrame()

    df = top_sql_df[
        (top_sql_df["SNAP_ID"] >= start_snap)
        & (top_sql_df["SNAP_ID"] <= end_snap)
    ]

    if df.empty:
        return pd.DataFrame()

    by_sql = df.groupby("SQL_ID").agg(
        schemas=("PARSING_SCHEMA_NAME", lambda x: list(sorted(set(x)))),
        modules=("MODULE", lambda x: list(sorted(set(x)))),
        plan_hashes=("PLAN_HASH", lambda x: list(sorted(set(x)))),
        min_plans=("PLANS", "min"),
        max_plans=("PLANS", "max"),
        max_plan_change=("PLAN_CHANGE", "max"),
        snaps=("SNAP_ID", "nunique"),
        total_elapsed_s=("ELAP_S", "sum"),
        total_cpu_s=("CPU_T_S", "sum"),
    ).reset_index()

    mask = (
        (by_sql["max_plan_change"] > 0)
        | (by_sql["max_plans"] > 1)
        | (by_sql["plan_hashes"].apply(lambda x: len(x) > 1))
    )

    return by_sql[mask]


def summarize_plan_change_text(
    plan_df: pd.DataFrame,
    start_snap: int,
    end_snap: int,
) -> str:
    """
    Text summary for SQL with plan changes.
    """
    if plan_df is None or plan_df.empty:
        return f"Tidak ditemukan SQL dengan plan change signifikan antara SNAP_ID {start_snap} dan {end_snap}."

    lines = [
        f"SQL dengan indikasi plan change antara SNAP_ID {start_snap} dan {end_snap}:",
        f"Total: {len(plan_df)} SQL."
    ]

    for _, row in plan_df.iterrows():
        lines.append(
            f"""
- SQL_ID {row['SQL_ID']}:
  - Schemas: {', '.join(map(str, row['schemas']))}
  - Modules: {', '.join(map(str, row['modules']))}
  - Plan hashes: {', '.join(map(str, row['plan_hashes']))}
  - Plans (min-max): {row['min_plans']}–{row['max_plans']}
  - Max PLAN_CHANGE flag: {row['max_plan_change']}
  - Snapshot muncul: {row['snaps']}
  - Total elapsed: {row['total_elapsed_s']:.1f} s
  - Total CPU: {row['total_cpu_s']:.1f} s
""".rstrip()
        )

    return "\n".join(lines)


def classify_sql_bottleneck(row: pd.Series) -> List[str]:
    """
    Classify bottleneck type for a single aggregated SQL row.
    """
    labels: List[str] = []

    cpu = row.get("total_cpu_s", 0.0)
    ela = row.get("total_elapsed_s", 0.0)
    io_wait = row.get("total_io_wait_s", 0.0)
    read_mb = row.get("total_read_mb", 0.0)
    phys_gb = row.get("total_physical_read_gb", 0.0)
    direct_w = row.get("total_direct_w_gb", 0.0)
    plans = row.get("plans_seen", 0)
    plan_change = row.get("max_plan_changes", 0)
    execs = row.get("total_execs", 0.0)

    # CPU-bound
    if ela > 0 and (cpu / ela) > 0.7 and cpu > 10:
        labels.append("CPU-bound")

    # IO-bound
    if io_wait > 10 and (read_mb > 100 or phys_gb > 1.0):
        labels.append("IO-bound")

    # PX-intensive (heuristic via direct write)
    if direct_w > 5.0:
        labels.append("PX-intensive")

    # Plan instability
    if plan_change > 0 or plans > 1:
        labels.append("Plan-change-risk")

    # Low impact
    if ela < 10 and cpu < 5 and execs < 10:
        labels.append("Low-impact")

    if not labels:
        labels.append("Mixed/Other")

    return labels


def classify_all_sql_in_range(
    top_sql_df: pd.DataFrame,
    start_snap: int,
    end_snap: int,
) -> pd.DataFrame:
    """
    Apply bottleneck classification to all SQL in a range.
    """
    agg = sql_trend_over_range(top_sql_df, start_snap, end_snap)
    if agg is None or agg.empty:
        return agg

    agg = agg.copy()
    agg["bottleneck_class"] = agg.apply(classify_sql_bottleneck, axis=1)
    return agg


def summarize_bottleneck_classes_text(
    agg_cls: pd.DataFrame,
    top_n: int = 10,
) -> str:
    """
    Text summary for bottleneck classification of SQL in range.
    """
    if agg_cls is None or agg_cls.empty:
        return "Tidak ada SQL signifikan yang terdeteksi dalam range ini."

    lines = ["Klasifikasi bottleneck SQL dalam range snapshot:"]

    from collections import Counter

    flat = [c for classes in agg_cls["bottleneck_class"] for c in classes]
    dist = Counter(flat)
    lines.append(
        "Distribusi kategori: "
        + ", ".join(f"{k}: {v}" for k, v in dist.items())
    )

    top = agg_cls.sort_values("total_elapsed_s", ascending=False).head(top_n)
    for _, row in top.iterrows():
        labels = ", ".join(row["bottleneck_class"])
        lines.append(
            f"""
- SQL_ID {row['SQL_ID']} ({labels}):
  - Schema: {row['schema']}, Module: {row['module']}, Command: {row['command']}
  - Muncul di {row['snaps']} snapshot
  - Total elapsed: {row['total_elapsed_s']:.1f} s, CPU: {row['total_cpu_s']:.1f} s
  - Execs: {int(row['total_execs'])}, Buffer gets: {int(row['total_buffer_gets'])}
  - Read MB: {row['total_read_mb']:.1f}, IO wait: {row['total_io_wait_s']:.1f} s
"""
        )

    return "\n".join(lines)


# ============================================================
# 5. Documents → LangChain Documents & Redis Upsert
# ============================================================

def docs_to_langchain_documents(docs: List[Dict[str, Any]]) -> List[Document]:
    """
    Convert list[{text, metadata}] to LangChain Document objects.
    """
    return [Document(page_content=d["text"], metadata=d["metadata"]) for d in docs]


def upsert_documents_to_redis(
    vectorstore: Redis,
    docs: List[Dict[str, Any]],
    batch_size: int = 100,
) -> None:
    """
    Upsert documents into Redis VectorStore in batches.
    """
    if not docs:
        return

    lc_docs = docs_to_langchain_documents(docs)
    for i in range(0, len(lc_docs), batch_size):
        batch = lc_docs[i : i + batch_size]
        vectorstore.add_documents(batch)


# ============================================================
# 6. RAG Prompts & Chains
# ============================================================

def format_docs(docs: List[Document]) -> str:
    """
    Join multiple documents into a single context string.
    """
    return "\n\n---\n\n".join(d.page_content for d in docs)


BASE_SYSTEM_PROMPT = """
Anda adalah Oracle Database Performance Expert dengan pengalaman lebih dari 15 tahun.
Anda menganalisis AWR, ASH, AWR Miner, dan RAC untuk sistem berskala besar.

Gunakan HANYA konteks yang diberikan.
JANGAN membuat metrik atau data baru yang tidak ada di konteks.

Fokus pada:
- Identifikasi bottleneck
- Root cause analysis
- Rekomendasi tuning yang spesifik dan actionable
"""


def create_qa_rag_chain(llm: ChatOpenAI, vectorstore: Redis):
    """
    RAG chain untuk tanya-jawab bebas seputar performa database.
    Menggunakan semua dokumen dalam Redis.
    """
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6},
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", BASE_SYSTEM_PROMPT),
            (
                "human",
                "Konteks AWR:\n\n{context}\n\nPertanyaan:\n{question}",
            ),
        ]
    )

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def create_snapshot_report_chain(
    llm: ChatOpenAI,
    vectorstore: Redis,
    snap_id: int,
    instance: Optional[int] = None,
):
    """
    RAG chain untuk laporan per SNAP_ID (+optional instance).
    Mengutamakan dokumen type 'snapshot_superdoc'.
    """
    filter_meta: Dict[str, Any] = {
        "snap_id": snap_id,
        "type": "snapshot_superdoc",
    }
    if instance is not None:
        filter_meta["instance"] = instance

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 4,
            "filter": filter_meta,
        },
    )

    system_prompt = (
        BASE_SYSTEM_PROMPT
        + """

Tugas khusus Anda:
- Buat laporan performa lengkap untuk snapshot yang diberikan.
- Struktur jawaban:
  1. Executive Summary
  2. CPU & Workload Analysis
  3. Wait Event & AAS Analysis
  4. I/O & RAC Analysis
  5. Top SQL Analysis
  6. Root Cause
  7. Rekomendasi Tuning (prioritas tinggi → rendah)
"""
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                "Konteks AWR untuk SNAP_ID {snap_id}:\n\n{context}\n\n"
                "Buatkan laporan performa lengkap untuk snapshot ini.",
            ),
        ]
    )

    chain = (
        {
            "context": retriever | format_docs,
            "snap_id": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def create_range_report_chain(
    llm: ChatOpenAI,
    vectorstore: Redis,
    start_snap: int,
    end_snap: int,
):
    """
    RAG chain untuk laporan performa range SNAP_ID besar.
    Menggunakan snapshot_superdoc + hourly_superdoc (jika ada).
    """
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 20,
            "filter": {
                "snap_id": {"$gte": start_snap, "$lte": end_snap},
            },
        },
    )

    system_prompt = (
        BASE_SYSTEM_PROMPT
        + """

Tugas khusus Anda:
- Analisa performa database dalam rentang snapshot yang besar.
- Identifikasi tren CPU, wait event, AAS, I/O, dan RAC.
- Temukan bottleneck utama dan root cause untuk periode tersebut.
- Berikan rekomendasi tuning yang relevan untuk jangka waktu itu.

Gunakan struktur:
1. Executive Summary
2. CPU Trend Analysis
3. Wait Event Trend Analysis
4. AAS Trend Analysis
5. I/O Trend Analysis
6. RAC Trend Analysis
7. Root Cause Analysis
8. Rekomendasi Tuning
"""
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                "Konteks AWR untuk SNAP_ID {start_snap} sampai {end_snap}:\n\n"
                "{context}\n\n"
                "Buatkan laporan performa lengkap untuk rentang snapshot ini.",
            ),
        ]
    )

    chain = (
        {
            "context": retriever | format_docs,
            "start_snap": lambda _: start_snap,
            "end_snap": lambda _: end_snap,
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

