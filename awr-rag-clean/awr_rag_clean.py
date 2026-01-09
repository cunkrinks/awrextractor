"""
awr_rag.py (CLEAN VERSION)
RAG pipeline for Oracle AWR Miner:
- Ingestion from pandas DataFrames
- Snapshot Super-Document builder
- Hourly Super-Document builder
- Top SQL integration
- SQL trend analysis
- Redis VectorStore integration
- RAG chains for LM Studio (OpenAI-compatible)

Author: Irvansyah (Cunkrink) + Copilot
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_redis import RedisVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
import requests


# ============================================================
# Embedding Model (LM Studio)
# ============================================================

class LMStudioEmbedding(Embeddings):
    def __init__(self, url="http://localhost:1235/v1/embeddings",
                 model="text-embedding-bge-base-en-v1.5"):
        self.url = url
        self.model = model

    def embed_documents(self, texts):
        sanitized = []
        for t in texts:
            if isinstance(t, dict):
                t = t.get("content") or t.get("text") or str(t)
            if t is None:
                t = ""
            sanitized.append(str(t))

        payload = {"model": self.model, "input": sanitized}
        r = requests.post(self.url, json=payload)
        r.raise_for_status()
        data = r.json()
        return [item["embedding"] for item in data["data"]]

    def embed_query(self, text):
        if isinstance(text, dict):
            text = text.get("query", str(text))
        if text is None:
            text = ""
        payload = {"model": self.model, "input": [str(text)]}
        r = requests.post(self.url, json=payload)
        r.raise_for_status()
        return r.json()["data"][0]["embedding"]


# ============================================================
# Sanitizer
# ============================================================

def sanitize_numeric(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


# ============================================================
# LLM & Embeddings
# ============================================================

def create_llm(
    base_url="http://localhost:1235/v1",
    model="meta-llama-3.1-8b-instruct",
    api_key="lm-studio",
    temperature=0.1,
) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
    )


def create_embeddings():
    print("🔄 Using LM Studio Embedding Server (bge-base @ 1235)")
    return LMStudioEmbedding(
        url="http://localhost:1235/v1/embeddings",
        model="text-embedding-bge-base-en-v1.5",
    )


# ============================================================
# Redis VectorStore
# ============================================================

def create_redis_vectorstore(redis_url: str, index_name: str, embeddings):
    return RedisVectorStore(
        embeddings=embeddings,
        redis_url=redis_url,
        index_name=index_name,
    )


# ============================================================
# OS Info Document
# ============================================================

def build_os_info_doc(os_info: pd.DataFrame) -> Dict[str, Any]:
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

# ============================================================
# Top SQL Section
# ============================================================

def build_top_sql_section(top_sql_df: pd.DataFrame, snap_id: int) -> str:
    if top_sql_df is None or top_sql_df.empty:
        return "(No SQL data available for this snapshot)"

    top_sql_df = sanitize_numeric(
        top_sql_df,
        [
            "ELAP_S", "CPU_T_S", "EXECS", "BUFFER_GETS", "ROWS_PROC",
            "READ_MB", "IO_WAIT", "PHY_READ_GB", "DIRECT_W_GB",
            "PX_SERVERS_EXECS", "PLAN_CHANGE", "PLANS",
        ],
    )

    df = top_sql_df[top_sql_df["SNAP_ID"] == snap_id]
    if df.empty:
        return "(No SQL data available for this snapshot)"

    if "ELAP_RANK" in df.columns:
        df = df.sort_values("ELAP_RANK")
    else:
        df = df.sort_values("ELAP_S", ascending=False)

    lines = []
    for _, row in df.iterrows():
        lines.append(
            f"""
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
""".rstrip()
        )

    return "\n\n".join(lines) if lines else "(No SQL rows for this snapshot)"


def summarize_top_sql_for_snapshot(top_sql_df: pd.DataFrame, snap_id: int) -> str:
    if top_sql_df is None or top_sql_df.empty:
        return "Tidak ada data SQL untuk snapshot ini."

    df = top_sql_df[top_sql_df["SNAP_ID"] == snap_id].copy()
    if df.empty:
        return "Tidak ada data SQL untuk snapshot ini."

    df = sanitize_numeric(
        df,
        [
            "ELAP_S", "CPU_T_S", "EXECS", "BUFFER_GETS", "ROWS_PROC",
            "READ_MB", "IO_WAIT", "PHY_READ_GB", "DIRECT_W_GB",
        ],
    )

    if "ELAP_RANK" in df.columns:
        df = df.sort_values("ELAP_RANK")
    else:
        df = df.sort_values("ELAP_S", ascending=False)

    top1 = df.iloc[0]

    total_elapsed = float(df["ELAP_S"].sum())
    total_cpu = float(df["CPU_T_S"].sum())
    total_execs = int(df["EXECS"].sum()) if "EXECS" in df.columns else 0
    unique_sql = df["SQL_ID"].nunique()
    unique_schema = df["PARSING_SCHEMA_NAME"].nunique()
    top_module = (
        df["MODULE"].mode().iat[0]
        if "MODULE" in df.columns and not df["MODULE"].isna().all()
        else "N/A"
    )

    lines = [
        f"Total {unique_sql} SQL di Top SQL untuk SNAP_ID {snap_id} "
        f"dengan total elapsed time {total_elapsed:.1f} s dan total CPU {total_cpu:.1f} s.",
        f"Total executions: {total_execs}; schema dominan: {unique_schema} schema; "
        f"module paling sering muncul: {top_module}.",
        f"SQL paling berat adalah SQL_ID {top1['SQL_ID']} "
        f"dengan elapsed {float(top1['ELAP_S']):.1f} s, CPU {float(top1['CPU_T_S']):.1f} s, "
        f"execs {int(top1.get('EXECS', 0))}, buffer gets {int(top1.get('BUFFER_GETS', 0))}.",
    ]

    plan_change_sql = df[df.get("PLAN_CHANGE", 0) > 0] if "PLAN_CHANGE" in df.columns else pd.DataFrame()
    if not plan_change_sql.empty:
        lines.append(f"Terdapat {len(plan_change_sql)} SQL dengan plan change dalam snapshot ini.")

    return "\n".join(lines)


# ============================================================
# Snapshot Super-Documents
# ============================================================

def build_snapshot_superdocs_with_time(
    os_info: pd.DataFrame,
    os_memory: pd.DataFrame,
    main_metric: pd.DataFrame,
    aas: pd.DataFrame,
    top_wait: pd.DataFrame,
    top_sql: pd.DataFrame,
) -> List[Dict[str, Any]]:

    docs = []

    os_memory = sanitize_numeric(os_memory, ["SGA", "PGA", "TOTAL"])
    main_metric = sanitize_numeric(
        main_metric,
        [
            "os_cpu", "os_cpu_max", "OS_CPU", "OS_CPU_MAX",
            "db_cpu_ratio", "DB_CPU_RATIO", "db_wait_ratio", "DB_WAIT_RATIO",
            "aas", "AAS", "exec_s", "EXEC_S", "logons_s", "LOGONS_S",
            "sql_res_t_cs", "SQL_RES_T_CS",
            "read_mb_s", "READ_MB_S", "write_mb_s", "WRITE_MB_S",
            "read_iops", "READ_IOPS", "write_iops", "WRITE_IOPS",
            "redo_mb_s", "REDO_MB_S",
            "gc_cr_rec_s", "GC_CR_REC_S", "gc_cu_rec_s", "GC_CU_REC_S",
        ],
    )
    aas = sanitize_numeric(aas, ["AVG_SESS"])
    top_wait = sanitize_numeric(top_wait, ["PCTDBT", "TOTAL_TIME_S"])
    top_sql = sanitize_numeric(
        top_sql,
        [
            "ELAP_S", "CPU_T_S", "EXECS", "BUFFER_GETS", "ROWS_PROC",
            "READ_MB", "IO_WAIT", "PHY_READ_GB", "DIRECT_W_GB",
            "PX_SERVERS_EXECS", "PLAN_CHANGE", "PLANS",
        ],
    )

    os_dict = {row["STAT_NAME"]: row["STAT_VALUE"] for _, row in os_info.iterrows()}
    db_name = os_dict.get("DB_NAME")
    platform = os_dict.get("!PLATFORM_NAME")

    mm = main_metric.copy()
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
        mm["end"] = pd.to_datetime(
            mm["end"],
            format="%y/%m/%d %H:%M",
            errors="coerce",
        )

    for snap_id in sorted(mm["snap"].unique()):
        mm_df = mm[mm["snap"] == snap_id]

        for _, row in mm_df.iterrows():
            instance = int(row["inst"])
            end_time = row["end"]
            duration_min = float(row["dur_m"])
            start_time = end_time - pd.Timedelta(minutes=duration_min)

            start_fmt = start_time.strftime("%Y-%m-%d %H:%M")
            end_fmt = end_time.strftime("%Y-%m-%d %H:%M")

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
Start Time: {start_fmt}
End Time: {end_fmt}
Duration: {duration_min} minutes

============================================================
2. CPU & Workload Summary
============================================================
- OS CPU Usage: {row.get('os_cpu', row.get('OS_CPU', 'N/A'))}% 
  (max {row.get('os_cpu_max', row.get('OS_CPU_MAX', 'N/A'))}%)
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


# ============================================================
# Hourly Super-Documents
# ============================================================

def build_hourly_superdocs(
    os_info: pd.DataFrame,
    os_memory: pd.DataFrame,
    main_metric: pd.DataFrame,
    aas: pd.DataFrame,
    top_wait: pd.DataFrame,
) -> List[Dict[str, Any]]:

    docs = []

    os_memory = sanitize_numeric(os_memory, ["SGA", "PGA", "TOTAL"])
    main_metric = sanitize_numeric(
        main_metric,
        [
            "os_cpu", "os_cpu_max", "OS_CPU", "OS_CPU_MAX",
            "db_cpu_ratio", "DB_CPU_RATIO", "db_wait_ratio", "DB_WAIT_RATIO",
            "aas", "AAS", "exec_s", "EXEC_S", "logons_s", "LOGONS_S",
            "sql_res_t_cs", "SQL_RES_T_CS",
            "read_mb_s", "READ_MB_S", "write_mb_s", "WRITE_MB_S",
            "read_iops", "READ_IOPS", "write_iops", "WRITE_IOPS",
            "redo_mb_s", "REDO_MB_S",
            "gc_cr_rec_s", "GC_CR_REC_S", "gc_cu_rec_s", "GC_CU_REC_S",
        ],
    )
    aas = sanitize_numeric(aas, ["AVG_SESS"])
    top_wait = sanitize_numeric(top_wait, ["PCTDBT", "TOTAL_TIME_S"])

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
        mm["end"] = pd.to_datetime(
            mm["end"],
            format="%y/%m/%d %H:%M",
            errors="coerce",
        )

    for hour in sorted(mm["end"].dt.hour.unique()):
        mm_hour = mm[mm["end"].dt.hour == hour]
        if len(mm_hour) == 0:
            continue

        snap_ids = mm_hour["snap"].unique()
        mem_hour = os_memory[os_memory["SNAP_ID"].isin(snap_ids)]
        aas_hour = aas[aas["SNAP_ID"].isin(snap_ids)]
        tw_hour = top_wait[top_wait["SNAP_ID"].isin(snap_ids)]

        cpu_series = mm_hour.get("os_cpu", mm_hour.get("OS_CPU"))
        cpu_max_series = mm_hour.get("os_cpu_max", mm_hour.get("OS_CPU_MAX"))
        cpu_avg = cpu_series.mean() if cpu_series is not None else 0.0
        cpu_max = cpu_max_series.max() if cpu_max_series is not None else 0.0

        cpu_lines = []
        for _, row in mm_hour.iterrows():
            cpu_lines.append(
                f"""Snapshot {row['snap']} (Instance {row['inst']}):
- OS CPU: {row.get('os_cpu', row.get('OS_CPU', 'N/A'))}% 
  (max {row.get('os_cpu_max', row.get('OS_CPU_MAX', 'N/A'))}%)
- DB CPU Ratio: {row.get('db_cpu_ratio', row.get('DB_CPU_RATIO', 'N/A'))}%
- DB Wait Ratio: {row.get('db_wait_ratio', row.get('DB_WAIT_RATIO', 'N/A'))}%
- AAS: {row.get('aas', row.get('AAS', 'N/A'))}
- Exec/s: {row.get('exec_s', row.get('EXEC_S', 'N/A'))}
- Logons/s: {row.get('logons_s', row.get('LOGONS_S', 'N/A'))}"""
            )

        mem_lines = []
        for inst in sorted(mem_hour["INSTANCE_NUMBER"].unique()):
            df_i = mem_hour[mem_hour["INSTANCE_NUMBER"] == inst]
            mem_lines.append(
                f"Instance {inst}: SGA={df_i['SGA'].mean()} GB, PGA={df_i['PGA'].mean()} GB"
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

        read_mb_s = mm_hour.get("read_mb_s", mm_hour.get("READ_MB_S"))
        write_mb_s = mm_hour.get("write_mb_s", mm_hour.get("WRITE_MB_S"))
        read_iops = mm_hour.get("read_iops", mm_hour.get("READ_IOPS"))
        write_iops = mm_hour.get("write_iops", mm_hour.get("WRITE_IOPS"))
        redo_mb_s = mm_hour.get("redo_mb_s", mm_hour.get("REDO_MB_S"))

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
Read MB/s: {read_mb_s.mean() if read_mb_s is not None else 'N/A'}
Write MB/s: {write_mb_s.mean() if write_mb_s is not None else 'N/A'}
Read IOPS: {read_iops.mean() if read_iops is not None else 'N/A'}
Write IOPS: {write_iops.mean() if write_iops is not None else 'N/A'}
Redo MB/s: {redo_mb_s.mean() if redo_mb_s is not None else 'N/A'}

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
# SQL Trend Aggregation
# ============================================================

def sql_trend_over_range(
    top_sql_df: pd.DataFrame,
    start_snap: int,
    end_snap: int,
) -> pd.DataFrame:

    if top_sql_df is None or top_sql_df.empty:
        return pd.DataFrame()

    top_sql_df = sanitize_numeric(
        top_sql_df,
        [
            "EXECS", "CPU_T_S", "ELAP_S", "BUFFER_GETS",
            "READ_MB", "IO_WAIT", "PHY_READ_GB", "DIRECT_W_GB",
            "PLAN_CHANGE", "PLANS",
        ],
    )

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

    return grouped.sort_values("total_elapsed_s", ascending=False)


def summarize_sql_trend_text(
    agg_df: pd.DataFrame,
    start_snap: int,
    end_snap: int,
    top_n: int = 5,
) -> str:

    if agg_df is None or agg_df.empty:
        return f"Tidak ada data SQL untuk SNAP_ID {start_snap} sampai {end_snap}."

    top = agg_df.head(top_n)

    lines = [
        f"Analisa SQL dalam range SNAP_ID {start_snap} sampai {end_snap}:",
        f"Total SQL unik: {agg_df['SQL_ID'].nunique()}, jumlah entri agregat: {len(agg_df)}.",
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
  - Total IO wait: {row['total_io_wait_s']:.1f} s
  - Plan changes: {row['max_plan_changes']}
  - Plans seen: {row['plans_seen']}
""".rstrip()
        )

    return "\n".join(lines)