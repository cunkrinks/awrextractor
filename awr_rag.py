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
- RAG chains for LM Studio (OpenAI-compatible) for LLM
- HuggingFaceEmbeddings (nomic-ai/nomic-embed-text-v1) for embeddings

Author: Irvansyah (Cunkrink) + Copilot
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_redis import RedisVectorStore
from langchain_community.vectorstores import Redis
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from tqdm import tqdm
import torch
import requests

from langchain_core.embeddings import Embeddings
import requests

class LMStudioEmbedding(Embeddings):
    def __init__(self, url="http://localhost:1235/v1/embeddings", model="text-embedding-bge-reranker-v2-m3"):
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
        # --- FIX: jika input dict, ambil field "query" ---
        if isinstance(text, dict):
            if "query" in text:
                text = text["query"]
            else:
                text = str(text)

        if text is None:
            text = ""

        text = str(text)

        payload = {"model": self.model, "input": [text]}
        r = requests.post(self.url, json=payload)
        r.raise_for_status()
        return r.json()["data"][0]["embedding"]


BASE_SYSTEM_PROMPT = """
Anda adalah Senior Database Performance Engineer & DBA Specialist.
Gunakan konteks yang diberikan untuk menjelaskan tren, bottleneck,
dan rekomendasi tuning secara teknis dan akurat.

PEDOMAN ANALISIS:
- Hubungan Metrik: Jika CPU tinggi, periksa apakah berkaitan dengan 'Slow Queries' atau 'High Connections'. Jika Latency tinggi tapi CPU rendah, periksa 'Disk I/O' atau 'Network Wait'.
- Batas Ambang (Threshold): Anggap CPU > 80% sebagai peringatan, dan > 90% sebagai kritis. Buffer Cache Hit Ratio di bawah 90% mengindikasikan masalah memori.
- Fokus pada Solusi: Jangan hanya menyebutkan angka, jelaskan *mengapa* angka itu bermasalah dan *apa* perintah SQL atau tindakan yang harus diambil.

GAYA BAHASA:
- Profesional, teknis, dan langsung ke poin (concise).
- Gunakan istilah teknis seperti: Index Scan, Sequential Scan, Deadlock, Connection Pooling, I/O Wait, dan Cache Hit Ratio

"""

# ============================================================
# GLOBAL SANITIZER
# ============================================================

def sanitize_numeric(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    """
    Convert selected columns to numeric safely.
    Any invalid value becomes 0.0.
    """
    if df is None or df.empty:
        return df

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df


# ============================================================
# 1. LLM & Embeddings
# ============================================================

def create_llm(
    base_url: str = "http://localhost:1235/v1",
    model: str = "meta-llama-3.1-8b-instruct",
    api_key: str = "lm-studio",
    temperature: float = 0.1,
) -> ChatOpenAI:
    """
    Create LLM instance (e.g. LM Studio / OpenAI-compatible endpoint).
    This is used only for generation / reasoning, NOT embeddings.
    """
    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
    )


def create_embeddings():
    """
    Premium embedding model for vectorstore.
    Using HuggingFace local embeddings (no LM Studio needed).

    Model: nomic-ai/nomic-embed-text-v1
    - Sangat cocok untuk dokumen teknis panjang (AWR super-docs).
    - Akurat untuk semantic search di konteks RAG.
    """
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(f"Using device for embeddings: {device}")
    #return HuggingFaceEmbeddings(
    #    model_name="nomic-ai/nomic-embed-text-v1",
    #    model_kwargs={
    #        #"device": "cpu", # gunakan GPU jika ada; ganti ke "cpu" jika perlu
    #        "device": device,
    #        "trust_remote_code": True
    #    },
    #    encode_kwargs={"normalize_embeddings": True},
    #)
    print("🔄 Using LM Studio Embedding Server (bge-base @ 1235)")
    return LMStudioEmbedding(
         url="http://localhost:1235/v1/embeddings",
         model="text-embedding-bge-reranker-v2-m3"
     )


# ============================================================
# 2. VectorStore Redis
# ============================================================

from langchain_redis import RedisVectorStore

#def create_redis_vectorstore(redis_url: str, index_name: str, embeddings):
#    return RedisVectorStore(
#        embeddings=embeddings,
#        redis_url=redis_url,
#        index_name=index_name,
#    )
def create_redis_vectorstore(redis_url: str, index_name: str, embeddings):
    return RedisVectorStore(
        embeddings=embeddings,
        redis_url=redis_url,
        index_name=index_name,
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

    # Sanitize numeric columns globally
    top_sql_df = sanitize_numeric(
        top_sql_df,
        [
            "ELAP_S",
            "CPU_T_S",
            "EXECS",
            "BUFFER_GETS",
            "ROWS_PROC",
            "READ_MB",
            "IO_WAIT",
            "PHY_READ_GB",
            "DIRECT_W_GB",
            "PX_SERVERS_EXECS",
            "PLAN_CHANGE",
            "PLANS",
        ],
    )

    df = top_sql_df[top_sql_df["SNAP_ID"] == snap_id]
    if df.empty:
        return "(No SQL data available for this snapshot)"

    # Sort by elapsed rank or elapsed time
    if "ELAP_RANK" in df.columns:
        df = df.sort_values("ELAP_RANK")
    elif "ELAP_S" in df.columns:
        df = df.sort_values("ELAP_S", ascending=False)

    lines: List[str] = []

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
    """
    Short verbal summary of Top SQL for a given SNAP_ID.
    """
    if top_sql_df is None or top_sql_df.empty:
        return "Tidak ada data SQL untuk snapshot ini."

    df = top_sql_df[top_sql_df["SNAP_ID"] == snap_id].copy()
    if df.empty:
        return "Tidak ada data SQL untuk snapshot ini."

    # Convert numeric columns safely
    df = sanitize_numeric(
        df,
        [
            "ELAP_S",
            "CPU_T_S",
            "EXECS",
            "BUFFER_GETS",
            "ROWS_PROC",
            "READ_MB",
            "IO_WAIT",
            "PHY_READ_GB",
            "DIRECT_W_GB",
        ],
    )

    # Sorting setelah numeric conversion
    if "ELAP_RANK" in df.columns:
        df = df.sort_values("ELAP_RANK")
    else:
        df = df.sort_values("ELAP_S", ascending=False)

    # Ambil top1 setelah numeric conversion
    top1 = df.iloc[0]

    # Total summary
    total_elapsed = float(df["ELAP_S"].sum())
    total_cpu = float(df["CPU_T_S"].sum())
    total_execs = float(df["EXECS"].sum()) if "EXECS" in df.columns else 0.0
    unique_sql = df["SQL_ID"].nunique()
    unique_schema = df["PARSING_SCHEMA_NAME"].nunique()
    top_module = (
        df["MODULE"].mode().iat[0]
        if "MODULE" in df.columns and not df["MODULE"].isna().all()
        else "N/A"
    )

    lines: List[str] = []

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
        f"dengan elapsed {float(top1['ELAP_S']):.1f} s, CPU {float(top1['CPU_T_S']):.1f} s, "
        f"execs {int(top1.get('EXECS', 0))}, buffer gets {int(top1.get('BUFFER_GETS', 0))}."
    )

    plan_change_sql = (
        df[df.get("PLAN_CHANGE", 0) > 0]
        if "PLAN_CHANGE" in df.columns
        else pd.DataFrame()
    )
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

    # Sanitize numeric columns used here
    os_memory = sanitize_numeric(os_memory, ["SGA", "PGA", "TOTAL"])
    main_metric = sanitize_numeric(
        main_metric,
        [
            "os_cpu",
            "os_cpu_max",
            "OS_CPU",
            "OS_CPU_MAX",
            "db_cpu_ratio",
            "DB_CPU_RATIO",
            "db_wait_ratio",
            "DB_WAIT_RATIO",
            "aas",
            "AAS",
            "exec_s",
            "EXEC_S",
            "logons_s",
            "LOGONS_S",
            "sql_res_t_cs",
            "SQL_RES_T_CS",
            "read_mb_s",
            "READ_MB_S",
            "write_mb_s",
            "WRITE_MB_S",
            "read_iops",
            "READ_IOPS",
            "write_iops",
            "WRITE_IOPS",
            "redo_mb_s",
            "REDO_MB_S",
            "gc_cr_rec_s",
            "GC_CR_REC_S",
            "gc_cu_rec_s",
            "GC_CU_REC_S",
        ],
    )
    aas = sanitize_numeric(aas, ["AVG_SESS"])
    top_wait = sanitize_numeric(top_wait, ["PCTDBT", "TOTAL_TIME_S"])
    top_sql = sanitize_numeric(
        top_sql,
        [
            "ELAP_S",
            "CPU_T_S",
            "EXECS",
            "BUFFER_GETS",
            "ROWS_PROC",
            "READ_MB",
            "IO_WAIT",
            "PHY_READ_GB",
            "DIRECT_W_GB",
            "PX_SERVERS_EXECS",
            "PLAN_CHANGE",
            "PLANS",
        ],
    )

    os_dict = {row["STAT_NAME"]: row["STAT_VALUE"] for _, row in os_info.iterrows()}
    db_name = os_dict.get("DB_NAME")
    platform = os_dict.get("!PLATFORM_NAME")

    mm = main_metric.copy()
    # Expect columns: snap, end, dur_m, inst, ...
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
            format="%y/%m/%d %H:%M",  # contoh: 20/09/02 02:59
            errors="coerce"
        )   
    for snap_id in sorted(mm["snap"].unique()):
        mm_df = mm[mm["snap"] == snap_id]

        for _, row in mm_df.iterrows():
            instance = int(row["inst"])
            end_time = row["end"]
            duration_min = float(row["dur_m"])
            start_time = end_time - pd.Timedelta(minutes=duration_min)
            # Format timestamp agar konsisten
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

    # Sanitize numeric
    os_memory = sanitize_numeric(os_memory, ["SGA", "PGA", "TOTAL"])
    main_metric = sanitize_numeric(
        main_metric,
        [
            "os_cpu",
            "os_cpu_max",
            "OS_CPU",
            "OS_CPU_MAX",
            "db_cpu_ratio",
            "DB_CPU_RATIO",
            "db_wait_ratio",
            "DB_WAIT_RATIO",
            "aas",
            "AAS",
            "exec_s",
            "EXEC_S",
            "logons_s",
            "LOGONS_S",
            "sql_res_t_cs",
            "SQL_RES_T_CS",
            "read_mb_s",
            "READ_MB_S",
            "write_mb_s",
            "WRITE_MB_S",
            "read_iops",
            "READ_IOPS",
            "write_iops",
            "WRITE_IOPS",
            "redo_mb_s",
            "REDO_MB_S",
            "gc_cr_rec_s",
            "GC_CR_REC_S",
            "gc_cu_rec_s",
            "GC_CU_REC_S",
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
            errors="coerce"
        )


    for hour in sorted(mm["end"].dt.hour.unique()):
        mm_hour = mm[mm["end"].dt.hour == hour]
        snap_ids = mm_hour["snap"].unique()
        if len(mm_hour) == 0:
            continue

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
- OS CPU: {row.get('os_cpu', row.get('OS_CPU', 'N/A'))}% (max {row.get('os_cpu_max', row.get('OS_CPU_MAX', 'N/A'))}%)
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

        # I/O aggregated
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

    top_sql_df = sanitize_numeric(
        top_sql_df,
        [
            "EXECS",
            "CPU_T_S",
            "ELAP_S",
            "BUFFER_GETS",
            "READ_MB",
            "IO_WAIT",
            "PHY_READ_GB",
            "DIRECT_W_GB",
            "PLAN_CHANGE",
            "PLANS",
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

    top_sql_df = sanitize_numeric(top_sql_df, ["PLANS", "PLAN_CHANGE", "ELAP_S", "CPU_T_S"])

    df = top_sql_df[
        (top_sql_df["SNAP_ID"] >= start_snap)
        & (top_sql_df["SNAP_ID"] <= end_snap)
    ]

    if df.empty:
        return pd.DataFrame()

    by_sql = (
        df.groupby("SQL_ID")
        .agg(
            schemas=("PARSING_SCHEMA_NAME", lambda x: list(sorted(set(x)))),
            modules=("MODULE", lambda x: list(sorted(set(x)))),
            plan_hashes=("PLAN_HASH", lambda x: list(sorted(set(x)))),
            min_plans=("PLANS", "min"),
            max_plans=("PLANS", "max"),
            max_plan_change=("PLAN_CHANGE", "max"),
            snaps=("SNAP_ID", "nunique"),
            total_elapsed_s=("ELAP_S", "sum"),
            total_cpu_s=("CPU_T_S", "sum"),
        )
        .reset_index()
    )

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
        f"Total: {len(plan_df)} SQL.",
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

    if ela > 0 and (cpu / ela) > 0.7 and cpu > 10:
        labels.append("CPU-bound")

    if io_wait > 10 and (read_mb > 100 or phys_gb > 1.0):
        labels.append("IO-bound")

    if direct_w > 5.0:
        labels.append("PX-intensive")

    if plan_change > 0 or plans > 1:
        labels.append("Plan-change-risk")

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

def docs_to_langchain_documents(docs):
    lc_docs = []
    for d in docs:
        # Jika sudah Document, langsung pakai
        if isinstance(d, Document):
            lc_docs.append(d)
            continue

        # Format dict yang sudah dinormalisasi
        lc_docs.append(
            Document(
                page_content=d["content"],
                metadata=d["metadata"]
            )
        )
    return lc_docs




#def docs_to_langchain_documents(docs: List[Dict[str, Any]]) -> List[Document]:
#    """
#    Convert list[{text, metadata}] to LangChain Document objects.
#    """
#    return [Document(page_content=d["text"], metadata=d["metadata"]) for d in docs]


def upsert_documents_to_redis(vectorstore, docs, batch_size=10):
    from tqdm import tqdm

    lc_docs = docs_to_langchain_documents(docs)

    print(f"📥 Ingesting {len(lc_docs)} documents into Redis...")

    for i in tqdm(range(0, len(lc_docs), batch_size), desc="Ingesting"):
        batch = lc_docs[i:i+batch_size]
        vectorstore.add_documents(batch)

from concurrent.futures import ProcessPoolExecutor
from langchain_core.documents import Document

def embed_batch(embeddings, docs):
    """Embed a batch of documents (runs in parallel)."""
    texts = [d.page_content for d in docs]
    metas = [d.metadata for d in docs]
    vectors = embeddings.embed_documents(texts)
    return list(zip(vectors, metas, texts))


def upsert_documents_to_redis_parallel(vectorstore, docs, embeddings, workers=4, batch_size=20):
    from tqdm import tqdm

    # Pastikan docs sudah berupa Document
    lc_docs = []
    for d in docs:
        if isinstance(d, Document):
            lc_docs.append(d)
        else:
            lc_docs.append(Document(page_content=d["content"], metadata=d["metadata"]))

    print(f"📥 Parallel ingest: {len(lc_docs)} documents, {workers} workers, batch={batch_size}")

    # Bagi dokumen menjadi batch kecil
    batches = [lc_docs[i:i+batch_size] for i in range(0, len(lc_docs), batch_size)]

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = []
        for batch in batches:
            futures.append(executor.submit(embed_batch, embeddings, batch))

        for fut in tqdm(futures, desc="Parallel embedding"):
            vectors = fut.result()
            # Insert ke Redis
            for vec, meta, text in vectors:
                vectorstore.add_texts([text], metadatas=[meta])

def normalize_doc(d):
    # Jika sudah Document, biarkan
    if isinstance(d, Document):
        return d

    # Jika pakai key 'text', ubah ke 'content'
    if "text" in d:
        return {
            "content": d["text"],
            "metadata": d.get("metadata", {})
        }

    # Jika pakai key 'page_content', ubah ke 'content'
    if "page_content" in d:
        return {
            "content": d["page_content"],
            "metadata": d.get("metadata", {})
        }

    # Jika sudah benar
    if "content" in d:
        return d

    # Jika tidak dikenal
    raise ValueError(f"Unknown document format: {d}")


# ============================================================
# 6. RAG Prompts & Chains
# ============================================================

def format_docs(docs):
    """
    Gabungkan dokumen untuk dijadikan context LLM,
    sambil membatasi panjang tiap dokumen agar tidak meledak.
    """
    MAX_CHARS_PER_DOC = 5000  # ~1250 token per dokumen (kasar)

    parts = []
    for i, d in enumerate(docs, start=1):
        text = d.page_content if hasattr(d, "page_content") else d.get("content", "")
        if len(text) > MAX_CHARS_PER_DOC:
            text = text[:MAX_CHARS_PER_DOC] + "\n...[TRUNCATED]..."
        parts.append(f"[DOC {i}]\n{text}")
    return "\n\n".join(parts)

def compress_docs(docs, max_docs=8, max_chars_per_doc=5000, max_total_chars=40000):
    """
    Kompres dokumen tanpa modul LangChain tambahan.
    - Ambil dokumen paling relevan (yang pendek dulu)
    - Potong dokumen panjang
    - Batasi total context
    """
    # Urutkan dokumen berdasarkan panjang (pendek lebih relevan)
    docs_sorted = sorted(docs, key=lambda d: len(d.page_content))

    out = []
    total = 0

    for d in docs_sorted[:max_docs]:
        text = d.page_content

        # Potong per dokumen
        if len(text) > max_chars_per_doc:
            text = text[:max_chars_per_doc] + "\n...[TRUNCATED]..."

        # Batasi total context
        if total + len(text) > max_total_chars:
            remaining = max_total_chars - total
            if remaining <= 0:
                break
            text = text[:remaining] + "\n...[GLOBAL TRUNCATION]..."

        out.append(text)
        total += len(text)

    return "\n\n".join(out)


def compress_docs(docs, max_docs=8, max_chars_per_doc=5000, max_total_chars=40000):
    """
    Kompres dokumen tanpa modul LangChain tambahan.
    - Ambil dokumen paling relevan (yang pendek dulu)
    - Potong dokumen panjang
    - Batasi total context
    """
    docs_sorted = sorted(docs, key=lambda d: len(d.page_content))

    out = []
    total = 0

    for d in docs_sorted[:max_docs]:
        text = d.page_content

        if len(text) > max_chars_per_doc:
            text = text[:max_chars_per_doc] + "\n...[TRUNCATED]..."

        if total + len(text) > max_total_chars:
            remaining = max_total_chars - total
            if remaining <= 0:
                break
            text = text[:remaining] + "\n...[GLOBAL TRUNCATION]..."

        out.append(text)
        total += len(text)

    return "\n\n".join(out)


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
from awr_engine.prompting import build_prompt
def create_range_report_chain(
    llm,
    vectorstore,
    start_snap,
    end_snap,
    analysis_level="technical",
    recommendation_level="medium",
    language="id",
    style="hybrid",
):
    system_prompt = build_prompt(
        analysis_level=analysis_level,
        recommendation_level=recommendation_level,
        language=language,
        style=style,
    )

     # ❗ FIX: Hapus filter dict yang bikin RedisVL error
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 20,
            "filter": None,   # ← WAJIB None
        },
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

def rag_run_all(
        awr_data: dict,
        redis_url: str,
        index_name: str,
        start_snap: int,
        end_snap: int,
        llm_base_url: str = "http://localhost:1235/v1",
        llm_model: str = "meta-llama-3.1-8b-instruct",
    ):
        """
        Full pipeline:
        1. Build all documents (snapshot superdoc, hourly superdoc, OS info)
        2. Ingest into Redis
        3. Run RAG snapshot report
        4. Run RAG range report
        """
    
        print("🔄 Membuat embedding model...")
        embeddings = create_embeddings()
    
        print("🔄 Membuat vectorstore Redis...")
        vectorstore = create_redis_vectorstore(
            redis_url=redis_url,
            index_name=index_name,
            embeddings=embeddings,
        )
    
        print("🔄 Membuat LLM...")
        llm = create_llm(
            base_url=llm_base_url,
            model=llm_model,
            api_key="lm-studio",
            temperature=0.1,
        )
    
        print("📄 Membuat dokumen superdoc...")
        docs = []
    
        # OS Info
        docs.append(build_os_info_doc(awr_data["OS-INFORMATION"]))
    
        # Snapshot superdocs
        snapshot_docs = build_snapshot_superdocs_with_time(
            awr_data["OS-INFORMATION"],
            awr_data["MEMORY"],
            awr_data["MAIN-METRICS"],
            awr_data["AVERAGE-ACTIVE-SESSIONS"],
            awr_data["TOP-N-TIMED-EVENTS"],
            awr_data["TOP-SQL-BY-SNAPID"],
        )
        docs.extend(snapshot_docs)
    
        # Hourly superdocs
        hourly_docs = build_hourly_superdocs(
            awr_data["OS-INFORMATION"],
            awr_data["MEMORY"],
            awr_data["MAIN-METRICS"],
            awr_data["AVERAGE-ACTIVE-SESSIONS"],
            awr_data["TOP-N-TIMED-EVENTS"],
        )
        docs.extend(hourly_docs)
    
        print(f"📥 Meng‑ingest {len(docs)} dokumen ke Redis...")
        upsert_documents_to_redis(vectorstore, docs)
    
        print("🤖 Menjalankan RAG snapshot report...")
        snapshot_chain = create_snapshot_report_chain(
            llm=llm,
            vectorstore=vectorstore,
            snap_id=start_snap,
        )
        snapshot_report = snapshot_chain.invoke(start_snap)
    
        print("🤖 Menjalankan RAG range report...")
        range_chain = create_range_report_chain(
            llm=llm,
            vectorstore=vectorstore,
            start_snap=start_snap,
            end_snap=end_snap,
        )
        range_report = range_chain.invoke(None)
    
        print("✅ Selesai. Menggabungkan laporan...")
    
        final_report = f"""
    ============================================================
    AWR RAG FULL REPORT
    SNAP_ID {start_snap} → {end_snap}
    ============================================================
    
    1. Snapshot Report (SNAP_ID {start_snap})
    ------------------------------------------------------------
    {snapshot_report}
    
    2. Range Performance Report ({start_snap} → {end_snap})
    ------------------------------------------------------------
    {range_report}
    
    ============================================================
    END OF REPORT
    ============================================================
    """
    
        return final_report

