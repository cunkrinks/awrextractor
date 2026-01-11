# AWR Extractor RAG Analyzer - AI Coding Guidelines

## Project Overview
This is an Oracle Database performance analysis tool that parses AWR Miner output files and generates professional performance reports using Retrieval-Augmented Generation (RAG). The system combines pandas DataFrame processing, Redis vector storage, and LM Studio LLM integration.

## Architecture
- **`awrextractor.py`**: Main CLI tool for parsing AWR text files, data processing, and RAG pipeline orchestration
- **`awr_rag.py`**: RAG engine handling document creation, embeddings, vector storage, and LLM chains
- **`awr_engine/prompting.py`**: Dynamic prompt engineering for customizable analysis levels and languages

## Key Patterns & Conventions

### Data Parsing
- Parse AWR Miner format using `~~BEGIN-SECTION~~` / `~~END-SECTION~~` markers
- Extract dash-delimited tables by detecting `---+` lines for column boundaries
- Convert to pandas DataFrames with automatic type coercion
- Normalize column names (e.g., `SNAP_ID` → `snap`, `DUR_M` → `dur_m`)

### Document Structure
- **OS Info Docs**: Single document with database environment details
- **Snapshot Superdocs**: Per-SNAP_ID documents containing CPU, AAS, wait events, top SQL, I/O, RAC metrics
- **Hourly Superdocs**: Aggregated documents combining multiple snapshots by hour

### RAG Pipeline
- Use LM Studio embeddings (`text-embedding-bge-base-en-v1.5`) via `http://localhost:1235/v1/embeddings`
- Store documents in Redis vectorstore with index `awr_index`
- Generate reports using LLM chains with customizable analysis levels

### Report Structure
Always include these sections in order:
1. Executive Summary
2. Status Summary (Sehat/Peringatan/Kritis)
3. Temuan Utama (Key Findings)
4. CPU Trend Analysis
5. Wait Event Analysis
6. AAS Trend Analysis
7. I/O Analysis
8. RAC Analysis
9. Top SQL Analysis
10. Akar Masalah (Root Cause)
11. Rekomendasi Tindakan (Recommendations)

### Language & Style
- Default to Indonesian (`language="id"`) for reports
- Use hybrid style: AWR structure + ADDM insights + AI interpretation
- Include timestamps in format: `YYYY-MM-DD HH24:MI`
- Reference SNAP_ID ranges with timestamps: `SNAP_ID 3450-3451 (2024-11-12 14:00–15:00)`

### File Naming
Reports saved as: `{dbname}_{dbid}_{start_ts}_{end_ts}_awr_rag_output.txt`
- `start_ts`/`end_ts`: `YYYYMMDD-HHMM` format
- Extract from MAIN-METRICS DataFrame `start`/`end` columns

## Common Workflows

### Basic Parsing
```bash
python awrextractor.py awr-hist.out --csv-all --outdir out_sections
```

### Full RAG Analysis
```bash
python awrextractor.py awr-hist.out --rag-run-all --save out_reports
```

### Custom Analysis
```bash
python awrextractor.py awr.out --rag-run-all \
    --analysis-level deepdive \
    --recommendation-level expert \
    --language en \
    --preset expert
```

## Dependencies & Setup
- **Redis**: Must be running on `localhost:6379`
- **LM Studio**: Running on port 1235 with:
  - LLM model (e.g., `meta-llama-3.1-8b-instruct`)
  - Embedding model (`text-embedding-bge-base-en-v1.5`)
- **Python packages**: See `requirements.txt`

## Testing Patterns
- Use simple test files like `test_llm.py`, `test_embeding.py` for component validation
- Test parsing with sample AWR output files
- Validate document ingestion with `test_ingest.py`

## Error Handling
- Check for required DataFrames (OS-INFORMATION, MAIN-METRICS, etc.) before RAG operations
- Validate document structure before Redis ingestion
- Handle missing sections gracefully with informative messages

## Code Style Notes
- Use Indonesian comments and strings for user-facing text
- Follow pandas conventions for DataFrame operations
- Use tqdm for progress bars during long operations
- Normalize metadata to strings for Redis compatibility</content>
<parameter name="filePath">d:\code\github\awrextractor\.github\copilot-instructions.md