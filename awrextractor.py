#!/usr/bin/env python3
"""
awrtest2.py - AWR-Miner text file extractor to pandas DataFrames and CSVs

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

todo:
- FInalize machine learning model with orange3
- Add model saving/loading with pickle
======================================================================================================
"""

import re
import os
import sys
import argparse
from typing import List, Tuple, Optional
import Orange
import pandas as pd
import warnings
import pyarrow as pa
try:
    import openpyxl
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False

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
    p.add_argument('--csv-all', action='store_true', help='Write CSV for all sections (even if parsing uncertain)')
    p.add_argument('--excel', action='store_true', help='Write all sections to a single Excel file (requires openpyxl)')
    p.add_argument('--excel-filename', default='awr_extracted_sections.xlsx', help='Excel output filename (default: awr_extracted_sections.xlsx)')
    args = p.parse_args()
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
            num_cpus = df[df['STAT_NAME'] == '!CPU_COUNT']['STAT_VALUE'].values[0]
            print(f"Number of CPUs: {num_cpus}")
            print(f"Database Name/ID: {dbname}/{dbid}")
    
        # =====Show a small preview
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
                    print(f"Wrote sheet: {sheet_name}")
            print(f"Excel file created: {excel_path}")
        except Exception as e:
            print(f"Failed to write Excel file: {e}", file=sys.stderr)
            sys.exit(5)


        # =====Store in Redis
        #try:
        #    r = redis.Redis(host='192.168.1.233', port=6379, db=2)
        #    key_name = f"{dbname}/{dbid}:{sanitize_name(name)}"
        #    df_bytes = pa.serialize_pandas(df).to_pybytes()
        #    r.set(key_name, df_bytes)
        #    print(f"Stored section '{name}' in Redis with key '{key_name}'")
        #except Exception as e:
        #    print(f"Failed to store section '{name}' in Redis: {e}")
        #
        #r.close()
    
    # ======machine learning with orange3
    #  Data prepation
    #df = dfs.get("AVERAGE-ACTIVE-SESSIONS")    
    #pivot_df = df.pivot_table(index='SNAP_ID', columns='WAIT_CLASS', values='AVG_SESS', aggfunc='sum', fill_value=0)
    #pivot_df = pivot_df.astype(float)
    ##pivot_df.reset_index(inplace=True)
    ##print(f"Columns: {list(pivot_df.columns)}") 
#
    ## add total column 
    #pivot_df['Total'] = pivot_df[list(pivot_df.columns)].sum(axis=1)
#
    ## add stats
    #stats=[]
    #for i in pivot_df['Total']:
    #    if i  > int(num_cpus):
    #        stats.append(1)
    #    else:
    #        stats.append(0)
    #pivot_df['STATS'] = stats
    #print(f"Pivot Table: {pivot_df}")
#
#
    #
    ##target_variable =  Orange.data.DiscreteVariable("STATS", values=("OK", "NOT OK"))
#
    ##domain = Orange.data.Domain([Orange.data.ContinuousVariable("Administrative"),
    ##             Orange.data.ContinuousVariable("Application"),
    ##             Orange.data.ContinuousVariable("Cluster"),
    ##             Orange.data.ContinuousVariable("Commit"),
    ##             Orange.data.ContinuousVariable("Concurrency"),
    ##             Orange.data.ContinuousVariable("Configuration"),
    ##             Orange.data.ContinuousVariable("DB CPU"),
    ##             Orange.data.ContinuousVariable("Network"),
    ##             Orange.data.ContinuousVariable("Other"),
    ##             Orange.data.ContinuousVariable("Scheduler"),
    ##             Orange.data.ContinuousVariable("System I/O"),
    ##             Orange.data.ContinuousVariable("User I/O"),
    ##             Orange.data.ContinuousVariable("TOTAL"),
    ##             ], target_variable)
    #
    #orange_data = Orange.data.Table.from_numpy(domain=None, X=pivot_df.drop('STATS', axis=1).to_numpy(), Y=pivot_df['STATS'] )
    #learner = Orange.classification.LogisticRegressionLearner(max_iter=10000,  C=1.0)
    #from Orange.evaluation import CrossValidation, scoring
    #warnings.filterwarnings("ignore", category=DeprecationWarning, module='Orange')
    #results = CrossValidation(orange_data, [learner], k=5)
    #
    #print(" ")
    #print("================================ Orange3 Machine Learning Results ================================")
    #print(f"Result Actual: {results.actual}")
    #print(f"Result predicted: {results.predicted}")
    #accuracy = scoring.CA(results)
    #print(f"Accuracy: {accuracy}")
    #auc = scoring.AUC(results)
    #print(f"AUC: {auc}")            
    ##print(f"Results: {results}")     
    ##hasil = pivot_df
    ##hasil['prediction'] = results.predicted[0]
    ##print(f"Hasil: {hasil}")




if __name__ == '__main__':
    main()



"""
Kumpulan Code Percobaan Machine Learning dengan Orange3
siapa tahu bisa berguna di lain waktu

    
 
 
    replace_dict = {'Administrative':0,'Application':1,'Cluster':2,'Commit':3,'Concurrency':4,'Configuration':5,'DB CPU':6,'Network':7,'Other':8,'Scheduler':9,'System I/O':10,'User I/O':11,'Queueing':12}
    data = df.copy()
    data['WAIT_CLASS'] = data['WAIT_CLASS'].replace(replace_dict)
    stats = []
    for i in data['AVG_SESS']:
        if i  > num_cpus:
            stats.append(1)
        else:
            stats.append(0)
    data['STATS'] = stats
    print(data)
    feitur = data[['AVG_SESS','STATS']]
    target = data['WAIT_CLASS']
    #print(data)
    feature_1 = Orange.data.ContinuousVariable("AVG_SESS")
    feature_2 = Orange.data.ContinuousVariable("STATS")
    target_variable = Orange.data.DiscreteVariable("WAIT_CLASS", values=["Administrative","Application","Cluster","Commit","Concurrency","Configuration","DB CPU","Network","Other","Scheduler","System I/O","User I/O","Queueing"])
    domain = Orange.data.Domain([feature_1, feature_2], target_variable)           
    #domain = Orange.data.Domain([feature], target)
    orange_data = Orange.data.Table.from_numpy(domain, X=feitur, Y=target)
    test = Orange.data.pandas_compat.table_to_frame(orange_data)
    print(test)
    #print (orange_data)
    #
    ##train model with cross validation
    learner = Orange.classification.LogisticRegressionLearner(max_iter=10000, solver='lbfgs', C=1.0)
    #results_train = Orange.evaluation.TestOnTestData(orange_data[0:100], orange_data, [learner])
    #print(f"Prediction for the train data: {results_train}")
    from Orange.evaluation import CrossValidation, scoring
    results = CrossValidation(orange_data, [learner], k=5)
    print(f"Prediction for the test data: {results}")
    accuracy = scoring.CA(results)
    print(f"Accuracy: {accuracy}")
    auc = scoring.AUC(results)
    print(f"AUC: {auc}")            
    print(f"Results: {results}")     
    hasil = data
    hasil['prediction'] = results.predicted[0]
    print(f"Hasil: {hasil}")
    print(test)

    #print(f"Pivot Table: {pivot_df}")
    
    #f1_score = Orange.evaluation.scoring.F1(results)
    #print(f"F1 Score: {f1_score}")
    #precision = Orange.evaluation.scoring.Precision(results)
    #print(f"Precision: {precision}")
    #recall = Orange.evaluation.scoring.Recall(results)
    #print(f"Recall: {recall}")
    #save model to pickle
    #with open("logistic_regretions_model.pkcls", "wb") as model_file:
    #    pickle.dump(learner, model_file)
    # Load pre-trained model from pickle
    #with open("logistic_regretions_aas.pkcls", "rb") as model_file:
    #    model = pickle.load(model_file)
    #try:    
    #    prediction = model(orange_data) 
    #except Exception as e:
    #    print(f"Failed to make prediction: {e}")
    #    continue         
    #print(f"Prediction: {prediction}")
    #hasil = df
    #hasil['prediction'] = prediction
    #print(f"Hasil: {hasil}")
    #hasil.to_csv(os.path.join(args.outdir, 'section_' + sanitize_name(name) + '_with_prediction.csv'), index=False)
    #print(f"Wrote CSV with predictions: " + os.path.join(args.outdir, 'section_' + sanitize_name(name) + '_with_prediction.csv'))
    #print(prediction.classifier)
    #print(prediction.actual)
    #print(prediction.probs)
"""