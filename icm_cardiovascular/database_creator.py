from dash import Dash, dcc, html, Input, Output, State, callback_context, exceptions
import dash_bootstrap_components as dbc
from datetime import datetime
from pathlib import Path
from typing import Optional, Union
import pandas as pd
import re
import os
import sqlite3
import duckdb
import tempfile
import shutil

# ========= CONFIG =========
DEFAULT_ROOTS = [
    Path("/Volumes/Cerebrovascular Lab/ICM+ Main Data Folder/HBM Files"),
]

# ========= HELPERS =========
META_KEYS = [
    "Application", "Version", "Release", "CENTRE", "PROJECT", "DATE", "TIME",
    "FIRSTNAME", "LASTNAME", "COMPUTER", "BEDID", "PATIENTID", "ANONYMID", "GUID",
    "Measures", "Modalities",
]

def parse_info_file(path: Path) -> dict:
    d = {}
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                line = raw.strip()
                if not line or "=" not in line: 
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip()  # trim simple quotes/space
                d[k] = v
    except Exception as e:
        print(f"Error parsing {path}: {e}")
    return d

def split_list(s: str):
    if not isinstance(s, str) or not s.strip():
        return []
    return [x.strip() for x in s.split(",") if x.strip()]

def sanitize_token(s: str):
    return re.sub(r"[^A-Za-z0-9_]+", "_", s.replace(" ", "_"))

def iter_patient_dirs(root: Path, immediate_only=True):
    root = root.expanduser().resolve()
    if not root.exists():
        print(f"WARNING: root does not exist: {root}")
        return
    if immediate_only:
        for p in root.iterdir():
            if p.is_dir():
                yield p
    else:
        for p in root.rglob("*"):
            if p.is_dir():
                yield p

def find_info_file(subfolder: Path, case_insensitive=True, fallback_glob=None) -> Optional[Path]:
    # Prefer exact "{folder}.info"
    exact = subfolder / f"{subfolder.name}.info"
    if exact.exists() and exact.is_file():
        return exact
    # Case-insensitive check if enabled
    if case_insensitive:
        target = f"{subfolder.name}.info".lower()
        for p in subfolder.glob("*.info"):
            if p.name.lower() == target:
                return p
    # Fallback glob pattern if provided (first match wins)
    if fallback_glob:
        for p in subfolder.glob(fallback_glob):
            if p.is_file():
                return p
    return None

def scan_icm_folders(root_paths, immediate_only=True, case_insensitive=True, fallback_glob="*.info", log_callback=None):
    rows = []
    all_measures, all_modalities = set(), set()
    folders_with_info = 0
    
    if log_callback:
        log_callback(f"Starting scan of patient directories...")
    
    roots = [Path(root) for root in root_paths.split(";")]
    
    for root in roots:
        if log_callback:
            log_callback(f"Scanning root: {root}")
            
        if not root.exists():
            if log_callback:
                log_callback(f"ERROR: Root path does not exist: {root}")
            continue
        
        if immediate_only:
            patient_dirs = [d for d in root.iterdir() if d.is_dir()]
        else:
            patient_dirs = []
            for dirpath, dirnames, filenames in os.walk(root):
                for dirname in dirnames:
                    patient_dirs.append(Path(os.path.join(dirpath, dirname)))
        
        if log_callback:
            log_callback(f"Found {len(patient_dirs)} patient directories in {root}")
        
        for subfolder in patient_dirs:
            patient_folder = subfolder.name
            subfolder_path = str(subfolder)

            # always create a row (even if no .info)
            row = {
                "patient_folder": patient_folder, 
                "subfolder_path": subfolder_path,
                "has_info_file": False,  # Diagnostic flag
                "info_file_path": None   # Store the actual path if found
            }

            # Look for .info file with fallback strategies
            info_path = find_info_file(subfolder, case_insensitive, fallback_glob)
            
            if info_path:
                folders_with_info += 1
                row["has_info_file"] = True
                row["info_file_path"] = str(info_path)
                info = parse_info_file(info_path)
            else:
                # No .info file found - look for any .info as a last resort
                any_info = list(subfolder.glob("*.info"))
                if any_info:
                    info_path = any_info[0]
                    folders_with_info += 1
                    row["has_info_file"] = True
                    row["info_file_path"] = str(info_path)
                    info = parse_info_file(info_path)
                else:
                    info = {}

            # fill metadata (missing -> NaN)
            for k in META_KEYS:
                row[k] = info.get(k, None)

            # Process measures and modalities
            measures_list = split_list(row.get("Measures"))
            modalities_list = split_list(row.get("Modalities"))
            row["_measures_list"] = measures_list
            row["_modalities_list"] = modalities_list

            all_measures.update(measures_list)
            all_modalities.update(modalities_list)

            rows.append(row)

    if log_callback:
        log_callback(f"Scan complete: {len(rows)} folders found, {folders_with_info} with .info files")

    # Build DataFrame
    df = pd.DataFrame(rows)

    # make sure meta columns exist even if none were seen
    for k in META_KEYS:
        if k not in df.columns:
            df[k] = pd.NA

    # one-hot: 1 if listed, NaN otherwise (you can fillna(0) later)
    for m in sorted(all_measures):
        col_name = f"measure_{sanitize_token(m)}".lower()  # Normalize column name to lowercase
        df[col_name] = df["_measures_list"].apply(lambda lst, m=m: 1 if m in (lst or []) else pd.NA)

    for mod in sorted(all_modalities):
        col_name = f"modalities_{sanitize_token(mod)}".lower()  # Normalize column name to lowercase
        df[col_name] = df["_modalities_list"].apply(lambda lst, mod=mod: 1 if mod in (lst or []) else pd.NA)

    # optional timestamp
    def parse_ts(d, t):
        d = str(d) if d is not None else ""
        t = str(t) if t is not None else ""
        if len(d) == 8 and len(t) in (4,5,6):
            try:
                return pd.to_datetime(d + t.zfill(6), format="%Y%m%d%H%M%S", utc=True)
            except Exception:
                return pd.NaT
        return pd.NaT
    
    df["timestamp_utc"] = [parse_ts(d, t) for d, t in zip(df["DATE"], df["TIME"])]

    # cleanup helper columns
    df = df.drop(columns=["_measures_list", "_modalities_list"])
    
    # Normalize all column names to lowercase to avoid case-sensitivity issues
    df.columns = [col.lower() for col in df.columns]
    
    if log_callback:
        log_callback(f"DataFrame created with {len(df)} rows and {len(df.columns)} columns")
        log_callback(f"All column names normalized to lowercase to prevent duplicates")
    
    # Replace NA values with 0 in numeric columns when has_info_file is True
    if log_callback:
        log_callback("Processing NA values for numeric columns...")
    
    # Check if has_info_file column exists and has the right type
    if 'has_info_file' not in df.columns:
        if log_callback:
            log_callback("Warning: has_info_file column not found in DataFrame")
        return df
    
    # Ensure has_info_file is boolean type
    df['has_info_file'] = df['has_info_file'].astype(bool)
    
    # Get rows where has_info_file is True
    info_file_mask = df['has_info_file'] == True
    
    # Get all measure_ and modalities_ columns (they should be numeric)
    numeric_columns = [col for col in df.columns if col.startswith(('measure_', 'modalities_'))]
    
    if log_callback:
        log_callback(f"Found {len(numeric_columns)} numeric columns to process")
        log_callback(f"Found {info_file_mask.sum()} rows with has_info_file=True")
        log_callback(f"Sample data - has_info_file values: {df['has_info_file'].value_counts().to_dict()}")
    
    # Count NA values before replacement
    na_counts_before = {col: df[info_file_mask][col].isna().sum() for col in numeric_columns}
    total_na_before = sum(na_counts_before.values())
    
    if log_callback:
        log_callback(f"Found {total_na_before} NA values in numeric columns for rows with has_info_file=True")
    
    # Replace NA values with 0 only for rows where has_info_file is True
    for col in numeric_columns:
        # Using loc to target specific rows and columns
        df.loc[info_file_mask, col] = df.loc[info_file_mask, col].fillna(0)
    
    # Count NA values after replacement to verify
    na_counts_after = {col: df[info_file_mask][col].isna().sum() for col in numeric_columns}
    total_na_after = sum(na_counts_after.values())
    
    if log_callback:
        log_callback(f"Replaced {total_na_before - total_na_after} NA values with 0")
        log_callback("Finished processing NA values")
    
    return df

def create_database(df, db_path, db_type="duckdb", table_name="main_cardiovascular_table", log_callback=None):
    if log_callback:
        log_callback(f"Creating {db_type} database at {db_path}")
    
    try:
        if db_type.lower() == "duckdb":
            conn = duckdb.connect(db_path)
            # Use lowercase table name to be consistent
            table_name_lower = table_name.lower()
            conn.execute(f"DROP TABLE IF EXISTS {table_name_lower}")
            conn.register('temp_df', df)
            conn.execute(f"CREATE TABLE {table_name_lower} AS SELECT * FROM temp_df")
            conn.close()
            if log_callback:
                log_callback(f"Successfully created DuckDB database with {len(df)} records")
            return True
        else:  # SQLite
            conn = sqlite3.connect(db_path)
            # Use lowercase table name to be consistent
            table_name_lower = table_name.lower()
            df.to_sql(table_name_lower, conn, index=False, if_exists="replace")
            conn.close()
            if log_callback:
                log_callback(f"Successfully created SQLite database with {len(df)} records")
            return True
    except Exception as e:
        if log_callback:
            log_callback(f"Error creating database: {str(e)}", "error")
        return False

def update_database(df, db_path, table_name="main_cardiovascular_table", log_callback=None):
    if log_callback:
        log_callback(f"Updating database at {db_path}")
    
    try:
        # Detect database type
        if db_path.lower().endswith('.duckdb'):
            db_type = 'duckdb'
        else:
            db_type = 'sqlite'
        
        # Normalize table name to lowercase for consistency
        table_name = table_name.lower()
        
        if db_type == 'duckdb':
            conn = duckdb.connect(db_path)
            
            # Check if table exists - using DuckDB's information_schema instead of sqlite_master
            query = f"SELECT table_name FROM information_schema.tables WHERE lower(table_name)=lower('{table_name}')"
            if log_callback:
                log_callback(f"Checking for table: {table_name}")
                log_callback(f"Using query: {query}")
            
            result = conn.execute(query).fetchall()
            if log_callback:
                log_callback(f"Table check result: {result}")
            
            table_exists = len(result) > 0
            
            if table_exists:
                # Get existing paths
                existing_paths = conn.execute(f"SELECT subfolder_path FROM {table_name}").df()["subfolder_path"].tolist()
                new_paths = df["subfolder_path"].tolist()
                
                # Find duplicates
                duplicates = set(existing_paths) & set(new_paths)
                if duplicates:
                    if log_callback:
                        log_callback(f"Found {len(duplicates)} duplicate subfolder paths")
                        for path in list(duplicates)[:5]:
                            log_callback(f"  - {path}")
                        if len(duplicates) > 5:
                            log_callback(f"  ... and {len(duplicates)-5} more")
                    
                    # Filter out duplicates
                    original_count = len(df)
                    df = df[~df["subfolder_path"].isin(duplicates)]
                    if log_callback:
                        log_callback(f"Filtered out {original_count - len(df)} duplicate records")
                    
                    if len(df) == 0:
                        if log_callback:
                            log_callback("No new records to add after filtering duplicates")
                        conn.close()
                        return True
                
                # Get the existing table columns
                if log_callback:
                    log_callback("Checking table schema for column compatibility...")
                
                # Get existing columns in the table
                existing_cols_query = f"SELECT column_name FROM information_schema.columns WHERE table_name='{table_name.lower()}'"
                existing_cols = [row[0] for row in conn.execute(existing_cols_query).fetchall()]
                
                # Get columns in the new data
                new_cols = df.columns.tolist()
                
                if log_callback:
                    log_callback(f"Existing columns: {existing_cols[:5]}... ({len(existing_cols)} total)")
                    log_callback(f"New columns: {new_cols[:5]}... ({len(new_cols)} total)")
                
                # Normalize column names to lowercase for comparison
                existing_cols_lower = [c.lower() for c in existing_cols]
                new_cols_lower = [c.lower() for c in new_cols]
                
                # Find columns that exist in both (case-insensitive comparison)
                common_cols = [col for col in new_cols if col.lower() in existing_cols_lower]
                
                # Find missing columns (in new data but not in table) - case-insensitive
                missing_cols = [col for col in new_cols if col.lower() not in existing_cols_lower]
                
                # Find extra columns (in table but not in new data) - case-insensitive
                extra_cols = [col for col in existing_cols if col.lower() not in new_cols_lower]
                
                if missing_cols:
                    if log_callback:
                        log_callback(f"Found {len(missing_cols)} columns in new data not in existing table: {missing_cols[:5]}...")
                    
                    # Add missing columns to the table
                    for col in missing_cols:
                        col_type = "VARCHAR" if df[col].dtype == 'object' else "DOUBLE"
                        alter_query = f"ALTER TABLE {table_name} ADD COLUMN \"{col}\" {col_type}"
                        if log_callback:
                            log_callback(f"Adding column: {alter_query}")
                        conn.execute(alter_query)
                
                if extra_cols:
                    if log_callback:
                        log_callback(f"Found {len(extra_cols)} columns in existing table not in new data: {extra_cols[:5]}...")
                        log_callback("These columns will be NULL for the new records")
                
                # Append new records to existing table, using common column names
                conn.register('new_data', df)
                
                # Build a query that explicitly lists columns to insert
                cols_str = ", ".join([f'"{col}"' for col in new_cols])
                select_str = ", ".join([f'"{col}"' for col in new_cols])
                
                insert_query = f"INSERT INTO {table_name} ({cols_str}) SELECT {select_str} FROM new_data"
                if log_callback:
                    log_callback(f"Executing insert query: {insert_query}")
                
                conn.execute(insert_query)
                if log_callback:
                    log_callback(f"Added {len(df)} new records to existing database")
                conn.close()
            else:
                if log_callback:
                    log_callback(f"Table '{table_name}' not found in existing database. Creating new table.")
                conn.register('new_data', df)
                conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM new_data")
                if log_callback:
                    log_callback(f"Created new table with {len(df)} records")
                conn.close()
        else:
            # SQLite approach
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Check if table exists
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
            table_exists = cursor.fetchone()
            
            if table_exists:
                # Get existing paths
                existing_df = pd.read_sql(f"SELECT subfolder_path FROM {table_name}", conn)
                existing_paths = existing_df["subfolder_path"].tolist()
                new_paths = df["subfolder_path"].tolist()
                
                # Find duplicates
                duplicates = set(existing_paths) & set(new_paths)
                if duplicates:
                    if log_callback:
                        log_callback(f"Found {len(duplicates)} duplicate subfolder paths")
                        for path in list(duplicates)[:5]:
                            log_callback(f"  - {path}")
                        if len(duplicates) > 5:
                            log_callback(f"  ... and {len(duplicates)-5} more")
                    
                    # Filter out duplicates
                    original_count = len(df)
                    df = df[~df["subfolder_path"].isin(duplicates)]
                    if log_callback:
                        log_callback(f"Filtered out {original_count - len(df)} duplicate records")
                    
                    if len(df) == 0:
                        if log_callback:
                            log_callback("No new records to add after filtering duplicates")
                        conn.close()
                        return True
                
                # Get the existing table columns
                if log_callback:
                    log_callback("Checking table schema for column compatibility...")
                
                # Get existing columns in the table
                cursor.execute(f"PRAGMA table_info({table_name})")
                existing_cols = [row[1] for row in cursor.fetchall()]
                
                # Get columns in the new data
                new_cols = df.columns.tolist()
                
                if log_callback:
                    log_callback(f"Existing columns: {existing_cols[:5]}... ({len(existing_cols)} total)")
                    log_callback(f"New columns: {new_cols[:5]}... ({len(new_cols)} total)")
                
                # Normalize column names to lowercase for comparison
                existing_cols_lower = [c.lower() for c in existing_cols]
                new_cols_lower = [c.lower() for c in new_cols]
                
                # Find columns that exist in both (case-insensitive comparison)
                common_cols = [col for col in new_cols if col.lower() in existing_cols_lower]
                
                # Find missing columns (in new data but not in table) - case-insensitive
                missing_cols = [col for col in new_cols if col.lower() not in existing_cols_lower]
                
                # Find extra columns (in table but not in new data) - case-insensitive
                extra_cols = [col for col in existing_cols if col.lower() not in new_cols_lower]
                
                if missing_cols:
                    if log_callback:
                        log_callback(f"Found {len(missing_cols)} columns in new data not in existing table: {missing_cols[:5]}...")
                    
                    # Add missing columns to the table
                    for col in missing_cols:
                        col_type = "TEXT" if df[col].dtype == 'object' else "REAL"
                        alter_query = f"ALTER TABLE {table_name} ADD COLUMN \"{col}\" {col_type}"
                        if log_callback:
                            log_callback(f"Adding column: {alter_query}")
                        cursor.execute(alter_query)
                
                if extra_cols:
                    if log_callback:
                        log_callback(f"Found {len(extra_cols)} columns in existing table not in new data: {extra_cols[:5]}...")
                        log_callback("These columns will be NULL for the new records")
                
                # Create a temporary table with the right schema
                temp_table = f"temp_{table_name}"
                if log_callback:
                    log_callback(f"Creating temporary table {temp_table} with matching schema")
                
                # Use only the columns that exist in both the dataframe and the target table
                df_to_insert = df[common_cols].copy()
                df_to_insert.to_sql(temp_table, conn, if_exists='replace', index=False)
                
                # Insert from temp table to main table, specifying columns
                cols_str = ", ".join([f'"{col}"' for col in common_cols])
                insert_query = f"INSERT INTO {table_name} ({cols_str}) SELECT {cols_str} FROM {temp_table}"
                
                if log_callback:
                    log_callback(f"Executing insert query: {insert_query}")
                
                cursor.execute(insert_query)
                cursor.execute(f"DROP TABLE {temp_table}")
                conn.commit()
                
                if log_callback:
                    log_callback(f"Added {len(df)} new records to existing database")
            else:
                if log_callback:
                    log_callback(f"Table '{table_name}' not found in existing database. Creating new table.")
                df.to_sql(table_name, conn, if_exists='replace', index=False)
                if log_callback:
                    log_callback(f"Created new table with {len(df)} records")
            
            conn.close()
        return True
    except Exception as e:
        if log_callback:
            log_callback(f"Error updating database: {str(e)}", "error")
        return False

# ========= DASH APP =========
def create_app():
    """Create and configure the Dash app for database creation"""
    app = Dash(__name__, 
               external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
               suppress_callback_exceptions=True)

    app.layout = html.Div([
        dbc.Navbar(
            dbc.Container([
                html.Div([
                    html.I(className="fas fa-database me-2"),
                    html.Span("ICM+ Database Creator", className="navbar-brand mb-0 h1")
                ])
            ]),
            color="primary",
            dark=True,
            className="mb-3",
        ),
        
        dbc.Container([
            dbc.Card([
                dbc.CardHeader([
                    html.H4("Database Operations", className="mb-0")
                ]),
                dbc.CardBody([
                    # Operation mode selection
                    dbc.Row([
                        dbc.Col([
                            html.Label("Select Operation", className="mb-2"),
                            dbc.RadioItems(
                                options=[
                                    {"label": "Create New Database", "value": "create"},
                                    {"label": "Update Existing Database", "value": "update"}
                                ],
                                value="create",
                                id="operation-mode",
                                inline=True,
                                className="mb-3"
                            )
                        ])
                    ]),
                    
                    # Dynamic form container
                    html.Div(id="dynamic-form-container"),
                    
                    # Status and logs
                    dbc.Row([
                        dbc.Col([
                            html.H5("Operation Status", className="mt-4 mb-3"),
                            dbc.Card([
                                dbc.CardBody([
                                    html.Div(id="status-log", className="small", style={"maxHeight": "300px", "overflowY": "auto"})
                                ])
                            ])
                        ])
                    ])
                ])
            ])
        ]),
        
        # Hidden stores for intermediate data
        dcc.Store(id="scan-data"),
        dcc.Loading(id="loading-scan", type="circle", children=html.Div(id="loading-output-scan"))
    ])

    # Callback for dynamic form
    @app.callback(
        Output("dynamic-form-container", "children"),
        Input("operation-mode", "value")
    )
    def update_form(operation_mode):
        if operation_mode == "create":
            return html.Div([
                # Database type
                dbc.Row([
                    dbc.Col([
                        html.Label("Database Type", className="mb-2"),
                        dbc.RadioItems(
                            options=[
                                {"label": "SQLite", "value": "sqlite"},
                                {"label": "DuckDB", "value": "duckdb"}
                            ],
                            value="duckdb",
                            id="database-type",
                            inline=True,
                            className="mb-3"
                        )
                    ])
                ]),
                
                # Database file path
                dbc.Row([
                    dbc.Col([
                        html.Label("Output Database File Path", className="mb-2"),
                        dbc.InputGroup([
                            dbc.Input(id="database-path", 
                                      placeholder="Path to save the database file",
                                      value=f"CardiovascularLab_DB_{datetime.now().strftime('%Y%m%d_%H%M%S')}.duckdb"),
                            dbc.InputGroupText(".duckdb", id="db-extension")
                        ]),
                        html.Small("Enter a path where the database file will be saved", className="text-muted mb-3")
                    ])
                ]),
                
                # ICM+ directory path
                dbc.Row([
                    dbc.Col([
                        html.Label("ICM+ Files Directory Path", className="mb-2"),
                        dbc.Textarea(id="icm-dirs", 
                                   placeholder="Enter one or more directories containing ICM+ files (separate multiple paths with semicolons)",
                                   value="/Volumes/Cerebrovascular Lab/ICM+ Main Data Folder/HBM Files",
                                   style={"height": "100px"}),
                        html.Small([
                            "Enter one or more paths to folders containing ICM+ files. ",
                            "Separate multiple paths with semicolons (;)."
                        ], className="text-muted mb-3")
                    ])
                ]),
                
                # Scan options
                dbc.Row([
                    dbc.Col([
                        dbc.Checkbox(id="immediate-only", 
                                    label="Scan immediate subdirectories only (don't scan recursively)", 
                                    value=True, 
                                    className="mb-2"),
                        dbc.Checkbox(id="case-insensitive", 
                                    label="Case-insensitive .info file matching", 
                                    value=True, 
                                    className="mb-2"),
                        dbc.Checkbox(id="save-csv", 
                                    label="Save data as CSV file", 
                                    value=True, 
                                    className="mb-3"),
                    ])
                ]),
                
                # Conditional CSV path
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.Label("Output CSV File Path", className="mb-2"),
                            dbc.Input(id="csv-path", 
                                     placeholder="Path to save the CSV file",
                                     value=f"catalog_tables_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"),
                            html.Small("Enter a path where the CSV file will be saved", className="text-muted mb-3")
                        ], id="csv-path-container")
                    ])
                ]),
                
                # Process button
                dbc.Row([
                    dbc.Col([
                        dbc.Button("Scan Directory & Create Database", 
                                  id="process-button", 
                                  color="primary", 
                                  className="mt-3")
                    ])
                ])
            ])
        else:  # update mode
            return html.Div([
                # Hidden database type field (will be auto-detected from file extension)
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            dbc.RadioItems(
                                options=[
                                    {"label": "SQLite", "value": "sqlite"},
                                    {"label": "DuckDB", "value": "duckdb"}
                                ],
                                value="duckdb",
                                id="database-type",
                                inline=True,
                                className="mb-3"
                            )
                        ], style={"display": "none"})  # Hide this element
                    ])
                ]),
                
                # Existing database path
                dbc.Row([
                    dbc.Col([
                        html.Label("Existing Database File Path", className="mb-2"),
                        dbc.Input(id="database-path", 
                                 placeholder="Path to the existing database file",
                                 value="CardiovascularLab_DB.duckdb"),
                        html.Small("Enter the path to the existing database file", className="text-muted mb-3")
                    ])
                ]),
                
                # ICM+ directory path
                dbc.Row([
                    dbc.Col([
                        html.Label("ICM+ Files Directory Path", className="mb-2"),
                        dbc.Textarea(id="icm-dirs", 
                                   placeholder="Enter one or more directories containing ICM+ files (separate multiple paths with semicolons)",
                                   value="/Volumes/Cerebrovascular Lab/ICM+ Main Data Folder/HBM Files",
                                   style={"height": "100px"}),
                        html.Small([
                            "Enter one or more paths to folders containing ICM+ files. ",
                            "Separate multiple paths with semicolons (;)."
                        ], className="text-muted mb-3")
                    ])
                ]),
                
                # Scan options
                dbc.Row([
                    dbc.Col([
                        dbc.Checkbox(id="immediate-only", 
                                    label="Scan immediate subdirectories only (don't scan recursively)", 
                                    value=True, 
                                    className="mb-2"),
                        dbc.Checkbox(id="case-insensitive", 
                                    label="Case-insensitive .info file matching", 
                                    value=True, 
                                    className="mb-2"),
                        dbc.Checkbox(id="save-csv", 
                                    label="Save data as CSV file", 
                                    value=True, 
                                    className="mb-3"),
                    ])
                ]),
                
                # Conditional CSV path
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.Label("Output CSV File Path", className="mb-2"),
                            dbc.Input(id="csv-path", 
                                     placeholder="Path to save the CSV file",
                                     value=f"update_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"),
                            html.Small("Enter a path where the CSV file will be saved", className="text-muted mb-3")
                        ], id="csv-path-container")
                    ])
                ]),
                
                # Process button
                dbc.Row([
                    dbc.Col([
                        dbc.Button("Scan Directory & Update Database", 
                                  id="process-button", 
                                  color="primary", 
                                  className="mt-3")
                    ])
                ])
            ])

    # Toggle CSV path visibility based on checkbox
    @app.callback(
        Output("csv-path-container", "style"),
        Input("save-csv", "value")
    )
    def toggle_csv_path(save_csv):
        if save_csv:
            return {"display": "block"}
        return {"display": "none"}

    # Update database file extension based on type selection
    @app.callback(
        Output("db-extension", "children"),
        Input("database-type", "value")
    )
    def update_db_extension(db_type):
        if not db_type or db_type == "duckdb":
            return ".duckdb"
        return ".db"

    # Auto-update database type based on file extension in update mode
    @app.callback(
        Output("database-type", "value"),
        Input("operation-mode", "value"),
        Input("database-path", "value"),
        prevent_initial_call=True
    )
    def update_database_type(operation_mode, db_path):
        if operation_mode != "update" or not db_path:
            raise exceptions.PreventUpdate
        
        if db_path.lower().endswith('.duckdb'):
            return "duckdb"
        return "sqlite"

    # Process the database operation
    @app.callback(
        Output("status-log", "children"),
        Output("scan-data", "data"),
        Output("loading-output-scan", "children"),
        Input("process-button", "n_clicks"),
        State("operation-mode", "value"),
        State("database-path", "value"),
        State("icm-dirs", "value"),
        State("immediate-only", "value"),
        State("case-insensitive", "value"),
        State("save-csv", "value"),
        State("csv-path", "value"),
        State("database-type", "value"),
        prevent_initial_call=True
    )
    def process_database_operation(n_clicks, operation_mode, db_path, icm_dirs, 
                                   immediate_only, case_insensitive, save_csv, 
                                   csv_path, db_type):
        if not n_clicks:
            raise exceptions.PreventUpdate
        
        # Handle potentially None db_type for update mode
        if operation_mode == "update" and not db_type:
            # For update, detect the type from the file extension
            if db_path.lower().endswith('.duckdb'):
                db_type = 'duckdb'
            else:
                db_type = 'sqlite'
        
        # Initialize log messages
        log_messages = []
        def add_log(message, level="info"):
            timestamp = datetime.now().strftime("%H:%M:%S")
            if level == "error":
                color = "danger"
            elif level == "warning":
                color = "warning"
            elif level == "success":
                color = "success"
            else:
                color = "info"
            
            log_messages.append(
                html.Div([
                    html.Span(f"[{timestamp}] ", className="text-muted"),
                    html.Span(message, className=f"text-{color}")
                ], className="mb-1")
            )
        
        try:
            # Validate inputs
            if not icm_dirs:
                add_log("Error: ICM+ files directory path is required", "error")
                return html.Div(log_messages), None, ""
            
            if not db_path:
                add_log("Error: Database file path is required", "error")
                return html.Div(log_messages), None, ""
            
            # Ensure database file has correct extension
            if operation_mode == "create":
                if (not db_type or db_type == "duckdb") and not db_path.lower().endswith('.duckdb'):
                    db_path = f"{db_path}.duckdb"
                elif db_type == "sqlite" and not db_path.lower().endswith('.db'):
                    db_path = f"{db_path}.db"
            
            add_log(f"Starting {operation_mode} operation...")
            add_log(f"ICM+ directories: {icm_dirs}")
            add_log(f"Database path: {db_path}")
            
            # Scan the ICM+ folders
            add_log("Scanning ICM+ folders...")
            df = scan_icm_folders(
                icm_dirs,
                immediate_only=immediate_only,
                case_insensitive=case_insensitive,
                log_callback=add_log
            )
            
            if len(df) == 0:
                add_log("No data found in the specified directories", "warning")
                return html.Div(log_messages), None, ""
            
            # Save to CSV if requested
            if save_csv and csv_path:
                try:
                    df.to_csv(csv_path, index=False)
                    add_log(f"Successfully saved data to CSV: {csv_path}", "success")
                except Exception as e:
                    add_log(f"Error saving CSV: {str(e)}", "error")
            
            # Process database operation
            if operation_mode == "create":
                success = create_database(
                    df, 
                    db_path, 
                    db_type=db_type,
                    log_callback=add_log
                )
            else:  # update
                success = update_database(
                    df, 
                    db_path,
                    log_callback=add_log
                )
            
            if success:
                add_log(f"Database operation completed successfully!", "success")
            
            return html.Div(log_messages), df.to_json(date_format='iso', orient='split'), ""
        
        except Exception as e:
            add_log(f"Error during operation: {str(e)}", "error")
            import traceback
            add_log(traceback.format_exc(), "error")
            return html.Div(log_messages), None, ""

    return app

def create_app():
    """Create and configure the Dash app for database creation"""
    app = Dash(__name__, 
               external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
               suppress_callback_exceptions=True)

    app.layout = html.Div([
        dbc.Navbar(
            dbc.Container([
                html.Div([
                    html.I(className="fas fa-database me-2"),
                    html.Span("ICM+ Database Creator", className="navbar-brand mb-0 h1")
                ])
            ]),
            color="primary",
            dark=True,
            className="mb-3",
        ),
        
        dbc.Container([
            dbc.Card([
                dbc.CardHeader([
                    html.H4("Database Operations", className="mb-0")
                ]),
                dbc.CardBody([
                    # Operation mode selection
                    dbc.Row([
                        dbc.Col([
                            html.Label("Select Operation", className="mb-2"),
                            dbc.RadioItems(
                                options=[
                                    {"label": "Create New Database", "value": "create"},
                                    {"label": "Update Existing Database", "value": "update"}
                                ],
                                value="create",
                                id="operation-mode",
                                inline=True,
                                className="mb-3"
                            )
                        ])
                    ]),
                    
                    # Dynamic form container
                    html.Div(id="dynamic-form-container"),
                    
                    # Status and logs
                    dbc.Row([
                        dbc.Col([
                            html.H5("Operation Status", className="mt-4 mb-3"),
                            dbc.Card([
                                dbc.CardBody([
                                    html.Div(id="status-log", className="small", style={"maxHeight": "300px", "overflowY": "auto"})
                                ])
                            ])
                        ])
                    ])
                ])
            ])
        ]),
        
        # Hidden stores for intermediate data
        dcc.Store(id="scan-data"),
        dcc.Loading(id="loading-scan", type="circle", children=html.Div(id="loading-output-scan"))
    ])

    # Callback for dynamic form
    @app.callback(
        Output("dynamic-form-container", "children"),
        Input("operation-mode", "value")
    )
    def update_form(operation_mode):
        if operation_mode == "create":
            return html.Div([
                # Database type
                dbc.Row([
                    dbc.Col([
                        html.Label("Database Type", className="mb-2"),
                        dbc.RadioItems(
                            options=[
                                {"label": "SQLite", "value": "sqlite"},
                                {"label": "DuckDB", "value": "duckdb"}
                            ],
                            value="duckdb",
                            id="database-type",
                            inline=True,
                            className="mb-3"
                        )
                    ])
                ]),
                
                # Database file path
                dbc.Row([
                    dbc.Col([
                        html.Label("Output Database File Path", className="mb-2"),
                        dbc.InputGroup([
                            dbc.Input(id="database-path", 
                                      placeholder="Path to save the database file",
                                      value=f"/path/to/CardiovascularLab_DB_{datetime.now().strftime('%Y%m%d_%H%M%S')}.duckdb"),
                            dbc.InputGroupText(".duckdb", id="db-extension")
                        ]),
                        html.Small("Enter a path where the database file will be saved. Example: /Desktop/ICMPlus/Databases/database.duckdb", className="text-muted mb-3")
                    ])
                ]),
                
                # ICM+ directory path
                dbc.Row([
                    dbc.Col([
                        html.Label("ICM+ Files Directory Path", className="mb-2"),
                        dbc.Textarea(id="icm-dirs", 
                                   placeholder="Enter one or more directories containing ICM+ files (separate multiple paths with semicolons)",
                                   value="/Path/to/ICM+ Main Data Folder/HBM Files",
                                   style={"height": "100px"}),
                        html.Small([
                            "Enter one or more paths to folders containing ICM+ files. ",
                            "Separate multiple paths with semicolons (;)."
                        ], className="text-muted mb-3")
                    ])
                ]),
                
                # Scan options
                dbc.Row([
                    dbc.Col([
                        dbc.Checkbox(id="immediate-only", 
                                    label="Scan immediate subdirectories only (don't scan recursively)", 
                                    value=True, 
                                    className="mb-2"),
                        dbc.Checkbox(id="case-insensitive", 
                                    label="Case-insensitive .info file matching", 
                                    value=True, 
                                    className="mb-2"),
                        dbc.Checkbox(id="save-csv", 
                                    label="Save data as CSV file", 
                                    value=True, 
                                    className="mb-3"),
                    ])
                ]),
                
                # Conditional CSV path
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.Label("Output CSV File Path", className="mb-2"),
                            dbc.Input(id="csv-path", 
                                     placeholder="Path to save the CSV file",
                                     value=f"path/to/catalog_tables_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"),
                            html.Small("Enter a path where the CSV file will be saved. Example: /Desktop/ICMPlus/catalogFolder/catalog_tables.csv", className="text-muted mb-3")
                        ], id="csv-path-container")
                    ])
                ]),
                
                # Process button
                dbc.Row([
                    dbc.Col([
                        dbc.Button("Scan Directory & Create Database", 
                                  id="process-button", 
                                  color="primary", 
                                  className="mt-3")
                    ])
                ])
            ])
        else:  # update mode
            return html.Div([
                # Hidden database type field (will be auto-detected from file extension)
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            dbc.RadioItems(
                                options=[
                                    {"label": "SQLite", "value": "sqlite"},
                                    {"label": "DuckDB", "value": "duckdb"}
                                ],
                                value="duckdb",
                                id="database-type",
                                inline=True,
                                className="mb-3"
                            )
                        ], style={"display": "none"})  # Hide this element
                    ])
                ]),
                
                # Existing database path
                dbc.Row([
                    dbc.Col([
                        html.Label("Existing Database File Path", className="mb-2"),
                        dbc.Input(id="database-path", 
                                 placeholder="Path to the existing database file",
                                 value="/path/to/CardiovascularLab_DB.duckdb"),
                        html.Small("Enter the path to the existing database file, example: /Desktop/ICMPlus/Databases/CardiovascularLab_DB.duckdb", className="text-muted mb-3")
                    ])
                ]),
                
                # ICM+ directory path
                dbc.Row([
                    dbc.Col([
                        html.Label("ICM+ Files Directory Path", className="mb-2"),
                        dbc.Textarea(id="icm-dirs", 
                                   placeholder="Enter one or more directories containing ICM+ files (separate multiple paths with semicolons)",
                                   value="/Volumes/Cerebrovascular Lab/ICM+ Main Data Folder/HBM Files",
                                   style={"height": "100px"}),
                        html.Small([
                            "Enter one or more paths to folders containing ICM+ files. ",
                            "Separate multiple paths with semicolons (;)."
                        ], className="text-muted mb-3")
                    ])
                ]),
                
                # Scan options
                dbc.Row([
                    dbc.Col([
                        dbc.Checkbox(id="immediate-only", 
                                    label="Scan immediate subdirectories only (don't scan recursively)", 
                                    value=True, 
                                    className="mb-2"),
                        dbc.Checkbox(id="case-insensitive", 
                                    label="Case-insensitive .info file matching", 
                                    value=True, 
                                    className="mb-2"),
                        dbc.Checkbox(id="save-csv", 
                                    label="Save data as CSV file", 
                                    value=True, 
                                    className="mb-3"),
                    ])
                ]),
                
                # Conditional CSV path
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.Label("Output CSV File Path", className="mb-2"),
                            dbc.Input(id="csv-path", 
                                     placeholder="Path to save the CSV file",
                                     value=f"/path/to/update_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"),
                            html.Small("Enter a path where the CSV file will be saved, example: /Desktop/ICMPlus/catalogFolder_update/update_data.csv", className="text-muted mb-3")
                        ], id="csv-path-container")
                    ])
                ]),
                
                # Process button
                dbc.Row([
                    dbc.Col([
                        dbc.Button("Scan Directory & Update Database", 
                                  id="process-button", 
                                  color="primary", 
                                  className="mt-3")
                    ])
                ])
            ])

    # Toggle CSV path visibility based on checkbox
    @app.callback(
        Output("csv-path-container", "style"),
        Input("save-csv", "value")
    )
    def toggle_csv_path(save_csv):
        if save_csv:
            return {"display": "block"}
        return {"display": "none"}

    # Update database file extension based on type selection
    @app.callback(
        Output("db-extension", "children"),
        Input("database-type", "value")
    )
    def update_db_extension(db_type):
        if not db_type or db_type == "duckdb":
            return ".duckdb"
        return ".db"

    # Auto-update database type based on file extension in update mode
    @app.callback(
        Output("database-type", "value"),
        Input("operation-mode", "value"),
        Input("database-path", "value"),
        prevent_initial_call=True
    )
    def update_database_type(operation_mode, db_path):
        if operation_mode != "update" or not db_path:
            raise exceptions.PreventUpdate
        
        if db_path.lower().endswith('.duckdb'):
            return "duckdb"
        return "sqlite"

    # Process the database operation
    @app.callback(
        Output("status-log", "children"),
        Output("scan-data", "data"),
        Output("loading-output-scan", "children"),
        Input("process-button", "n_clicks"),
        State("operation-mode", "value"),
        State("database-path", "value"),
        State("icm-dirs", "value"),
        State("immediate-only", "value"),
        State("case-insensitive", "value"),
        State("save-csv", "value"),
        State("csv-path", "value"),
        State("database-type", "value"),
        prevent_initial_call=True
    )
    def process_database_operation(n_clicks, operation_mode, db_path, icm_dirs, 
                                   immediate_only, case_insensitive, save_csv, 
                                   csv_path, db_type):
        if not n_clicks:
            raise exceptions.PreventUpdate
        
        # Handle potentially None db_type for update mode
        if operation_mode == "update" and not db_type:
            # For update, detect the type from the file extension
            if db_path.lower().endswith('.duckdb'):
                db_type = 'duckdb'
            else:
                db_type = 'sqlite'
        
        # Initialize log messages
        log_messages = []
        def add_log(message, level="info"):
            timestamp = datetime.now().strftime("%H:%M:%S")
            if level == "error":
                color = "danger"
            elif level == "warning":
                color = "warning"
            elif level == "success":
                color = "success"
            else:
                color = "info"
            
            log_messages.append(
                html.Div([
                    html.Span(f"[{timestamp}] ", className="text-muted"),
                    html.Span(message, className=f"text-{color}")
                ], className="mb-1")
            )
        
        try:
            # Validate inputs
            if not icm_dirs:
                add_log("Error: ICM+ files directory path is required", "error")
                return html.Div(log_messages), None, ""
            
            if not db_path:
                add_log("Error: Database file path is required", "error")
                return html.Div(log_messages), None, ""
            
            # Ensure database file has correct extension
            if operation_mode == "create":
                if (not db_type or db_type == "duckdb") and not db_path.lower().endswith('.duckdb'):
                    db_path = f"{db_path}.duckdb"
                elif db_type == "sqlite" and not db_path.lower().endswith('.db'):
                    db_path = f"{db_path}.db"
            
            add_log(f"Starting {operation_mode} operation...")
            add_log(f"ICM+ directories: {icm_dirs}")
            add_log(f"Database path: {db_path}")
            
            # Scan the ICM+ folders
            add_log("Scanning ICM+ folders...")
            df = scan_icm_folders(
                icm_dirs,
                immediate_only=immediate_only,
                case_insensitive=case_insensitive,
                log_callback=add_log
            )
            
            if len(df) == 0:
                add_log("No data found in the specified directories", "warning")
                return html.Div(log_messages), None, ""
            
            # Save to CSV if requested
            if save_csv and csv_path:
                try:
                    df.to_csv(csv_path, index=False)
                    add_log(f"Successfully saved data to CSV: {csv_path}", "success")
                except Exception as e:
                    add_log(f"Error saving CSV: {str(e)}", "error")
            
            # Process database operation
            if operation_mode == "create":
                success = create_database(
                    df, 
                    db_path, 
                    db_type=db_type,
                    log_callback=add_log
                )
            else:  # update
                success = update_database(
                    df, 
                    db_path,
                    log_callback=add_log
                )
            
            if success:
                add_log(f"Database operation completed successfully!", "success")
            
            return html.Div(log_messages), df.to_json(date_format='iso', orient='split'), ""
        
        except Exception as e:
            add_log(f"Error during operation: {str(e)}", "error")
            import traceback
            add_log(traceback.format_exc(), "error")
            return html.Div(log_messages), None, ""

    return app

def main():
    """Main entry point for the ICM Database Creator application"""
    print("Starting ICM+ Database Creator...")
    print("This will open a web interface in your browser.")
    print("Access the application at: http://localhost:8050")
    
    app = create_app()
    port = int(os.environ.get("PORT", 8050))
    debug = bool(os.environ.get("DEBUG", True))
    
    try:
        app.run(debug=debug, port=port, host='0.0.0.0')
    except KeyboardInterrupt:
        print("\nApplication stopped by user.")
    except Exception as e:
        print(f"Error starting application: {e}")

if __name__ == "__main__":
    main()
