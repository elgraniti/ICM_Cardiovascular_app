from dash import Dash, dcc, html, Input, Output, State, dash_table, callback_context, MATCH, ALL, no_update, exceptions
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import sqlite3
import os
from datetime import datetime
from io import StringIO

# Add DuckDB import
import duckdb

# === Simple, standalone, single-page Dash app ===
# No auth, no roles, no sessions, no logging — just load data, filter, visualize, merge, and download.

# === Config ===
DEFAULT_DB_PATH = os.path.expanduser('icm_files_data.db')  # default/fallback
DB_STATE = {
    'path': DEFAULT_DB_PATH,
    'type': 'sqlite'  # 'sqlite' or 'duckdb'
}  # mutable holder to avoid globals

# === In-memory temp tables ===
TEMP_TABLES = {}            # name -> DataFrame
CURRENT_FILTERED_DF = None  # latest filtered view for saving

# === Helpers ===
def get_db_connection():
    """Get a database connection based on DB_STATE configuration"""
    if DB_STATE['type'] == 'duckdb':
        return duckdb.connect(DB_STATE['path'])
    else:  # default to sqlite
        return sqlite3.connect(DB_STATE['path'])

def detect_db_type(db_path):
    """Try to detect database type from file or extension"""
    if not os.path.exists(db_path):
        return 'sqlite'  # default
        
    # Check file extension
    if db_path.lower().endswith('.duckdb'):
        return 'duckdb'
    
    # Try opening with DuckDB first - it will fail if not a DuckDB database
    try:
        conn = duckdb.connect(db_path)
        # Try a simple query
        conn.execute("SELECT 1")
        conn.close()
        return 'duckdb'
    except:
        # If DuckDB fails, assume it's SQLite
        try:
            conn = sqlite3.connect(db_path)
            conn.execute("SELECT 1")
            conn.close()
            return 'sqlite'
        except:
            # If both fail, default to SQLite
            return 'sqlite'

def get_table_names():
    """List DB tables from DB_STATE['path'] + in-memory temp tables."""
    db_tables = []
    
    try:
        if DB_STATE['type'] == 'duckdb':
            conn = duckdb.connect(DB_STATE['path'])
            # DuckDB uses information_schema
            result = conn.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
            ).fetchall()
            db_tables = [row[0] for row in result]
            conn.close()
        else:
            # SQLite approach
            conn = sqlite3.connect(DB_STATE['path'])
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            db_tables = [row[0] for row in cursor.fetchall()]
            conn.close()
    except Exception as e:
        print(f"Error getting table names: {e}")
        db_tables = []
    
    return db_tables + list(TEMP_TABLES.keys())


def load_data(table_name, limit=None):
    """Read a DB or temp table into a DataFrame using DB_STATE['path']."""
    if table_name in TEMP_TABLES:
        df = TEMP_TABLES[table_name]
        return df.head(limit) if limit else df
    
    try:
        if DB_STATE['type'] == 'duckdb':
            conn = duckdb.connect(DB_STATE['path'])
            if limit:
                query = f"SELECT * FROM \"{table_name}\" LIMIT {int(limit)}"
            else:
                query = f"SELECT * FROM \"{table_name}\""
                
            # DuckDB can return pandas DataFrames directly
            df = conn.execute(query).df()
            conn.close()
        else:
            # SQLite approach
            conn = sqlite3.connect(DB_STATE['path'])
            if limit:
                df = pd.read_sql(f"SELECT * FROM [{table_name}] LIMIT {int(limit)}", conn)
            else:
                df = pd.read_sql(f"SELECT * FROM [{table_name}]", conn)
            conn.close()
            
        return df
    except Exception as e:
        print(f"Error loading table {table_name}: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error


def detect_column_types(df):
    """Heuristic typing for dynamic filters."""
    column_types = {}
    for col in df.columns:
        # Special handling for modalities and measures columns (lists)
        if col.lower() == 'modalities' or col.lower() == 'measures':
            column_types[col] = 'list_contains'
            continue
            
        # Special handling for measure_* and modalities_* columns (binary indicators)
        if col.lower().startswith('measure_') or col.lower().startswith('modalities_'):
            column_types[col] = 'binary_indicator'
            continue
            
        # skip large-ID style cols
        if 'ID' in col and df[col].nunique() > 100:
            continue
        # date-ish
        if any(key in col.upper() for key in ['DATE', 'DOB', 'BIRTH', 'DEATH']):
            column_types[col] = 'date'
            continue
        # categorical (few uniques)
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) <= 50:
            column_types[col] = 'categorical'
            continue
        # numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            column_types[col] = 'numeric'
            continue
        # default text
        column_types[col] = 'text'
    return column_types


def merge_tables(tables, join_columns, join_types, limit=None):
    """Progressively merge N tables by N-1 keys/types."""
    if not tables or len(tables) < 2:
        return None
    result_df = load_data(tables[0], limit=limit).copy()
    for i in range(1, len(tables)):
        next_df = load_data(tables[i])
        join_col = join_columns[i-1]
        join_type = join_types[i-1]
        if join_col in result_df.columns and join_col in next_df.columns:
            result_df = pd.merge(result_df, next_df, on=join_col, how=join_type)
        else:
            return None
        if limit and len(result_df) > limit:
            result_df = result_df.head(limit)
    return result_df


def get_common_columns(tables):
    """Find columns appearing in multiple tables to suggest join keys."""
    common = {}
    for t in tables or []:
        df = load_data(t, limit=1)  # just need the columns
        for c in df.columns:
            common.setdefault(c, []).append(t)
    return {c: ts for c, ts in common.items() if len(ts) > 1}


# === Dash app ===
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME], suppress_callback_exceptions=True)
app.title = "Data Dashboard (SQLite & DuckDB)"

# --- Layout ---
app.layout = html.Div([
    dbc.Navbar(
        dbc.Container([
            html.Div([
                html.I(className="fas fa-database me-2"),
                html.Span("Data Dashboard", className="navbar-brand mb-0 h1")
            ]),
        ]),
        color="dark",
        dark=True,
        className="mb-3",
    ),

    dbc.Row([
        # Sidebar
        dbc.Col([
            html.H5("Database Source", className="mb-2"),
            dbc.Input(id='db-path-input', placeholder='Absolute path to .db or .duckdb file', value=DEFAULT_DB_PATH, className='mb-2'),
            html.Div([
                dbc.RadioItems(
                    options=[
                        {"label": "Auto-detect", "value": "auto"},
                        {"label": "SQLite", "value": "sqlite"},
                        {"label": "DuckDB", "value": "duckdb"}
                    ],
                    value="auto",
                    id="db-type-select",
                    inline=True,
                    className="mb-2"
                )
            ]),
            dbc.Button("Connect", id="connect-db", color="primary", size="sm", className="w-100 mb-2"),
            dcc.Upload(id='db-upload', children=html.Div([html.I(className="fas fa-upload me-2"), "…or drop a .db/.duckdb file here"]), multiple=False, className='mb-2', style={'width':'100%','border':'1px dashed #aaa','borderRadius':'8px','padding':'8px','textAlign':'center'}),
            html.Div(id='db-status', className='small text-muted mb-3'),
            html.H5("Database Tables", className="mb-3"),
            dbc.Button("Merge Tables", id="open-merge-modal", color="info", outline=True, className="w-100 mb-3", size="sm"),
            dbc.ListGroup(id='table-list', children=[], className="mb-4"),
            dbc.Button("Show/Hide Filters", id="toggle-filters", color="secondary", className="mb-3 w-100", outline=True, size="sm"),
            dbc.Collapse(html.Div([
                html.Div(id="dynamic-filters", className="mt-2"),
                html.Div([
                    dbc.Button("Clear All Filters", id="clear-filters", color="primary", outline=True, size="sm")
                ], className="d-flex justify-content-end mt-2")
            ]), id="filter-collapse", is_open=False),
        ], width=3, className="border-end p-3", style={"height": "100vh", "overflowY": "auto", "backgroundColor": "#f8f9fa"}),

        # Main area
        dbc.Col([
            html.H2("Data Viewer", className="mb-3"),
            html.Div(id='selected-table-header'),
            html.Div(id='filter-summary', className="mb-2"),
            html.Div([html.B("Total Records: "), html.Span(id="record-count")], className="mb-3"),
            html.Div(id='table-container', className="overflow-auto", style={"maxHeight": "420px"}),

            html.Hr(),
            dbc.Button("Show Visualization", id="toggle-viz-btn", color="secondary", outline=True, className="mb-3"),
            dbc.Collapse([
                html.Div([
                    html.Label("Select columns to visualize:"),
                    dcc.Dropdown(id="viz-column-selector", multi=True, placeholder="Choose columns"),
                    html.Small("Pick categorical / low-cardinality columns.", className="text-muted")
                ], className="mb-3"),
                dcc.Graph(id='data-visualization')
            ], id="viz-collapse", is_open=False),
        ], width=9, className="p-3")
    ], className="g-0"),

    # Merge modal
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Merge Tables")),
        dbc.ModalBody([
            dbc.Alert([
                html.H6("How it works", className="alert-heading"),
                html.Ol([
                    html.Li("Select 2+ tables in order"),
                    html.Li("Choose join columns (N-1)"),
                    html.Li("Pick join types (N-1): inner, left, right, outer"),
                ])
            ], color="info", className="mb-3"),
            dbc.Row([
                dbc.Col([
                    html.Label("Tables to Merge (in order)"),
                    dcc.Dropdown(id="merge-tables", options=[{"label": t, "value": t} for t in get_table_names()], multi=True)
                ], width=12)
            ], className="mb-3"),
            dbc.Row([
                dbc.Col([
                    html.Label("Save as Temp Table (optional)"),
                    dbc.InputGroup([
                        dbc.InputGroupText("Temp_"),
                        dbc.Input(id="temp-table-name", placeholder="custom_name", type="text"),
                    ]),
                ], width=12)
            ], className="mb-3"),
            dbc.Row([
                dbc.Col([
                    html.Label("Join Columns (N-1)"),
                    dcc.Dropdown(id="merge-columns", multi=True)
                ], width=6),
                dbc.Col([
                    html.Label("Join Types (N-1)"),
                    dcc.Dropdown(id="merge-types", options=[
                        {"label": "Inner", "value": "inner"},
                        {"label": "Left", "value": "left"},
                        {"label": "Right", "value": "right"},
                        {"label": "Outer", "value": "outer"},
                    ], multi=True)
                ], width=6)
            ], className="mb-3"),
            html.Div(id="merge-validation", className="mb-2"),
            html.Div(id="merge-preview", className="mt-2")
        ]),
        dbc.ModalFooter([
            dbc.Button("Close", id="close-merge-modal", className="me-2", color="secondary"),
            dbc.Button("Merge Tables", id="execute-merge", color="primary")
        ])
    ], id="merge-modal", size="lg", is_open=False),

    # Save filtered data modal
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Save Filtered Data as New Table")),
        dbc.ModalBody([
            html.P("Save the current filtered view as a table for later use (merge/analysis)."),
            dbc.Form([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Table Type"),
                        dbc.RadioItems(options=[{"label": "Temporary", "value": "temp"}, {"label": "Permanent (write to DB)", "value": "perm"}], value="temp", id="filtered-table-type", inline=True)
                    ], width=12)
                ], className="mb-3"),
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Table Name"),
                        dbc.InputGroup([
                            dbc.InputGroupText(id="table-prefix", children="Temp_"),
                            dbc.Input(id="filtered-table-name", placeholder="filtered_name", type="text")
                        ]),
                        html.Small("Enter a short name (no spaces).", className="text-muted")
                    ], width=12)
                ])
            ])
        ]),
        dbc.ModalFooter([
            dbc.Button("Cancel", id="cancel-save-filtered", className="me-2", color="secondary"),
            dbc.Button("Save Filtered Data", id="confirm-save-filtered", color="success")
        ])
    ], id="save-filtered-modal", size="md", is_open=False),

    # Stores
    dcc.Store(id='selected-table'),
    dcc.Store(id='table-data'),
    dcc.Store(id='merged-data'),
    dcc.Store(id='db-path', data=DEFAULT_DB_PATH),
    dcc.Store(id='db-type', data='sqlite'),
    dcc.Store(id='tables-version', data=0),
])


# === Callbacks ===
# Sidebar: select table
@app.callback(
    Output('selected-table', 'data'),
    Output('selected-table-header', 'children'),
    Input({'type': 'table-item', 'index': ALL}, 'n_clicks'),
    State({'type': 'table-item', 'index': ALL}, 'id'),
    prevent_initial_call=True
)
def select_table(clicks, ids):
    ctx = callback_context
    if not ctx.triggered:
        return None, html.H4("Select a table from the sidebar")
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    try:
        btn = eval(button_id)  # id dict
        table_name = btn['index']
        return table_name, html.H4(f"Table: {table_name}")
    except Exception:
        return None, html.H4("Error loading table")


# Toggle sections
@app.callback(Output("filter-collapse", "is_open"), Input("toggle-filters", "n_clicks"), State("filter-collapse", "is_open"))
def toggle_filters(n, is_open):
    return (not is_open) if n else is_open


@app.callback(Output("viz-collapse", "is_open"), Input("toggle-viz-btn", "n_clicks"), State("viz-collapse", "is_open"))
def toggle_viz(n, is_open):
    return (not is_open) if n else is_open


# Build dynamic filters + load data
@app.callback(Output('dynamic-filters', 'children'), Output('table-data', 'data'), Input('selected-table', 'data'))
def create_filters(table_name):
    if not table_name:
        return [], None
    df = load_data(table_name)
    column_types = detect_column_types(df)
    filters = []
    for col, col_type in column_types.items():
        # Special filter for modalities and measures columns (lists that need contains search)
        if col_type == 'list_contains':
            filters.append(html.Div([
                html.Label(f"Filter {col} (contains)"),
                html.Small("Enter value to search within the list", className="text-muted d-block mb-2"),
                dbc.Input(id={'type': 'filter-list-contains', 'column': col}, 
                         type="text", 
                         placeholder="Enter value to search for...")
            ], className="mb-3"))
        
        # Special filter for measure_* and modalities_* binary indicators (0, 1, NA)
        elif col_type == 'binary_indicator':
            filters.append(html.Div([
                html.Label(f"Filter {col}"),
                dcc.Dropdown(
                    id={'type': 'filter-binary', 'column': col}, 
                    options=[
                        {'label': 'Has value (1)', 'value': '1'},
                        {'label': 'No value (0)', 'value': '0'},
                        {'label': 'Unknown (NA)', 'value': 'na'}
                    ],
                    multi=True,
                    placeholder="Select values..."
                )
            ], className="mb-3"))
        
        elif col_type == 'categorical':
            unique_vals = sorted(df[col].dropna().astype(str).unique().tolist())
            filters.append(html.Div([
                html.Label(f"Filter by {col}"),
                dcc.Dropdown(id={'type': 'filter-dropdown', 'column': col}, options=[{'label': v, 'value': v} for v in unique_vals], multi=True)
            ], className="mb-3"))
        elif col_type == 'date':
            date_series = pd.to_datetime(df[col], errors='coerce')
            min_date = (date_series.min().date() if not pd.isna(date_series.min()) else datetime(1900,1,1).date())
            max_date = (date_series.max().date() if not pd.isna(date_series.max()) else datetime(2100,1,1).date())
            filters.append(html.Div([
                html.Label(f"Filter by {col}"),
                dcc.DatePickerRange(id={'type': 'filter-daterange', 'column': col}, min_date_allowed=min_date, max_date_allowed=max_date)
            ], className="mb-3"))
        elif col_type == 'numeric':
            min_val = df[col].min(); max_val = df[col].max()
            try:
                min_val = float(min_val); max_val = float(max_val)
            except Exception:
                continue
            step = (max_val - min_val) / 100 if max_val > min_val else 1
            filters.append(html.Div([
                html.Label(f"Filter by {col}"),
                dcc.RangeSlider(id={'type': 'filter-range', 'column': col}, min=min_val, max=max_val, step=step, marks={min_val: str(min_val), max_val: str(max_val)}, value=[min_val, max_val])
            ], className="mb-3"))
        else:
            filters.append(html.Div([
                html.Label(f"Search {col}"),
                dbc.Input(id={'type': 'filter-text', 'column': col}, type="text", placeholder=f"contains...")
            ], className="mb-3"))
    return filters, df.to_json(date_format='iso', orient='split')


# Update table + viz + summary
@app.callback(
    Output('table-container', 'children'),
    Output('data-visualization', 'figure'),
    Output('record-count', 'children'),
    Output('filter-summary', 'children'),
    Input('table-data', 'data'),
    Input({'type': 'filter-dropdown', 'column': ALL}, 'value'),
    Input({'type': 'filter-daterange', 'column': ALL}, 'start_date'),
    Input({'type': 'filter-daterange', 'column': ALL}, 'end_date'),
    Input({'type': 'filter-range', 'column': ALL}, 'value'),
    Input({'type': 'filter-text', 'column': ALL}, 'value'),
    # New filter inputs for measures and modalities
    Input({'type': 'filter-list-contains', 'column': ALL}, 'value'),
    Input({'type': 'filter-binary', 'column': ALL}, 'value'),
    Input('viz-column-selector', 'value'),
    State({'type': 'filter-dropdown', 'column': ALL}, 'id'),
    State({'type': 'filter-daterange', 'column': ALL}, 'id'),
    State({'type': 'filter-range', 'column': ALL}, 'id'),
    State({'type': 'filter-text', 'column': ALL}, 'id'),
    # New filter states for measures and modalities
    State({'type': 'filter-list-contains', 'column': ALL}, 'id'),
    State({'type': 'filter-binary', 'column': ALL}, 'id'),
    State('selected-table', 'data'),
    Input('clear-filters', 'n_clicks'),
    State('filter-collapse', 'is_open')
)

def update_view(table_json, dd_vals, d_start, d_end, range_vals, text_vals, list_contains_vals, binary_vals, viz_cols, 
               dd_ids, d_ids, r_ids, t_ids, list_contains_ids, binary_ids, current_table, clear_clicks, filters_visible):
    if table_json is None:
        return html.Div("No data"), {}, "0", html.Div()
    df = pd.read_json(StringIO(table_json), orient='split')

    # Clear filters
    ctx = callback_context
    if ctx.triggered and 'clear-filters' in ctx.triggered[0]['prop_id']:
        fig = create_visualization(df, viz_cols)
        return create_table_from_dataframe(df), fig, str(len(df)), html.Div("Filters cleared")

    # If filters hidden, show unfiltered
    if not filters_visible:
        fig = create_visualization(df, viz_cols)
        return create_table_from_dataframe(df), fig, str(len(df)), html.Div("Filters hidden")

    filtered = df.copy()
    applied = []

    # categorical
    for vals, idd in zip(dd_vals, dd_ids):
        if vals:
            col = idd['column']
            filtered = filtered[filtered[col].astype(str).isin([str(v) for v in vals])]
            applied.append(f"{col}: {', '.join(map(str, vals))}")
    
    # dates
    for s,e, idd in zip(d_start, d_end, d_ids):
        col = idd['column']
        if s:
            filtered = filtered[pd.to_datetime(filtered[col], errors='coerce') >= pd.to_datetime(s)]
            applied.append(f"{col} ≥ {s}")
        if e:
            filtered = filtered[pd.to_datetime(filtered[col], errors='coerce') <= pd.to_datetime(e)]
            applied.append(f"{col} ≤ {e}")
    
    # numeric
    for rv, idd in zip(range_vals, r_ids):
        if rv:
            mn, mx = rv
            col = idd['column']
            filtered = filtered[(filtered[col] >= mn) & (filtered[col] <= mx)]
            applied.append(f"{mn} ≤ {col} ≤ {mx}")
    
    # text
    for txt, idd in zip(text_vals, t_ids):
        if txt:
            col = idd['column']
            filtered = filtered[filtered[col].astype(str).str.contains(txt, case=False, na=False)]
            applied.append(f"{col} contains '{txt}'")
    
    # list contains (for modalities and measures)
    for txt, idd in zip(list_contains_vals, list_contains_ids):
        if txt:
            col = idd['column']
            # Apply contains filter on list-like columns (case-insensitive)
            filtered = filtered[filtered[col].astype(str).str.contains(txt, case=False, na=False)]
            applied.append(f"{col} contains '{txt}'")
    
    # binary indicators (for measure_* and modalities_* columns)
    for vals, idd in zip(binary_vals, binary_ids):
        if vals:
            col = idd['column']
            # Create masks for each selected value
            masks = []
            selected_labels = []
            
            for val in vals:
                if val == '1':
                    # For value 1
                    masks.append(filtered[col] == 1)
                    selected_labels.append("Has value (1)")
                elif val == '0':
                    # For value 0
                    masks.append(filtered[col] == 0)
                    selected_labels.append("No value (0)")
                elif val == 'na':
                    # For NA values
                    masks.append(filtered[col].isna())
                    selected_labels.append("Unknown (NA)")
            
            if masks:
                # Combine masks with OR
                combined_mask = pd.Series(False, index=filtered.index)
                for mask in masks:
                    combined_mask = combined_mask | mask
                
                filtered = filtered[combined_mask]
                applied.append(f"{col}: {', '.join(selected_labels)}")

    global CURRENT_FILTERED_DF
    CURRENT_FILTERED_DF = filtered

    # filter summary + save button
    if applied:
        summary = html.Div([
            html.H6("Applied Filters:"),
            html.Ul([html.Li(x) for x in applied]),
            dbc.Button("Save Filtered Data", id="save-filtered-btn", color="success", size="sm", className="mt-2")
        ])
    else:
        summary = html.Div("No filters applied")

    fig = create_visualization(filtered, viz_cols)
    return create_table_from_dataframe(filtered), fig, str(len(filtered)), summary


# Build DataTable + download controls
from dash import dcc as _dcc

def create_table_from_dataframe(df, table_name=None):
    display_name = (table_name or "current-table")
    if display_name.startswith("Temp_"):
        display_name = display_name[5:]
    return html.Div([
        dbc.Row([
            dbc.Col(dbc.Input(id={'type': 'download-filename', 'index': table_name or "current-table"}, type="text", placeholder="filename (no .csv)", value=display_name), width=8),
            dbc.Col(dbc.Button([html.I(className="fas fa-download me-2"), "Download CSV"], id={'type': 'download-button', 'index': table_name or "current-table"}, color="success", className="w-100"), width=4)
        ], className="mb-2"),
        _dcc.Download(id={'type': 'download-data', 'index': table_name or "current-table"}),
        dash_table.DataTable(
            id={'type': 'data-table', 'index': table_name or "current-table"},
            data=df.to_dict('records'),
            columns=[{'name': c, 'id': c} for c in df.columns],
            page_size=15,
            style_table={'overflowX': 'auto'},
            style_cell={'height': 'auto', 'minWidth': '100px', 'maxWidth': '220px', 'whiteSpace': 'normal', 'textAlign': 'left'},
            style_header={'backgroundColor': 'rgb(230,230,230)', 'fontWeight': 'bold'},
            filter_action="native",
            sort_action="native",
            page_action='native'
        )
    ])


# Download callback (pattern-matching for each table view)
@app.callback(
    Output({'type': 'download-data', 'index': MATCH}, 'data'),
    Input({'type': 'download-button', 'index': MATCH}, 'n_clicks'),
    State('table-data', 'data'),
    State({'type': 'download-filename', 'index': MATCH}, 'value'),
    prevent_initial_call=True
)

def download_current(n, table_json, filename):
    if not n:
        raise exceptions.PreventUpdate
    df = pd.read_json(StringIO(table_json), orient='split')
    name = (filename or 'data').strip().replace(' ', '_') + '.csv'
    return _dcc.send_data_frame(df.to_csv, name, index=False)


# Visualization helper
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px

def create_visualization(df, selected_cols=None):
    if df is None or df.empty:
        return {'data': [], 'layout': {'title': 'No Data'}}
    # honor selected cols if valid
    categorical_cols = [c for c in (selected_cols or []) if c in df.columns]
    if not categorical_cols:
        # auto-pick reasonable columns
        for col in df.columns:
            if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
                if df[col].nunique() <= 30 and df[col].notna().sum() > len(df)*0.5:
                    categorical_cols.append(col)
            elif pd.api.types.is_numeric_dtype(df[col]) and 1 < df[col].nunique() <= 15:
                categorical_cols.append(col)
    if not categorical_cols:
        return {'data': [], 'layout': {'title': 'No categorical columns to visualize'}}

    rows = len(categorical_cols)
    vertical_spacing = 0.1 if rows == 1 else min(0.1, 0.9 / (rows - 1))
    fig = make_subplots(rows=rows, cols=1, subplot_titles=[f'Distribution by {c}' for c in categorical_cols], vertical_spacing=vertical_spacing)
    colors = px.colors.qualitative.Plotly * 3
    for i, col in enumerate(categorical_cols, start=1):
        vc = df[col].astype(str).value_counts().head(10)
        fig.add_trace(go.Bar(x=vc.index.tolist(), y=vc.values.tolist(), marker_color=[colors[j % len(colors)] for j in range(len(vc))], showlegend=False, text=vc.values.tolist(), textposition='outside', texttemplate='%{text}', textfont=dict(size=12)), row=i, col=1)
        fig.update_xaxes(title=col, tickangle=45, row=i, col=1)
        fig.update_yaxes(title='Count', row=i, col=1)
    fig.update_layout(title='Categorical Distributions', height=600*rows, showlegend=False, margin={'l':40,'r':40,'t':60,'b':80}, uniformtext_minsize=10, uniformtext_mode='hide')
    return fig


# Clear all filters (reset UI state)
@app.callback(
    Output({'type': 'filter-dropdown', 'column': ALL}, 'value'),
    Output({'type': 'filter-daterange', 'column': ALL}, 'start_date'),
    Output({'type': 'filter-daterange', 'column': ALL}, 'end_date'),
    Output({'type': 'filter-range', 'column': ALL}, 'value'),
    Output({'type': 'filter-text', 'column': ALL}, 'value'),
    # New filter outputs for clearing
    Output({'type': 'filter-list-contains', 'column': ALL}, 'value'),
    Output({'type': 'filter-binary', 'column': ALL}, 'value'),
    Output('viz-column-selector', 'value', allow_duplicate=True),
    Input('clear-filters', 'n_clicks'),
    State('table-data', 'data'),
    State({'type': 'filter-dropdown', 'column': ALL}, 'id'),
    State({'type': 'filter-daterange', 'column': ALL}, 'id'),
    State({'type': 'filter-range', 'column': ALL}, 'id'),
    State({'type': 'filter-text', 'column': ALL}, 'id'),
    # New filter states
    State({'type': 'filter-list-contains', 'column': ALL}, 'id'),
    State({'type': 'filter-binary', 'column': ALL}, 'id'),
    State('viz-column-selector', 'options'),
    prevent_initial_call=True
)

def clear_filters(n, table_json, dd_ids, d_ids, r_ids, t_ids, list_contains_ids, binary_ids, viz_opts):
    if not n or not table_json:
        raise exceptions.PreventUpdate
    df = pd.read_json(StringIO(table_json), orient='split')
    # reset ranges to full span
    range_values = []
    for rid in r_ids:
        col = rid['column']
        try:
            mn = float(df[col].min()); mx = float(df[col].max())
            range_values.append([mn, mx])
        except Exception:
            range_values.append(None)
    viz_values = [opt['value'] for opt in (viz_opts or [])][:2]
    # Return empty values for all filter types, including new ones
    return ([None]*len(dd_ids), 
            [None]*len(d_ids), 
            [None]*len(d_ids), 
            range_values, 
            [""]*len(t_ids), 
            [""]*len(list_contains_ids), 
            [None]*len(binary_ids), 
            viz_values)


# Keep viz column options in sync with table
@app.callback(Output('viz-column-selector', 'options'), Input('table-data', 'data'))

def update_viz_options(table_json):
    if not table_json:
        return []
    df = pd.read_json(StringIO(table_json), orient='split')
    opts = []
    for col in df.columns:
        if (pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col])) and df[col].nunique() <= 30:
            opts.append({'label': col, 'value': col})
        elif pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() <= 15:
            opts.append({'label': col, 'value': col})
    return opts


@app.callback(Output('viz-column-selector', 'value'), Input('selected-table', 'data'), Input('viz-column-selector', 'options'))

def set_viz_defaults(_sel, options):
    return [opt['value'] for opt in (options or [])][:2]


# Merge modal show/hide
@app.callback(Output("merge-modal", "is_open"), [Input("open-merge-modal", "n_clicks"), Input("close-merge-modal", "n_clicks")], State("merge-modal", "is_open"))

def toggle_merge_modal(n1, n2, open_):
    if n1 or n2:
        return not open_
    return open_


# Validate merge selections
@app.callback(Output("merge-validation", "children"), Input("merge-tables", "value"), Input("merge-columns", "value"), Input("merge-types", "value"))

def validate_merge(tables, cols, types):
    if not tables or len(tables) < 2:
        return html.Div("Select at least two tables", className="text-danger")
    need = len(tables) - 1
    if not cols or len(cols) < need:
        return html.Div(f"Pick {need} join column(s)", className="text-danger")
    if not types or len(types) < need:
        return html.Div(f"Pick {need} join type(s)", className="text-danger")
    if len(cols) > need or len(types) > need:
        return html.Div("You selected too many join options; extras will be ignored.", className="text-warning")
    return html.Div(f"Ready to merge {len(tables)} tables with {need} joins", className="text-success")


# Suggest join columns based on overlap
@app.callback(Output('merge-columns', 'options'), Input('merge-tables', 'value'))

def suggest_join_cols(tables):
    if not tables or len(tables) < 2:
        return []
    common = get_common_columns(tables)
    return [{'label': c, 'value': c} for c in common.keys()]


# Execute merge, preview, and add temp table
@app.callback(Output("merge-preview", "children"), Output("tables-version", "data", allow_duplicate=True), Input("execute-merge", "n_clicks"), State("merge-tables", "value"), State("merge-columns", "value"), State("merge-types", "value"), State("temp-table-name", "value"), State('tables-version','data'), prevent_initial_call=True)

def do_merge(n, tables, cols, types, temp_name, version):
    if not n:
        raise exceptions.PreventUpdate
    if not tables or len(tables) < 2:
        return html.Div("Select 2+ tables", className="text-danger"), no_update
    need = len(tables) - 1
    cols = (cols or [])[:need]
    types = (types or [])[:need]
    df = merge_tables(tables, cols, types)
    if df is None or df.empty:
        return html.Div("Merge produced no rows or failed.", className="text-danger"), no_update
    # Save temp table if requested
    if temp_name:
        name = f"Temp_{temp_name}"
        TEMP_TABLES[name] = df.copy()
    preview = create_table_from_dataframe(df, table_name=(temp_name and f"Temp_{temp_name}"))
    new_version = (0 if version is None else version) + 1
    return preview, new_version


# Open/close Save Filtered modal
@app.callback(Output("save-filtered-modal", "is_open"), [Input("save-filtered-btn", "n_clicks"), Input("cancel-save-filtered", "n_clicks"), Input("confirm-save-filtered", "n_clicks")], State("save-filtered-modal", "is_open"), prevent_initial_call=True)

def toggle_save_modal(n1, n2, n3, open_):
    ctx = callback_context
    if not ctx.triggered:
        return open_
    trig = ctx.triggered[0]['prop_id'].split('.')[0]
    if trig == 'save-filtered-btn' and n1:
        return True
    if trig in ('cancel-save-filtered', 'confirm-save-filtered') and open_:
        return False
    return open_


# Save filtered data as temp/permanent table
@app.callback(Output("filter-summary", "children", allow_duplicate=True), Output("tables-version", "data", allow_duplicate=True), Input("confirm-save-filtered", "n_clicks"), State("filtered-table-name", "value"), State("filtered-table-type", "value"), State('tables-version','data'), prevent_initial_call=True)

def save_filtered(n, name, table_type, version):
    if not n or not name:
        raise exceptions.PreventUpdate
    if CURRENT_FILTERED_DF is None or CURRENT_FILTERED_DF.empty:
        return html.Div("No filtered data to save.", className="text-danger"), no_update
    
    prefix = "Perm_" if table_type == "perm" else "Temp_"
    full_name = f"{prefix}{name}"
    
    # Save in memory
    TEMP_TABLES[full_name] = CURRENT_FILTERED_DF.copy()
    
    # Optionally persist to DB
    if table_type == 'perm':
        try:
            if DB_STATE['type'] == 'duckdb':
                conn = duckdb.connect(DB_STATE['path'])
                # Create a table from the DataFrame - DuckDB has built-in pandas integration
                conn.execute(f"DROP TABLE IF EXISTS \"{full_name}\"")
                # Register the dataframe and create a permanent table from it
                conn.register('temp_df', CURRENT_FILTERED_DF)
                conn.execute(f"CREATE TABLE \"{full_name}\" AS SELECT * FROM temp_df")
                conn.unregister('temp_df')
                conn.close()
            else:
                # SQLite approach
                conn = sqlite3.connect(DB_STATE['path'])
                CURRENT_FILTERED_DF.to_sql(full_name, conn, index=False, if_exists='replace')
                conn.close()
        except Exception as e:
            print(f"Error saving permanent table: {e}")
            # Keep temp version even if DB write fails
    
    msg = html.Div([
        html.Span("Saved filtered data as "), html.B(full_name), html.Span(f" ({len(CURRENT_FILTERED_DF)} rows).")
    ], className="text-success")
    
    new_version = (0 if version is None else version) + 1
    return msg, new_version


# Update table name prefix based on selection
@app.callback(Output("table-prefix", "children"), Input("filtered-table-type", "value"))

def set_prefix(t):
    return "Perm_" if t == 'perm' else "Temp_"


# === New: DB connect / upload callbacks with DuckDB support ===
@app.callback(
    Output('db-status', 'children'),
    Output('db-path', 'data'),
    Output('db-type', 'data'),
    Input('connect-db', 'n_clicks'),
    State('db-path-input', 'value'),
    State('db-type-select', 'value'),
    prevent_initial_call=True
)
def connect_db(n, path, db_type_selection):
    if not n:
        raise exceptions.PreventUpdate
        
    path = (path or '').strip()
    if not path:
        return html.Span("Please enter a database path."), no_update, no_update
    
    # Auto-detect or use specified type
    if db_type_selection == 'auto':
        db_type = detect_db_type(path)
    else:
        db_type = db_type_selection
    
    # Try to connect with the determined type
    try:
        if db_type == 'duckdb':
            conn = duckdb.connect(path)
            conn.execute("SELECT 1")
        else:
            conn = sqlite3.connect(path)
            conn.execute("SELECT 1")
        conn.close()
    except Exception as e:
        return html.Span(f"Failed to open DB: {e}"), no_update, no_update
    
    # Update database path and type in global state
    DB_STATE['path'] = path
    DB_STATE['type'] = db_type
    
    db_type_display = 'DuckDB' if db_type == 'duckdb' else 'SQLite'
    return html.Span(f"Connected: {path} ({db_type_display})"), path, db_type


import base64
@app.callback(
    Output('db-status', 'children', allow_duplicate=True),
    Output('db-path', 'data', allow_duplicate=True),
    Output('db-type', 'data', allow_duplicate=True),
    Input('db-upload', 'contents'),
    State('db-upload', 'filename'),
    State('db-type-select', 'value'),
    prevent_initial_call=True
)
def handle_upload(contents, filename, db_type_selection):
    if not contents:
        raise exceptions.PreventUpdate
    
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    os.makedirs('uploaded_databases', exist_ok=True)
    safe_name = (filename or 'uploaded.db').replace(' ', '_')
    save_path = os.path.join('uploaded_databases', safe_name)
    
    with open(save_path, 'wb') as f:
        f.write(decoded)
    
    # Auto-detect or use specified type
    if db_type_selection == 'auto':
        db_type = detect_db_type(save_path)
    else:
        db_type = db_type_selection
    
    # Try to connect with the determined type
    try:
        if db_type == 'duckdb':
            conn = duckdb.connect(save_path)
            conn.execute("SELECT 1")
        else:
            conn = sqlite3.connect(save_path)
            conn.execute("SELECT 1")
        conn.close()
    except Exception as e:
        return html.Span(f"Upload saved but DB open failed: {e}"), no_update, no_update
    
    # Update database path and type in global state
    DB_STATE['path'] = save_path
    DB_STATE['type'] = db_type
    
    db_type_display = 'DuckDB' if db_type == 'duckdb' else 'SQLite'
    return html.Span(f"Uploaded and connected: {save_path} ({db_type_display})"), save_path, db_type


# Populate/refresh table list whenever db-path or temp tables change
@app.callback(Output('table-list', 'children'), Input('db-path', 'data'), Input('db-type', 'data'), Input('tables-version', 'data'))

def populate_tables(path, db_type, version):
    items = [
        dbc.ListGroupItem(name, id={'type': 'table-item', 'index': name}, action=True, n_clicks=0)
        for name in get_table_names()
    ]
    return items


def main():
    """Main entry point for the ICM Database Viewer application"""
    print("Starting ICM+ Database Viewer...")
    print("This will open a web interface in your browser.")
    print("Access the application at: http://localhost:8055")
    
    port = int(os.environ.get("PORT", 8055))
    debug = bool(os.environ.get("DEBUG", True))
    
    try:
        app.run(debug=debug, port=port, host='0.0.0.0')
    except KeyboardInterrupt:
        print("\nApplication stopped by user.")
    except Exception as e:
        print(f"Error starting application: {e}")

if __name__ == "__main__":
    main()