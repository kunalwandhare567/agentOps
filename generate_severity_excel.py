"""
generate_severity_excel.py
===========================
Generates an Excel workbook showing severity for every telemetry row.

Output: outputs/Severity_Report.xlsx  with 4 sheets:
  1. Episode_Severity    — one row per episode step (episode, elapsed_s, failure_mode, Severity, ...)
  2. Episode_Summary     — one row per episode: peak severity, % time at each level
  3. Failure_Mode_Pivot  — crosstab: failure_mode vs Severity count
  4. Indicator_Detail    — per-feature NORMAL/WARNING/CRITICAL labels (sample: first 5000 rows)
"""

import os
import pandas as pd
import numpy as np
from severity_engine.severity_engine import SeverityEngine

os.makedirs('outputs', exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load and prepare telemetry (same pipeline as generate_hmm_states.py)
# ─────────────────────────────────────────────────────────────────────────────
print("Loading metrics, logs, and traces ...")
metrics_df = pd.read_csv('data/telemetry_metrics.csv')
logs_df    = pd.read_csv('data/telemetry_logs.csv')
traces_df  = pd.read_csv('data/telemetry_traces.csv', low_memory=False)

# Fix circuit_breaker_state
metrics_df['circuit_breaker_state'] = (
    metrics_df['circuit_breaker_state']
    .map({'closed': 0, 'half-open': 1, 'open': 2})
    .fillna(0).astype(int)
)

# ── Log features ──────────────────────────────────────────────────────────────
print("Merging logs ...")
merged = pd.merge(
    metrics_df,
    logs_df[['episode_id', 'elapsed_s', 'log_level', 'exception_type']],
    on=['episode_id', 'elapsed_s'],
    how='inner'
)
merged['log_max_severity'] = (
    merged['log_level']
    .map({'': 0, 'INFO': 1, 'WARNING': 2, 'WARN': 2, 'ERROR': 3, 'CRITICAL': 4, 'FATAL': 4})
    .fillna(0)
)
merged['log_has_exception'] = merged['exception_type'].apply(
    lambda x: 0 if pd.isna(x) or x == '' else 1
)
merged['log_critical_count'] = (merged['log_max_severity'] >= 4).astype(int)

# ── Trace features ────────────────────────────────────────────────────────────
print("Engineering trace features ...")
is_root  = traces_df['parent_span_id'].isna() | (traces_df['parent_span_id'] == '')
is_child = ~is_root

root_spans   = traces_df[is_root]
root_counts  = root_spans.groupby(['episode_id', 'elapsed_s']).size()
root_errors  = (
    root_spans[root_spans['span_status'] == 'ERROR']
    .groupby(['episode_id', 'elapsed_s']).size()
)
root_span_error_rate = (root_errors / root_counts).fillna(0)

total_spans            = traces_df.groupby(['episode_id', 'elapsed_s']).size()
unique_traces          = traces_df.groupby(['episode_id', 'elapsed_s'])['trace_id'].nunique()
duplicate_trace_id_count = (total_spans - unique_traces).clip(lower=0)

error_spans             = traces_df[traces_df['span_status'] == 'ERROR']
distinct_error_services = error_spans.groupby(['episode_id', 'elapsed_s'])['service'].nunique()

root_durations  = traces_df[is_root].groupby(['episode_id', 'elapsed_s', 'trace_id'])['span_duration_ms'].max()
child_durations = traces_df[is_child].groupby(['episode_id', 'elapsed_s', 'trace_id'])['span_duration_ms'].sum()
tm = pd.DataFrame({'root_dur': root_durations, 'child_sum': child_durations}).fillna(0)
tm['gap_ratio'] = ((tm['root_dur'] - tm['child_sum']) / tm['root_dur']).clip(0, 1).fillna(0)
span_gap_ratio = tm.groupby(['episode_id', 'elapsed_s'])['gap_ratio'].mean()

child_min   = traces_df[is_child].groupby(['episode_id', 'elapsed_s', 'trace_id'])['span_duration_ms'].min()
child_max   = traces_df[is_child].groupby(['episode_id', 'elapsed_s', 'trace_id'])['span_duration_ms'].max()
child_count = traces_df[is_child].groupby(['episode_id', 'elapsed_s', 'trace_id']).size()
ts = pd.DataFrame({'min_dur': child_min, 'max_dur': child_max, 'count': child_count}).fillna(0)
ts['slowdown'] = 0.0
ts.loc[ts['count'] == 1, 'slowdown'] = 1.0
mask_multi = ts['count'] > 1
valid_max   = ts['max_dur'] > 0
ts.loc[mask_multi & valid_max,  'slowdown'] = ts['min_dur'] / ts['max_dur']
ts.loc[mask_multi & ~valid_max, 'slowdown'] = 1.0
uniform_slowdown_score = ts.groupby(['episode_id', 'elapsed_s'])['slowdown'].mean()

trace_features_df = pd.DataFrame({
    'root_span_error_rate':     root_span_error_rate,
    'span_gap_ratio':           span_gap_ratio,
    'uniform_slowdown_score':   uniform_slowdown_score,
    'duplicate_trace_id_count': duplicate_trace_id_count,
    'distinct_error_services':  distinct_error_services,
}).fillna(0).reset_index()

print("Merging trace features ...")
df = pd.merge(merged, trace_features_df, on=['episode_id', 'elapsed_s'], how='left').fillna(0)
print(f"  Full dataset shape: {df.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Run Severity Engine
# ─────────────────────────────────────────────────────────────────────────────
print("Running Severity Engine ...")
engine  = SeverityEngine()
results = engine.compute_severity(df)
print(f"  Done. Rows: {len(results)}")
print(results['Severity'].value_counts().sort_index())

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Build DataFrames for each sheet
# ─────────────────────────────────────────────────────────────────────────────
SEV_ORDER = {'P1': 1, 'P2': 2, 'P3': 3, 'P4': 4}

# ── Sheet 1: Episode_Severity ─────────────────────────────────────────────────
core_cols = [
    'episode_id', 'failure_mode', 'elapsed_s',
    'Severity', 'RawSeverity', 'WeightedScore',
    'CriticalCount', 'WarningCount', 'BlastSize',
    'HighRiskMode', 'BlastRadiusGrowing',
    'Reason', 'RecommendedAction',
    # Key raw metrics
    'cpu_utilization', 'memory_utilization', 'heap_mb',
    'p99_latency', 'error_rate', 'queue_lag',
    'cache_hit_rate', 'cache_miss_rate',
    'db_p99', 'disk_write_latency',
    'upstream_timeout_rate', 'retry_count_per_request',
    'root_span_error_rate', 'distinct_error_services',
]
available = [c for c in core_cols if c in results.columns]
sheet1 = results[available].copy().sort_values(['episode_id', 'elapsed_s']).reset_index(drop=True)

# ── Sheet 2: Episode_Summary ──────────────────────────────────────────────────
print("Building Episode Summary ...")
def peak_severity(series):
    best_num = series.map(SEV_ORDER).min()
    return {v: k for k, v in SEV_ORDER.items()}.get(best_num, 'P4')

summary_rows = []
for ep_id, grp in results.groupby('episode_id'):
    sev_counts = grp['Severity'].value_counts()
    total = len(grp)
    summary_rows.append({
        'Episode_ID':        ep_id,
        'Failure_Mode':      grp['failure_mode'].iloc[0],
        'Total_Steps':       total,
        'Peak_Severity':     peak_severity(grp['Severity']),
        'Steps_P1':          sev_counts.get('P1', 0),
        'Steps_P2':          sev_counts.get('P2', 0),
        'Steps_P3':          sev_counts.get('P3', 0),
        'Steps_P4':          sev_counts.get('P4', 0),
        'Pct_P1':            round(sev_counts.get('P1', 0) / total * 100, 1),
        'Pct_P2':            round(sev_counts.get('P2', 0) / total * 100, 1),
        'Pct_P3':            round(sev_counts.get('P3', 0) / total * 100, 1),
        'Pct_P4':            round(sev_counts.get('P4', 0) / total * 100, 1),
        'Max_CriticalCount': int(grp['CriticalCount'].max()),
        'Max_WeightedScore': round(float(grp['WeightedScore'].max()), 2),
        'Avg_CriticalCount': round(float(grp['CriticalCount'].mean()), 2),
    })
sheet2 = pd.DataFrame(summary_rows).sort_values(['Peak_Severity', 'Failure_Mode', 'Episode_ID']).reset_index(drop=True)

# ── Sheet 3: Failure_Mode_Pivot ───────────────────────────────────────────────
pivot = pd.crosstab(
    results['failure_mode'], results['Severity'],
    margins=True, margins_name='TOTAL'
)
for col in ['P1', 'P2', 'P3', 'P4']:
    if col not in pivot.columns:
        pivot[col] = 0
pivot = pivot[['P1', 'P2', 'P3', 'P4', 'TOTAL']]
pivot_pct = pd.crosstab(
    results['failure_mode'], results['Severity'],
    normalize='index'
).mul(100).round(1)
for col in ['P1', 'P2', 'P3', 'P4']:
    if col not in pivot_pct.columns:
        pivot_pct[col] = 0.0
pivot_pct = pivot_pct[['P1', 'P2', 'P3', 'P4']].rename(columns=lambda c: c + '_%')
sheet3 = pd.concat([pivot, pivot_pct], axis=1).reset_index()

# ── Sheet 4: Indicator_Detail (first 5000 rows to keep file manageable) ───────
ind_cols = [c for c in results.columns if c.startswith('Indicator_')]
id_cols  = ['episode_id', 'failure_mode', 'elapsed_s', 'Severity']
sheet4 = (
    results[[c for c in id_cols + ind_cols if c in results.columns]]
    .sort_values(['episode_id', 'elapsed_s'])
    .head(5000)
    .reset_index(drop=True)
)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Write Excel using xlsxwriter (fast bulk mode)
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_PATH = 'outputs/Severity_Report.xlsx'
print(f"\nWriting Excel: {OUTPUT_PATH} ...")

with pd.ExcelWriter(OUTPUT_PATH, engine='xlsxwriter') as writer:
    wb  = writer.book

    # ── Common formats ────────────────────────────────────────────────────────
    hdr_fmt = wb.add_format({
        'bold': True, 'font_color': 'white',
        'bg_color': '#1F3864', 'border': 1,
        'align': 'center', 'valign': 'vcenter',
        'text_wrap': True
    })
    base_fmt = wb.add_format({'border': 1, 'valign': 'vcenter'})
    pct_fmt  = wb.add_format({'border': 1, 'num_format': '0.0"%"', 'valign': 'vcenter'})
    num_fmt  = wb.add_format({'border': 1, 'num_format': '0.00', 'valign': 'vcenter'})

    # Severity colour formats
    sev_fmts = {
        'P1': wb.add_format({'bold': True, 'font_color': 'white', 'bg_color': '#FF4444', 'border': 1, 'align': 'center'}),
        'P2': wb.add_format({'bold': True, 'font_color': 'white', 'bg_color': '#FF9900', 'border': 1, 'align': 'center'}),
        'P3': wb.add_format({'bold': True, 'font_color': '#333333', 'bg_color': '#FFCC00', 'border': 1, 'align': 'center'}),
        'P4': wb.add_format({'bold': True, 'font_color': 'white', 'bg_color': '#44BB44', 'border': 1, 'align': 'center'}),
    }
    ind_fmts = {
        'CRITICAL': wb.add_format({'bold': True, 'font_color': 'white', 'bg_color': '#FF4444', 'border': 1, 'align': 'center', 'font_size': 9}),
        'WARNING':  wb.add_format({'bold': True, 'font_color': 'white', 'bg_color': '#FF9900', 'border': 1, 'align': 'center', 'font_size': 9}),
        'NORMAL':   wb.add_format({'font_color': '#006600', 'bg_color': '#CCFFCC',    'border': 1, 'align': 'center', 'font_size': 9}),
    }
    peak_fmts = sev_fmts  # same colours for peak severity

    def write_sheet(ws, df, sev_cols=None, ind_col_names=None):
        """Write df to ws with headers, auto-width, and coloured severity/indicator cells."""
        sev_cols      = sev_cols or []
        ind_col_names = ind_col_names or []
        cols = list(df.columns)

        # Header row
        for ci, col_name in enumerate(cols):
            ws.write(0, ci, col_name, hdr_fmt)

        # Data rows
        for ri, row in enumerate(df.itertuples(index=False), start=1):
            for ci, val in enumerate(row):
                col_name = cols[ci]
                # Choose format
                if col_name in sev_cols and str(val) in sev_fmts:
                    ws.write(ri, ci, val, sev_fmts[str(val)])
                elif col_name in ind_col_names and str(val) in ind_fmts:
                    ws.write(ri, ci, val, ind_fmts[str(val)])
                elif pd.isna(val) or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
                    ws.write(ri, ci, '', base_fmt)
                elif isinstance(val, (int, np.integer)):
                    ws.write_number(ri, ci, int(val), base_fmt)
                elif isinstance(val, (float, np.floating)):
                    ws.write_number(ri, ci, float(val), num_fmt)
                else:
                    ws.write(ri, ci, str(val) if val is not None else '', base_fmt)

        # Column widths
        for ci, col_name in enumerate(cols):
            sample = [str(col_name)] + [str(v) for v in df.iloc[:500, ci]]
            width  = min(max(len(s) for s in sample), 35) + 2
            ws.set_column(ci, ci, width)

        # Freeze header + first columns
        ws.freeze_panes(1, min(4, len(cols)))
        ws.set_row(0, 30)   # taller header

    # ── Sheet 1: Episode_Severity ─────────────────────────────────────────────
    print("  Writing Sheet 1: Episode_Severity ...")
    ws1 = wb.add_worksheet('Episode_Severity')
    writer.sheets['Episode_Severity'] = ws1
    write_sheet(ws1, sheet1,
                sev_cols=['Severity', 'RawSeverity'])

    # ── Sheet 2: Episode_Summary ──────────────────────────────────────────────
    print("  Writing Sheet 2: Episode_Summary ...")
    ws2 = wb.add_worksheet('Episode_Summary')
    writer.sheets['Episode_Summary'] = ws2
    write_sheet(ws2, sheet2,
                sev_cols=['Peak_Severity'])

    # ── Sheet 3: FailureMode_Pivot ────────────────────────────────────────────
    print("  Writing Sheet 3: FailureMode_Pivot ...")
    ws3 = wb.add_worksheet('FailureMode_Pivot')
    writer.sheets['FailureMode_Pivot'] = ws3
    write_sheet(ws3, sheet3)

    # ── Sheet 4: Indicator_Detail ─────────────────────────────────────────────
    print("  Writing Sheet 4: Indicator_Detail (first 5000 rows) ...")
    ws4 = wb.add_worksheet('Indicator_Detail')
    writer.sheets['Indicator_Detail'] = ws4
    write_sheet(ws4, sheet4,
                sev_cols=['Severity'],
                ind_col_names=ind_cols)

print(f"\nSaved: {os.path.abspath(OUTPUT_PATH)}")
print("\nSeverity distribution:")
print(results['Severity'].value_counts().sort_index().to_string())
print("\nEpisode Summary (Peak Severity counts):")
print(sheet2['Peak_Severity'].value_counts().to_string())
