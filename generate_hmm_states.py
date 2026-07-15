import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from collections import Counter

# Ensure data directory exists
os.makedirs('data', exist_ok=True)

print("--- PHASE 1: DATA PREPARATION & MERGE ---")
print("Loading metrics, logs, and traces...")
metrics_df = pd.read_csv('data/telemetry_metrics.csv')
logs_df = pd.read_csv('data/telemetry_logs.csv')
traces_df = pd.read_csv('data/telemetry_traces.csv')

# Preprocess circuit breaker state
metrics_df['cb_encoded'] = metrics_df['circuit_breaker_state'].map({'closed': 0, 'half-open': 1, 'open': 2}).fillna(0)

# Merge logs and metrics first
print("Merging metrics and logs...")
merged = pd.merge(
    metrics_df,
    logs_df[['episode_id', 'elapsed_s', 'log_level', 'exception_type', 'log_message']],
    on=['episode_id', 'elapsed_s'],
    how='inner'
)

# Extract log features at step level
merged['log_severity'] = merged['log_level'].map({'': 0, 'INFO': 1, 'WARNING': 2, 'WARN': 2, 'ERROR': 3, 'CRITICAL': 4, 'FATAL': 4}).fillna(0)
merged['log_has_exception'] = merged['exception_type'].apply(lambda x: 0 if pd.isna(x) or x == '' else 1)
merged['log_exception_encoded'] = merged['exception_type'].map({'': 0, 'RuntimeException': 1, 'SocketTimeoutException': 2, 'NullPointerException': 3, 'SystemOverloadException': 4}).fillna(0)

# Vectorized Trace Feature Engineering
print("Engineering trace features...")
is_root = traces_df['parent_span_id'].isna() | (traces_df['parent_span_id'] == '')
is_child = ~is_root

# 1. root_span_error_rate
root_spans = traces_df[is_root]
root_counts = root_spans.groupby(['episode_id', 'elapsed_s']).size()
root_errors = root_spans[root_spans['span_status'] == 'ERROR'].groupby(['episode_id', 'elapsed_s']).size()
root_span_error_rate = (root_errors / root_counts).fillna(0)

# 2. duplicate_trace_id_count
total_spans = traces_df.groupby(['episode_id', 'elapsed_s']).size()
unique_traces = traces_df.groupby(['episode_id', 'elapsed_s'])['trace_id'].nunique()
duplicate_trace_id_count = (total_spans - unique_traces).clip(lower=0)

# 3. distinct_error_services
error_spans = traces_df[traces_df['span_status'] == 'ERROR']
distinct_error_services = error_spans.groupby(['episode_id', 'elapsed_s'])['service'].nunique()

# 4. span_gap_ratio
root_durations = traces_df[is_root].groupby(['episode_id', 'elapsed_s', 'trace_id'])['span_duration_ms'].max()
child_durations = traces_df[is_child].groupby(['episode_id', 'elapsed_s', 'trace_id'])['span_duration_ms'].sum()
trace_metrics = pd.DataFrame({'root_dur': root_durations, 'child_sum': child_durations}).fillna(0)
trace_metrics['gap_ratio'] = ((trace_metrics['root_dur'] - trace_metrics['child_sum']) / trace_metrics['root_dur']).clip(0, 1).fillna(0)
span_gap_ratio = trace_metrics.groupby(['episode_id', 'elapsed_s'])['gap_ratio'].mean()

# 5. uniform_slowdown_score
child_min = traces_df[is_child].groupby(['episode_id', 'elapsed_s', 'trace_id'])['span_duration_ms'].min()
child_max = traces_df[is_child].groupby(['episode_id', 'elapsed_s', 'trace_id'])['span_duration_ms'].max()
child_count = traces_df[is_child].groupby(['episode_id', 'elapsed_s', 'trace_id']).size()
trace_slowdown = pd.DataFrame({'min_dur': child_min, 'max_dur': child_max, 'count': child_count}).fillna(0)
trace_slowdown['slowdown'] = 0.0
trace_slowdown.loc[trace_slowdown['count'] == 1, 'slowdown'] = 1.0
mask_multi = trace_slowdown['count'] > 1
valid_max = trace_slowdown['max_dur'] > 0
trace_slowdown.loc[mask_multi & valid_max, 'slowdown'] = trace_slowdown['min_dur'] / trace_slowdown['max_dur']
trace_slowdown.loc[mask_multi & ~valid_max, 'slowdown'] = 1.0
uniform_slowdown_score = trace_slowdown.groupby(['episode_id', 'elapsed_s'])['slowdown'].mean()

# Combine trace features into a dataframe
trace_features_df = pd.DataFrame({
    'root_span_error_rate': root_span_error_rate,
    'span_gap_ratio': span_gap_ratio,
    'uniform_slowdown_score': uniform_slowdown_score,
    'duplicate_trace_id_count': duplicate_trace_id_count,
    'distinct_error_services': distinct_error_services
}).fillna(0).reset_index()

# Merge trace features with the main dataset
print("Merging trace features...")
merged = pd.merge(merged, trace_features_df, on=['episode_id', 'elapsed_s'], how='left').fillna(0)

print(f"Data shape after merge: {merged.shape}")

print("\n--- PHASE 2: FAILURE MODE CLASSIFICATION ---")
features = [
    'cpu_utilization', 'memory_utilization', 'heap_mb', 'db_p99', 'p99_latency',
    'error_rate', 'disk_write_latency', 'disk_read_latency', 'queue_lag',
    'thread_pool_queue', 'upstream_timeout_rate', 'network_errors',
    'cache_miss_rate', 'cache_hit_rate', 'active_connections', 'db_connection_pool',
    'retry_count_per_request', 'rps', 'cb_encoded', 'log_severity',
    'log_has_exception', 'log_exception_encoded', 'root_span_error_rate',
    'span_gap_ratio', 'uniform_slowdown_score', 'duplicate_trace_id_count',
    'distinct_error_services'
]

# Split at episode level (80/20 train-test stratified split)
episodes = merged.groupby('episode_id').first().reset_index()
X_ep = episodes[['episode_id']]
y_ep = episodes['failure_mode']
X_train_ep, X_test_ep, y_train_ep, y_test_ep = train_test_split(X_ep, y_ep, test_size=0.20, stratify=y_ep, random_state=42)

train_ep_ids = X_train_ep['episode_id'].tolist()
test_ep_ids = sorted(X_test_ep['episode_id'].tolist()) # E00, E01... will correspond to alphabetical sort of test episodes
print(f"Total train episodes: {len(train_ep_ids)}")
print(f"Total test episodes: {len(test_ep_ids)}")

train_steps = merged[merged['episode_id'].isin(train_ep_ids)].copy()
test_steps = merged[merged['episode_id'].isin(test_ep_ids)].copy()

print("Training step-level Random Forest classifier...")
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(train_steps[features], train_steps['failure_mode'])
print("Classifier trained successfully.")

# Predict step-level failure modes for test set
test_steps['predicted_mode'] = clf.predict(test_steps[features])

print("\n--- PHASE 3: GENERATE HMM STATE SEQUENCES ---")
# Map predicted mode and metrics to HMM states with transition boundary rules
def map_prediction_to_state(row):
    elapsed = row['elapsed_s']
    pred = row['predicted_mode']
    db_p99 = row['db_p99']
    
    # 1. Enforce initial healthy state (first 48 seconds, i.e., Window 1)
    if elapsed < 48:
        return 'Normal'
        
    # 2. Enforce final crash/recovery states (last 48 seconds, i.e., Window 5)
    if elapsed >= 192:
        if pred in ['MEMORY_LEAK', 'CPU_SATURATION']:
            return 'Service_Crash'
        else:
            return 'Recovery'
            
    # 3. Active phase mapping with specific classification rules
    if pred == 'NONE':
        return 'Normal'
    elif pred == 'MEMORY_LEAK':
        return 'Memory_Leak'
    elif pred == 'CPU_SATURATION':
        return 'High_CPU'
    elif pred == 'LATENCY_SPIKE':
        if elapsed >= 144:
            return 'Packet_Loss'
        return 'Network_Latency'
    elif pred == 'ERROR_STORM':
        return 'Config_Error'
    elif pred == 'DB_SLOWDOWN' or pred == 'CACHE_STAMPEDE':
        return 'Database_Slowdown'
    elif pred == 'QUEUE_BACKUP':
        return 'Thread_Deadlock'
    elif pred == 'DEPENDENCY_TIMEOUT':
        return 'Dependency_Failure'
    elif pred == 'BAD_DEPLOY':
        return 'Bad_Deploy'
    elif pred == 'RETRY_STORM':
        if elapsed >= 144:
            return 'Database_Slowdown'
        return 'Disk_IO'
    elif pred == 'DISK_IO_SATURATION':
        if elapsed >= 144 and db_p99 > 300:
            return 'Database_Slowdown'
        return 'Disk_IO'
    elif pred == 'CASCADING_FAILURE':
        return 'Resource_Starvation'
        
    return 'Normal'

test_steps['predicted_state'] = test_steps.apply(map_prediction_to_state, axis=1)

# Format detailed state sequence Excel
# Rows: 120 steps, Columns: Time_Sequence, E00, E01, ..., E200
raw_seq_data = {'Time_Sequence': list(range(1, 121))}

# Map sorted episode_ids to E00, E01, ...
ep_to_col = {ep_id: f"E{idx:02d}" for idx, ep_id in enumerate(test_ep_ids)}

for ep_id in test_ep_ids:
    ep_df = test_steps[test_steps['episode_id'] == ep_id].sort_values('elapsed_s')
    col_name = ep_to_col[ep_id]
    raw_seq_data[col_name] = ep_df['predicted_state'].tolist()

raw_seq_df = pd.DataFrame(raw_seq_data)
raw_seq_df.to_excel('data/HMM_Test_Data_State_Sequence.xlsx', index=False)
print("Saved detailed state sequences to data/HMM_Test_Data_State_Sequence.xlsx")

print("\n--- PHASE 4: COMPUTE TUMBLING MODE ---")
# Generates the tumbling mode sequence exactly matching the user's requested format.
# Output Columns: Episode, Window, Observation, Mode. Sorted by Window then Episode.
tumble_rows = []

# Window sizes: we will generate 60-second windows (4 windows: 1-30, 31-60, 61-90, 91-120)
# and also 48-second windows (5 windows: 1-24, 25-48, 49-72, 73-96, 97-120)
# We will save the 60s version and 48s version.
for win_size, num_steps in [('60s', 30), ('48s', 24)]:
    win_rows = []
    n_windows = 120 // num_steps
    
    for w in range(n_windows):
        start_step = w * num_steps + 1
        end_step = (w + 1) * num_steps
        obs_range = f"{start_step}-{end_step}"
        window_num = w + 1
        
        for ep_id in test_ep_ids:
            col_name = ep_to_col[ep_id]
            ep_states = raw_seq_data[col_name][start_step-1 : end_step]
            # Compute mode of states in the window
            mode_state = Counter(ep_states).most_common(1)[0][0]
            
            win_rows.append({
                'Episode': col_name,
                'Window': window_num,
                'Observation': obs_range,
                'Mode': mode_state
            })
            
    tumble_df = pd.DataFrame(win_rows)
    
    # Save files
    if win_size == '60s':
        tumble_df.to_excel('data/HMM_Test_Data_State_Sequence_With_Tumbling_Mode_60s.xlsx', index=False)
        print("Saved 60s tumbling mode to data/HMM_Test_Data_State_Sequence_With_Tumbling_Mode_60s.xlsx")
    else:
        tumble_df.to_excel('data/HMM_Test_Data_State_Sequence_With_Tumbling_Mode.xlsx', index=False)
        tumble_df.to_csv('data/HMM_Test_Data_State_Sequence_With_Tumbling_Mode.csv', index=False)
        print("Saved 48s tumbling mode to data/HMM_Test_Data_State_Sequence_With_Tumbling_Mode.xlsx and .csv")

print("\nPipeline completed successfully!")
