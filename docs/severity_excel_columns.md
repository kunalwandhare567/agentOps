# Severity Report Excel Columns Documentation

This document explains the columns and structure of the generated severity Excel workbook, [Severity_Report.xlsx](file:///d:/DEVOPS/outputs/Severity_Report.xlsx), produced by the [generate_severity_excel.py](file:///d:/DEVOPS/generate_severity_excel.py) script.

The workbook contains four formatted sheets designed to analyze and summarize severity decisions across the telemetry dataset.

---

## Sheet 1: `Episode_Severity`
This sheet contains row-level details for every 2-second step of each episode, showing how raw telemetry features translate to raw and smoothed severity classifications.

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| **`episode_id`** | Text | Unique identifier for the specific telemetry episode. |
| **`failure_mode`** | Text | The simulated failure mode active during the episode step (e.g., `CPU_SATURATION`, `MEMORY_LEAK`, `QUEUE_BACKUP`). |
| **`elapsed_s`** | Integer | Elapsed seconds since the start of the current episode (ranges from `0` to `238` in 2-second intervals). |
| **`Severity`** | Text | The final **temporarily smoothed severity level** (`P1`, `P2`, `P3`, `P4`) after applying Exponential Moving Average (EMA) and hysteresis rules. |
| **`RawSeverity`** | Text | The raw severity level (`P1`, `P2`, `P3`, `P4`) determined by the deterministic 6-rule decision tree *before* temporal smoothing. |
| **`WeightedScore`** | Float | The aggregated severity score. Computed as the sum of each feature's effective severity level multiplied by its failure-mode-specific weight override. |
| **`CriticalCount`** | Integer | The number of features evaluated as `CRITICAL` during this cycle (including those escalated from `WARNING` via weight overrides). |
| **`WarningCount`** | Integer | The number of features evaluated as `WARNING` or `HIGH` during this cycle (excluding those escalated to `CRITICAL`). |
| **`BlastSize`** | Integer | Total number of anomalous features. Derived as: $\text{BlastSize} = \text{CriticalCount} + \text{WarningCount}$. |
| **`HighRiskMode`** | Boolean | `TRUE` if any `CRITICAL` feature belongs to the high-risk categories: `cpu_utilization`, `memory_utilization`, `error_rate`, `root_span_error_rate`, or `distinct_error_services`. |
| **`BlastRadiusGrowing`** | Boolean | `TRUE` if the number of distinct services reporting errors (`distinct_error_services`) strictly increased since the previous cycle. |
| **`Reason`** | Text | Detailed explanation listing the primary features responsible for the severity levels, high-risk status, and blast radius changes. |
| **`RecommendedAction`** | Text | Actionable mitigation recommendations based on the predicted failure mode and severity level. |
| **`cpu_utilization`** | Float | Raw CPU usage percentage. |
| **`memory_utilization`** | Float | Raw memory utilization ratio. |
| **`heap_mb`** | Float | Raw JVM Heap usage in megabytes. |
| **`p99_latency`** | Float | Raw 99th-percentile response latency in milliseconds. |
| **`error_rate`** | Float | Raw HTTP error rate ratio. |
| **`queue_lag`** | Integer | Raw message backlog / queue lag in milliseconds. |
| **`cache_hit_rate`** | Float | Cache hit rate ratio. |
| **`cache_miss_rate`** | Float | Cache miss rate ratio. |
| **`db_p99`** | Float | Raw 99th-percentile database response latency in milliseconds. |
| **`disk_write_latency`** | Float | Raw disk write latency in milliseconds. |
| **`upstream_timeout_rate`** | Float | Raw rate of timeout responses from upstream services. |
| **`retry_count_per_request`** | Float | Average count of retries per request. |
| **`root_span_error_rate`** | Float | Trace-derived error rate of root spans. |
| **`distinct_error_services`**| Integer | Count of unique services exhibiting trace errors. |

---

## Sheet 2: `Episode_Summary`
This sheet aggregates telemetry metrics to present an episode-level view of severity impact, highlighting peak severities and the duration profile of each episode.

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| **`Episode_ID`** | Text | Unique identifier for the episode. |
| **`Failure_Mode`** | Text | Active failure mode of the episode. |
| **`Total_Steps`** | Integer | Total number of telemetry cycles (rows) recorded for the episode (typically `120`). |
| **`Peak_Severity`** | Text | The maximum severity reached (`P1` is worst, `P4` is normal) at any point during the episode. |
| **`Steps_P1` / `_P2` / `_P3` / `_P4`**| Integer | Total step counts spent in each smoothed severity level during the episode. |
| **`Pct_P1` / `_P2` / `_P3` / `_P4`** | Float | Percentage of total episode duration spent in each smoothed severity level. |
| **`Max_CriticalCount`** | Integer | Peak count of critical features observed in a single cycle during the episode. |
| **`Max_WeightedScore`** | Float | Peak aggregated weighted score observed during the episode. |
| **`Avg_CriticalCount`** | Float | Average number of critical features per cycle over the entire episode duration. |

---

## Sheet 3: `Failure_Mode_Pivot`
A contingency table (crosstab) summarizing the overall distribution of severity classifications across different failure modes.

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| **`failure_mode`** | Text | Name of the system failure mode or `TOTAL` for row sums. |
| **`P1` / `P2` / `P3` / `P4`** | Integer | Volume of row-level telemetry steps classified at each severity. |
| **`TOTAL`** | Integer | Total telemetry rows processed for the corresponding failure mode. |
| **`P1_%` / `P2_%` / `P3_%` / `P4_%`**| Float | Percentage distribution of severity levels for the specific failure mode. |

---

## Sheet 4: `Indicator_Detail`
To help tune thresholds, this sheet provides a sample (first 5,000 rows) showing the evaluated category labels (`NORMAL`, `WARNING`, `CRITICAL`) for every individual feature before aggregation.

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| **`episode_id`** | Text | Unique identifier for the telemetry episode. |
| **`failure_mode`** | Text | Active failure mode. |
| **`elapsed_s`** | Integer | Elapsed seconds. |
| **`Severity`** | Text | Smooth severity level. |
| **`Indicator_<Feature>`** | Text | Evaluated status (`NORMAL`, `WARNING`, or `CRITICAL`) for each specific metric/log/trace feature (e.g., `Indicator_cpu_utilization`, `Indicator_queue_lag`). |
