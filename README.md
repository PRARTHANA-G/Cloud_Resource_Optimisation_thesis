# Optimizing Cloud Resource Allocation: A Comparative Simulation of Statistical Rightsizing

**Master's Thesis — Frankfurt School of Finance & Management, Applied Data Science (Intake 2024)**  
**Authors:** Prarthana Govindaraj · Pinki Kumari  
**Supervisors:** Prof. Dr. Jan Nagler · Prof. Levente Szabados  
**Submitted:** August 2026

---

## Overview

Cloud batch workloads are routinely provisioned based on operator intuition rather than data. The consequences show up directly in production traces: jobs either consume more CPU at peak than they were ever allocated, or they request cores they barely touch. Both failure modes exist simultaneously in the same cluster.

This thesis uses the **Alibaba Cluster Trace 2018** to study the gap between requested and actual CPU consumption at scale. After preprocessing — including a critical correction to a column swap in the raw data that affects every peak CPU analysis from this file — we retain **11,644 terminated batch jobs** and **16,094,656 instance-level records** for analysis.

The research is structured around **three progressive modelling stages** and a **complementary behavioural analysis**, each asking a different question about the same data.

---

## Key Findings

- **44.2% of terminated batch jobs are under-provisioned** at peak demand, with the provisioning error being roughly as common as over-provisioning
- A **sharp threshold at 2 requested CPU cores** separates two categorically different provisioning regimes: below it, under-provisioning rates range from 63% to 98.6%; above 5 cores, over-provisioning dominates
- **72.9% of all CPU requests are round numbers** — operators anchor on salient values (1, 2, 5 cores) rather than computing calibrated estimates from actual usage data
- **11.5% of jobs fall in a near-miss zone** (utilisation ratio 0.8–1.0), dangerously close to exhaustion with no feedback mechanism to flag it
- The **LSTM outperforms the Random Forest** on all three metrics (MAE 0.736 vs 0.796, RMSE 1.192 vs 1.619, R² 0.779 vs 0.694), concentrated in fewer extreme mispredictions at the high-severity tail
- **Chronos zero-shot fails substantially** (MAE 1.456, RMSE 2.610, R² 0.165) with a systematic underprediction bias of 1.435 cores — cloud batch CPU traces fall outside the effective zero-shot range of the current model
- On a statistical forecasting baseline using Nixtla StatsForecast, **Naive outperforms AutoARIMA, AutoETS, and SeasonalNaive** on the fair common-series benchmark, consistent with the execution-driven, non-periodic structure of batch CPU sequences

---

## Data Quality Finding

The `cpu_max` and `cpu_avg` columns in the raw `batch_instance.csv` file are **swapped** relative to their documented definitions. The column labelled `cpu_max` consistently contains values lower than `cpu_avg` across the vast majority of instances — physically impossible, since a maximum cannot be lower than a mean over the same interval.

All results in this repository use the corrected values. This is documented here for the benefit of future researchers using the Alibaba 2018 trace.

---

## Project Structure

```
Cloud_Resource_Optimisation_thesis/
│
├── data/
│   ├── raw/                          # Raw Alibaba 2018 trace files (not included)
│   │   └── clusterdata2018/
│   │       └── trace_201708/
│   │           ├── batch_task.csv
│   │           └── batch_instance.csv
│   └── processed/                    # Preprocessed outputs
│       ├── stage1b_merged_clean.csv  # 11,644-job analytical dataset
│       ├── nixtla_job_panel.csv      # Nixtla-ready panel dataset
│       └── ...
│
├── notebooks/
│   ├── data_mapping.ipynb  #Raw data anlysis
│   ├── 01_data_exploration.ipynb #Raw data analysis
│   ├── batch_instance_cpu_max_vs_avg.ipynb  #Column check
│   ├── EDAV2.ipynb                   # Exploratory analysis, column swap correction, Stage 1 EDA
│   ├── stage1b_provisioning_analysis.ipynb  # Stage 1 full pipeline
│   ├── stage1_underprovisioning_V2_analysis.ipynb  #Underprovisioning analysis
│   ├── stage1_EDA_V3_uniform join key.ipyn  #join key analysis
│   ├── lstm_v3.ipynb                 # Stage 2 — LSTM forecasting
│   ├── chronos.ipynb                 # Stage 3 — Chronos zero-shot evaluation
│   ├── nixtla_time_series_analysis.ipynb      # Stage 3 — Nixtla statistical baseline
│   └── risk_analysis.ipynb           # Behavioural analysis — risk premiums
│
├── results/                          # Model outputs and metrics
│
└── README.md
```

---

## Analytical Stages

### Stage 1 — Random Forest Classification and Rightsizing
*Prarthana Govindaraj & Pinki Kumari*

Asks whether under-provisioning can be detected at job submission time using only metadata observable before execution begins.

- **Feature engineering:** 8 features from job and instance metadata (plan_cpu_cores, task_count, instance_count, and derived ratios)
- **Classifier:** Random Forest with `class_weight='balanced'`, n_estimators=100 — ROC-AUC **0.908**, Precision **0.818**, F1 **0.802**
- **Severity regression:** Random Forest regressor on under-provisioned jobs — R² **0.694**, MAE **0.796**
- **Recommendation engine:** Severity-adjusted safety buffers (10–25%) with floor at observed cpu_max_peak

Key finding: the 2-core threshold, independently confirmed by both the utilisation ratio analysis and the risk premium analysis in Stage 4.

---

### Stage 2 — LSTM Peak CPU Forecasting
*Prarthana Govindaraj*

Asks whether a sequence model trained on early execution data can outperform the static Random Forest baseline.

- **Input:** First 40 `cpu_max` readings per job, ordered by (task_name, seq_no), left-zero-padded
- **Target:** Maximum `cpu_max` across all remaining readings after position 40 (genuine forecasting task)
- **Architecture:** 2-layer LSTM (64 units), batch normalisation, dropout 0.3, Linear(64→32)→ReLU→Linear(32→1)
- **Training:** Adam lr=0.001, MSE loss, gradient clipping max_norm=1.0, 60 epochs
- **Results:** MAE **0.736**, RMSE **1.192**, R² **0.779**, mean residual **-0.020** (essentially unbiased)

Architecture kept deliberately simple (no BiLSTM, no attention, no VMD) to isolate training regime as the variable of interest in the Stage 2 vs Stage 3 comparison.

---

### Stage 3 — Zero-Shot and Statistical Forecasting Baselines
*Prarthana Govindaraj (Chronos) · Pinki Kumari (Nixtla)*

Asks whether approaches that avoid domain-specific training can match the LSTM.

**Chronos (zero-shot foundation model):**
- Model: `amazon/chronos-t5-small` (46M parameters), zero-shot, no fine-tuning
- Same 40-step input sequences as the LSTM; median of 20 sampled trajectories; max of 10-step forecast as predicted peak
- Results: MAE **1.456**, RMSE **2.610**, R² **0.165**, mean residual **+1.435** (systematic underprediction)
- Failure is structural: batch CPU traces are execution-driven, episodic, and non-periodic — outside Chronos's effective zero-shot range

**Nixtla StatsForecast (statistical baseline):**
- Six models evaluated: AutoARIMA, AutoETS, Naive, SeasonalNaive, HistoricAverage, WindowAverage
- Fair common-series comparison (153 series): **Naive wins** (MAE 10.75) over AutoARIMA (16.46)
- AutoARIMA covers only 4.7% of series — too narrow for general deployment
- Consistent with Chronos: the data's lack of temporal structure defeats pattern-based models

---

### Behavioural Analysis — Provisioning Gaps as Revealed Risk Premiums
*Prarthana Govindaraj*

Reframes the provisioning gap as a revealed insurance decision and tests whether operator behaviour is consistent with rational risk management.

**Risk premium definition:**
```
risk_premium_pct = (plan_cpu_cores - cpu_max_peak) / cpu_max_peak × 100
```

**Findings:**
| Size Bucket | Jobs | Mean Premium | Median Premium |
|---|---|---|---|
| Tiny (0, 0.5] | 438 | -69.4% | -75.0% |
| Small (0.5, 1.0] | 3,867 | +169.4% | +11.1% |
| Medium (1.0, 2.0] | 1,409 | -37.3% | -50.0% |
| Large (2.0, 5.0] | 3,253 | +29.9% | +23.8% |
| XLarge (5.0, 10.0] | 2,193 | +153.2% | +170.9% |
| Huge (10.0+) | 484 | +621.5% | +550.0% |

The rational risk management hypothesis is **not supported**. Operators most exposed to tail risk (Tiny, Medium) pay the most negative premiums. The mechanisms: anchoring bias (72.9% round-number requests) and absent feedback loops (no visibility into near-miss frequency at submission time).

---

## Model Comparison

| Model | MAE | RMSE | R² | Notes |
|---|---|---|---|---|
| Random Forest (Stage 1) | 0.796 | 1.619 | 0.694 | Metadata only, submission time |
| LSTM (Stage 2) | 0.736 | 1.192 | 0.779 | Sequence model, in-distribution |
| Chronos (Stage 3) | 1.456 | 2.610 | 0.165 | Zero-shot, systematic bias |

All three evaluated on the same held-out test set of 1,686 jobs with identical metrics.

---

## Dataset

**Alibaba Cluster Trace 2018** — publicly available at [GitHub: alibaba/clusterdata](https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2018)

- ~4,000 machines, 8-day observation window
- `batch_task.csv`: 80,553 rows (job/task-level resource requests)
- `batch_instance.csv`: 16,094,656 rows (instance-level CPU consumption)

> **Important:** Apply the column swap correction before any analysis. The `cpu_max` and `cpu_avg` columns in the raw `batch_instance.csv` are transposed. See `EDAV2.ipynb` for the correction.

---

## Dependencies

**Python 3.12+**

```bash
# Core data stack
pip install pandas numpy scikit-learn matplotlib seaborn

# Deep learning (Stage 2)
pip install torch

# Foundation model (Stage 3 — Chronos)
pip install chronos-forecasting

# Statistical forecasting (Stage 3 — Nixtla)
pip install statsforecast

# Optional
pip install pyarrow  # for parquet support
```

---

## Citation

If you use this work , please cite:

```
Govindaraj, P. and Kumari, P. (2026) Optimizing Cloud Resource Allocation:
A Comparative Simulation of Statistical Rightsizing. Master's Thesis,
Frankfurt School of Finance & Management, Applied Data Science, Intake 2024.
```

---

## Licence

This repository is made available for academic and research purposes.  
The Alibaba Cluster Trace 2018 is subject to its own licence terms — see the [original repository](https://github.com/alibaba/clusterdata).
