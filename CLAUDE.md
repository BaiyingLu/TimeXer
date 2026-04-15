# TimeXer — Blood Glucose Forecasting (Phase 3+)
# CLAUDE.md — Project Memory File

---

## Project Goal

This repo is the TimeXer replication and extension for blood glucose forecasting.
Core metric: **Δ_drivers = RMSE(CGM-only) − RMSE(multivariate)**, computed per window type.

Core hypothesis:
- **H1:** Δ_drivers is larger in MEAL/BOLUS/EXERCISE than STABLE/NOCTURNAL
- **H2:** Event-conditioning (Phase 4) further reduces MEAL/BOLUS RMSE vs standard TimeXer
- **H3:** Δ_drivers magnitude correlates with subject CV%

---

## Current Status

| Phase | Description | Status |
|-------|-------------|--------|
| 3 | TimeXer baseline (original, unmodified) — BrisT1D + OhioT1DM | ⬜ In progress |
| 4 | Event-Conditioned TimeXer (core contribution) | ⬜ |
| 5 | Ablations + write-up | ⬜ |

LSTM baselines (Phases 1–2c) are complete in the parent repo (`multimodal_ts_llm`).
Do NOT re-run or replicate LSTM work here.

---

## LSTM Baseline Results (Reference — do not modify)

### CGM-only LSTM per-window RMSE (30-min horizon)

| Window | Ohio RMSE | BrisT1D RMSE |
|--------|-----------|--------------|
| stable_baseline | 19.09 | 20.75 |
| nocturnal | 13.19 | 17.94 |
| insulin_bolus | 20.57 | 25.62 |
| meal | 23.83 | 35.65 |
| exercise | 31.07 | 24.82 |
| mixed | 25.26 | 31.04 |

### Δ_drivers: Multivariate LSTM (log1p + z-score) − CGM-only

| Window | Ohio Δ_RMSE | BrisT1D Δ_RMSE |
|--------|-------------|----------------|
| stable_baseline | −2.48 | +0.08 |
| nocturnal | −0.30 | +0.20 |
| insulin_bolus | −1.08 | −0.20 |
| meal | +0.06 | **+0.95** |
| exercise | −2.92 (n=145, unreliable) | +0.56 |
| mixed | −2.12 | **+0.76** |

Key takeaway: vanilla LSTM cannot leverage exogenous features well — it learns glucose
autoregression first and exogenous weights stay weak. TimeXer's asymmetric attention
(exo→endo only) is expected to fix this.

---

## Datasets

### OhioT1DM
- **Subjects:** 559, 563, 570, 575, 588, 591 (6 subjects, ~8 weeks each)
- **Frequency:** 5-min CGM
- **Columns:** `timestamp, glucose, basal_amount, dose, total_insulin, carbs, steps`
- **Units:** glucose in mg/dL
- **Canonical source:** `multimodal_ts_llm/ohiot1dm/filter_train/` and `filter_test/`
- **Prepared CSVs for TimeXer:** `dataset/OhioT1DM/` (to be created — see Step 1)

### BrisT1D
- **Subjects:** P01–P24 (P08, P09, P14, P20 absent) = 20 subjects, ~7 months each
- **Frequency:** 5-min CGM
- **Columns:** `bg (mmol/L → ×18.018 → mg/dL), basal, bolus, insulin, carbs, steps`
- **Canonical source:** `multimodal_ts_llm/Bris-T1D Open/filter/`
- **Prepared CSVs for TimeXer:** `dataset/BrisT1D/` (to be created — see Step 1)
- Always `replace('', np.nan).astype(float)` — empty strings are missing values
- `bolus > 0` triggers INSULIN_BOLUS (equivalent to Ohio `dose > 0`)

### HUPA-UCM (future, if needed)
- `dataset/HUPA-UCM/` directory exists but is empty
- Integrate only after Phase 3 results are confirmed

---

## Window Definitions (for post-hoc per-window evaluation)

| Window | Trigger | Duration | Priority |
|--------|---------|----------|----------|
| MEAL | carbs > 0 | [event, +90 min] | 1 |
| INSULIN_BOLUS | dose/bolus > 0 | [+10, +120 min] | 2 |
| EXERCISE | steps ≥ 200, ≥3 rows | [bout−15, bout+60 min] | 3 |
| NOCTURNAL | 00:00–06:00 | fixed | 4 |
| STABLE_BASELINE | no event ±90 min | residual | 5 |
| MIXED | any overlap | kept separate | — |

Window assignment is **post-hoc** on test predictions — NOT used during training.

---

## Implementation Plan (Phase 3)

### Step 1 — Prepare unified CSVs

Target format for all CSVs (one per dataset):
```
date, participant_id, carbs, total_insulin, steps, glucose
```
- `participant_id` is a metadata column (subject ID string, e.g. `"559"`, `"P01"`); never fed to the model
- `glucose` must be **last column** (required for `MS` mode in TimeXer)
- `date` column: ISO format timestamp, timezone-naive, 5-min grid
- Apply `replace('', np.nan)` for BrisT1D; convert `bg` mmol/L → mg/dL (×18.018)
- OhioT1DM: concatenate all subjects, sorted by subject then timestamp → 70/15/15 split **within each subject** by time, then merge splits
- BrisT1D: same concatenation pattern → split by time **within each subject**, then merge

Produce per-split files:
```
dataset/OhioT1DM/ohio_train.csv
dataset/OhioT1DM/ohio_val.csv
dataset/OhioT1DM/ohio_test.csv
dataset/BrisT1D/bris_train.csv
dataset/BrisT1D/bris_val.csv
dataset/BrisT1D/bris_test.csv
```

Write prep script: `scripts/prepare_glucose_data.py`

### Step 2 — Write Dataset_BGlucose

Add to `data_provider/data_loader.py`.

#### Normalization strategy

| Feature | Transform | Scaler scope | Rationale |
|---------|-----------|-------------|-----------|
| `carbs` | log1p → z-score | **per-participant** | Zero-inflated; carb-counting precision varies per person |
| `total_insulin` | log1p → z-score | **per-participant** | TDD and ISF vary 2–10× across individuals |
| `steps` | log1p → z-score | **per-participant** | Baseline activity level varies widely |
| `glucose` | z-score (no log1p) | **population-level** | Range 70–400 mg/dL is universal; global scaler keeps `inverse_transform` compatible with the exp loop, which calls it on flat arrays without participant IDs |

Scaler fitting is **train split only**.
Scalers are saved to `{root_path}/scalers.pkl`:
```python
{
  'exo': { pid: StandardScaler(3 features) for each participant },
  'glucose': StandardScaler(1 feature)   # population-level
}
```
Val/test loads from pkl — fail fast with a clear message if not found (train must run first).

#### NaN handling (applied before log1p / scaling)

| Column | Rule | Rationale |
|--------|------|-----------|
| `glucose` | Mark row as `segment_break`; do NOT fill | Broken CGM sequence; forward-filling would corrupt window targets |
| `carbs` | Fill 0.0 | No carb record = no eating event |
| `steps` | Fill 0.0 | No step record = no activity |
| `total_insulin` | Forward-fill within participant, then 0.0 | Basal+bolus reported at injection time; ffill is physiologically reasonable |

#### Gap / boundary detection

Build `segment_break` boolean array (length N):

```python
segment_break[0] = True  # always
segment_break[i] = True if any of:
  - participant_id[i] != participant_id[i-1]   # subject boundary
  - date[i] - date[i-1] > pd.Timedelta('30min') # sensor gap
  - glucose[i] is NaN                            # missing glucose (detected before fill)
```

Build `valid_indices` using the cumsum trick (O(N)):
```python
L = seq_len + pred_len
cum = np.cumsum(segment_break)
valid_indices = [i for i in range(N - L + 1) if cum[i + L - 1] == cum[i]]
```

`__len__` = `len(valid_indices)`. `__getitem__` indexes into `valid_indices`.

#### Constructor flow

```
1. Resolve actual CSV from data_path + flag
   (strip _train/_val/_test suffix, re-add _{flag}.csv)
2. Read CSV; parse 'date' to datetime; extract participant_id as metadata array
3. Detect glucose NaN rows (before any fill) → pre-mark segment_break candidates
4. Fill NaNs per rules above
5. Apply log1p to exo features [carbs, total_insulin, steps]
6. Build full segment_break array (boundary + time gap + NaN glucose)
7. Normalization:
     if flag == 'train':
       fit per-participant StandardScaler on exo (only that participant's train rows)
       fit global StandardScaler on all train glucose
       save both to {root_path}/scalers.pkl
     else:
       load {root_path}/scalers.pkl  (KeyError → clear error message)
8. Apply per-participant exo scaler to each participant's rows (columns 0–2)
9. Apply global glucose scaler to all rows (column 3)
10. Build valid_indices via cumsum trick
11. Build data_stamp (time feature encoding for encoder/decoder marks)
```

#### `__getitem__` (identical signature to `Dataset_ETT_hour`)

```python
def __getitem__(self, index):
    i = self.valid_indices[index]
    s_end = i + self.seq_len
    r_end = s_end + self.pred_len
    seq_x      = self.data[i:s_end]
    seq_y      = self.data[s_end - self.label_len : r_end]
    seq_x_mark = self.data_stamp[i:s_end]
    seq_y_mark = self.data_stamp[s_end - self.label_len : r_end]
    return seq_x, seq_y, seq_x_mark, seq_y_mark
```

#### `inverse_transform` (compatible with existing exp loop)

```python
def inverse_transform(self, data):
    # data: (N, 4) — exp code calls this on tiled MS output then slices [:, :, -1]
    # Only the last column (glucose) needs to be correctly inverted
    out = data.copy()
    out[:, -1] = self.glucose_scaler.inverse_transform(data[:, -1:]).ravel()
    return out
```

The exp loop (`exp_long_term_forecasting.py`) tiles the single-column MS output to 4 columns,
calls `inverse_transform`, then slices `outputs[:, :, f_dim:]` — so only glucose is used.
No changes to `data_factory.py` or the exp loop are needed.

#### Constructor signature (must stay compatible with `data_factory.py`)

```python
class Dataset_BGlucose(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ohio_train.csv',
                 target='glucose', scale=True, timeenc=0, freq='t',
                 seasonal_patterns=None):
```

### Step 3 — Register in data_factory

In `data_provider/data_factory.py`, add:
```python
from data_provider.data_loader import ..., Dataset_BGlucose

data_dict = {
    ...
    'OhioT1DM': Dataset_BGlucose,
    'BrisT1D':  Dataset_BGlucose,
}
```

### Step 4 — Write training shell scripts

`scripts/forecast_exogenous/OhioT1DM/TimeXer.sh`:
```bash
python run.py \
  --task_name long_term_forecast \
  --model TimeXer \
  --data OhioT1DM \
  --root_path ./dataset/OhioT1DM/ \
  --data_path ohio_train.csv \   # Dataset_BGlucose handles val/test internally
  --features MS \
  --target glucose \
  --seq_len 24 \        # 2hr lookback (24 × 5min)
  --label_len 6 \       # 30min decoder warm-up
  --pred_len 6 \        # 30min horizon
  --freq t \
  --enc_in 4 \          # carbs, total_insulin, steps, glucose
  --dec_in 4 \
  --c_out 1             # predict glucose only (MS mode)
```

Mirror for `scripts/forecast_exogenous/BrisT1D/TimeXer.sh`.

### Step 5 — Post-hoc per-window evaluation

After `test()` saves `pred.npy` and `true.npy`:
1. Load test-split timestamps
2. Assign window types via the same `assign_window_types` logic from `multimodal_ts_llm`
3. Compute per-window RMSE
4. Compute Δ_drivers = LSTM_RMSE − TimeXer_RMSE per window (to quantify architectural gain)

Notebook: `notebooks/05_inspect_TimeXer.ipynb`

---

## Phase 4 — Event-Conditioned TimeXer (core contribution)

Single change to TimeXer: inject window_type embedding into global token initialization.

```python
# In models/TimeXer.py, inside forward():
window_emb = nn.Embedding(6, d_model)   # 6 window types
Gen = Gen + window_emb(window_id)       # residual, no other architectural change
```

The global token (`Gen`) is TimeXer's bridge between exogenous and endogenous streams.
Conditioning it on physiological context allows the model to upweight carbs in MEAL
windows and steps in EXERCISE windows before cross-attention fires.

`window_id` is passed as an extra input tensor (batch of integer window type labels).
Window labels are assigned at sliding-window construction time in `Dataset_BGlucose`.

### Ablation plan

| Ablation | Description | Tests |
|----------|-------------|-------|
| A1 | Remove window embedding | = standard TimeXer (Phase 3) |
| A2 | Remove all exogenous | = CGM-only |
| A3 | Random vs learned window embedding | Is semantic conditioning necessary? |

---

## Multivariate Feature Convention

Features (in column order): `[carbs, total_insulin, steps, glucose]`
- `glucose` is always the **last column** — required for `MS` mode
- Preprocessing:
  - `log1p(carbs, total_insulin, steps)` → **per-participant** z-score
  - `glucose` → **population-level** z-score (no log1p)
- Target glucose returned in raw mg/dL (inverse_transform at eval time using global glucose scaler)
- NaN fills: glucose → segment_break (exclude window) | carbs/steps → 0.0 | insulin → ffill within participant then 0.0
- `participant_id` column in CSVs is metadata only — used for boundary detection and scaler fitting, never passed to model

---

## Forecasting Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `seq_len` | 24 | 2hr lookback at 5-min resolution |
| `label_len` | 6 | 30-min decoder warm-up |
| `pred_len` | 6 | 30-min prediction horizon |
| `freq` | `t` | minute-level time features |
| `features` | `MS` | multi-in, single-out (glucose) |
| `enc_in` | 4 | carbs, total_insulin, steps, glucose |

---

## File Layout

```
TimeXer/
├── data_provider/
│   ├── data_factory.py       # add OhioT1DM, BrisT1D entries
│   └── data_loader.py        # add Dataset_BGlucose
├── dataset/
│   ├── OhioT1DM/             # ohio_{train,val,test}.csv
│   ├── BrisT1D/              # bris_{train,val,test}.csv
│   └── HUPA-UCM/             # future
├── scripts/
│   ├── prepare_glucose_data.py          # Step 1 prep script
│   └── forecast_exogenous/
│       ├── OhioT1DM/TimeXer.sh
│       └── BrisT1D/TimeXer.sh
├── notebooks/
│   └── 05_inspect_TimeXer.ipynb        # per-window eval
└── CLAUDE.md
```

---

## Coding Conventions

- Python 3.10+, pandas Timestamp (timezone-naive), snake_case, pathlib.Path
- Do NOT modify canonical source data in `multimodal_ts_llm/`
- Do NOT forward-fill glucose across gaps
- Do NOT slide windows across subject boundaries or sensor gaps > 30 min
- Do NOT hardcode subject IDs inside functions
- Do NOT apply a global z-score to exo features (carbs, total_insulin, steps) — use per-participant
- `Dataset_BGlucose` constructor signature must stay compatible with `data_factory.py`
- Gap-aware windowing is critical — silence here corrupts all per-window results
- Scaler pkl must be fit on train split only; val/test must load from that pkl (not refit)
- `participant_id` column in CSVs is required for both gap detection and per-participant scaling

---

## Target Conferences

| Venue | Deadline | Notes |
|-------|----------|-------|
| NeurIPS Workshop (TS4H) | ~Aug 2026 | Non-archival; test water |
| **AAAI 2027** | **Aug 1, 2026** | **Primary target** |
| ICLR 2027 | ~Sep/Oct 2026 | High bar |
| MLHC 2027 | ~Apr 2027 | Best domain fit if results need time |

Recommended path: finish Phase 4 by June 2026 → submit TS4H workshop Aug 2026 →
submit AAAI 2027 by Aug 1, 2026.
