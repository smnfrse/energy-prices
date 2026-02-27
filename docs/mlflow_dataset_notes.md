# MLflow Dataset Logging — Current State and Cleanup Notes

## What We're Doing Now (and Why It's Messy)

Preprocessed datasets (feature matrices X, y) are stored in MLflow using
`mlflow.sklearn.log_model()`, treating a parquet-backed dataset as if it were
a trained model. This works but conflates two distinct concepts:

```
"datasets" experiment
├── run abc123   ← actually a dataset, but stored like a model
│   ├── metadata/X.parquet
│   ├── metadata/y.parquet
│   ├── metadata/metadata.json
│   └── pipeline/   ← unfitted preprocessing pipeline (MLmodel format)
└── run xyz789   ← another dataset
    └── ...
```

The `pipeline` artifact is stored via `mlflow.sklearn.log_model(pipeline, "pipeline")`,
which means MLflow thinks it's a trained model. Loading it with
`mlflow.sklearn.load_model(f"runs:/{run_id}/pipeline")` works but is semantically wrong.

**Specific problems:**
1. `mlflow.search_runs(search_all_experiments=True, filter_string="metrics.mae > 0")`
   in `blend.py` will match dataset runs if they ever accidentally log a `mae` metric.
   Currently avoided by filtering on `params.model_class` (dataset runs don't have this).
2. The MLflow UI shows datasets and models interleaved — no visual distinction.
3. `load_pipeline()` returns an **unfitted** pipeline, but it's loaded via the
   `mlflow.sklearn` model API, implying it's a fitted estimator. Misleading.
4. No use of MLflow's built-in dataset provenance tracking (`mlflow.log_input`).

---

## What MLflow Actually Provides

MLflow has a proper dataset logging API (since MLflow 2.4):

```python
# Log a dataset as an input to a run
dataset = mlflow.data.from_pandas(df, source="data/processed/merged_dataset_hourly.parquet")
mlflow.log_input(dataset, context="training")
```

This attaches dataset provenance to model runs (not the other way around), which
is the correct relationship: *models are trained on datasets*, not *datasets contain models*.

---

## Proposed Cleanup

### Option A: Separate the "datasets" experiment properly

Keep the current storage format (parquets in MLflow artifacts) but:

1. **Tag dataset runs distinctly** so they never appear in model searches:
   ```python
   mlflow.set_tag("run_type", "dataset")   # add to build_pipeline()
   ```
   Then `blend.select_candidates()` can filter `tags.run_type != 'dataset'`.

2. **Rename the pipeline artifact** from `"pipeline"` to `"preprocessing_pipeline"`
   and log it as a raw artifact (pickle/joblib) rather than via `mlflow.sklearn.log_model`:
   ```python
   # In _save_dataset_to_mlflow:
   import tempfile, joblib
   with tempfile.NamedTemporaryFile(suffix=".joblib") as f:
       joblib.dump(pipeline, f.name)
       mlflow.log_artifact(f.name, artifact_path="preprocessing_pipeline")
   ```
   This makes clear it's a preprocessing artifact, not a trained model.

3. **Update `load_pipeline()`** to load from the joblib artifact:
   ```python
   local = mlflow.artifacts.download_artifacts(
       run_id=run_id, artifact_path="preprocessing_pipeline"
   )
   return joblib.load(local)
   ```

### Option B: Migrate to MLflow Dataset Tracking (larger effort)

Use `mlflow.log_input()` on model runs to record which dataset they were trained on.
The dataset artifacts (parquets) would move out of MLflow and into a versioned
location in `data/processed/` (already there for the merged parquet), with MLflow
only tracking the provenance link, not storing the data itself.

This is the "correct" MLflow architecture but requires restructuring `build_pipeline()`
and `load_dataset()`, and the cached feature matrices are large enough that keeping
them in MLflow artifact storage is a reasonable pragmatic choice.

**Verdict:** Option A is the low-risk cleanup — two small changes to
`_save_dataset_to_mlflow` and `load_pipeline()`. Option B is a larger refactor
to defer until the project is more stable.

---

## Immediate Fix (no refactor needed)

The current `blend.select_candidates()` already avoids the conflation problem
by filtering on `params.model_class` — dataset runs don't have this param.
But it's fragile. Adding `tags.run_type = "dataset"` to `build_pipeline()` and
filtering `tags.run_type != 'dataset'` in `select_candidates()` would make
this explicit and robust.

```python
# In build_pipeline() → _save_dataset_to_mlflow():
all_tags["run_type"] = "dataset"

# In select_candidates():
filter_string = "metrics.mae > 0 AND tags.run_type != 'dataset'"
```

This is a two-line change and the safest immediate improvement.
