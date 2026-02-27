"""Modeling configuration constants."""

# Experiment names
EXPERIMENTS = {
    "baselines": "0-baselines",
    "linear": "1-linear",
}

# Test split
TEST_SIZE = 0.2

# Peak hours (8am-8pm, 0-indexed)
PEAK_HOURS = list(range(8, 20))

# --- Blend / ensemble constants ---
BLEND_HOLDOUT_DAYS = 90
BLEND_CANDIDATES_TOP = 3
BLEND_CANDIDATES_RANDOM = 2
BLEND_CANDIDATES_RANDOM_POOL = 10
BLEND_FINAL_PER_CATEGORY = 2
BLEND_DEGRADATION_THRESHOLD = 0.20

# Classification by params.model_class (auto-logged by train_and_log)
BLEND_CATEGORY_MATCHERS = {
    "linear": [
        "Ridge",
        "Lasso",
        "ElasticNet",
        "LinearRegression",
        "HuberRegressor",
        "MultiOutputRegressor",
    ],
    "lgbm": ["LGBMRegressor"],
    "xgboost": ["XGBRegressor"],
    "catboost": ["CatBoostRegressor"],
}

# Always include these MLflow run IDs regardless of ranking
BLEND_FORCE_RUN_IDS: list[str] = []
