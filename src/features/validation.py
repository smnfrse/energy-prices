"""Leakage prevention validation for the daily feature pipeline.

Walks pipeline steps and verifies that each transformer respects data
availability constraints defined in AVAILABILITY_RULES.
"""

from fnmatch import fnmatch

from loguru import logger

from src.config.features import AVAILABILITY_RULES, ROLLING_FEATURE_SPECS
from src.features.ts_transforms import (
    DailyPivotTransformer,
    EWMATransformer,
    HourlyDailyAggregateTransformer,
    RollingStatsTransformer,
    SameHourLagTransformer,
)


def _match_rule(col_pattern, rules):
    """Find the AvailabilityRule that matches a column pattern.

    Returns the rule with the tightest constraint (most negative max_offset)
    if multiple rules match.
    """
    matches = []
    for rule in rules:
        # Check if the column pattern could match the rule pattern
        # Both may contain wildcards, so we check if they share a prefix
        if rule.pattern == col_pattern:
            matches.append(rule)
        elif "*" in col_pattern and "*" in rule.pattern:
            # Both are wildcards — check prefix overlap
            col_prefix = col_pattern.replace("*", "")
            rule_prefix = rule.pattern.replace("*", "")
            if col_prefix.startswith(rule_prefix) or rule_prefix.startswith(col_prefix):
                matches.append(rule)
        elif "*" in col_pattern:
            # Column pattern is wildcard, rule is exact
            col_prefix = col_pattern.replace("*", "")
            if rule.pattern.startswith(col_prefix):
                matches.append(rule)
        elif "*" in rule.pattern:
            # Rule is wildcard, column pattern is exact
            if fnmatch(col_pattern, rule.pattern):
                matches.append(rule)

    if not matches:
        return None
    # Return tightest constraint (most negative max_offset)
    return min(matches, key=lambda r: r.max_offset)


def validate_pipeline_leakage(pipeline):
    """Validate that a pipeline respects data availability constraints.

    Walks through pipeline steps and checks:
    1. RollingStatsTransformer: window end_day/end_hour vs AVAILABILITY_RULES
    2. DailyPivotTransformer: PivotSpec day_offset vs AVAILABILITY_RULES
    3. EWMATransformer: cutoff_day/cutoff_hour vs AVAILABILITY_RULES
    4. All columns with max_offset < 0 in AVAILABILITY_RULES are properly
       handled (appear in rolling specs or pivot specs, not just aggregated
       for day D)

    Args:
        pipeline: sklearn Pipeline to validate.

    Raises:
        ValueError: If any leakage violations are found.
    """
    violations = []
    rules = AVAILABILITY_RULES

    # Collect which column patterns are handled by rolling/pivot
    handled_patterns = set()

    for step_name, transformer in pipeline.steps:
        if isinstance(transformer, RollingStatsTransformer):
            _check_rolling(step_name, transformer, rules, violations, handled_patterns)
        elif isinstance(transformer, EWMATransformer):
            _check_ewma(step_name, transformer, rules, violations)
        elif isinstance(transformer, DailyPivotTransformer):
            _check_pivot(
                step_name,
                transformer,
                rules,
                violations,
                handled_patterns,
            )
        elif isinstance(transformer, SameHourLagTransformer):
            handled_patterns.add(transformer.column)
            # SameHourLag on target_price replaces daily pivot's
            # price_lag_* functionality
            if transformer.column == "target_price":
                handled_patterns.add("price_lag_*")
        elif isinstance(
            transformer,
            HourlyDailyAggregateTransformer,
        ):
            for col_pattern in transformer.columns:
                handled_patterns.add(col_pattern)
        # CreateCustomColumns wraps sub-transformers — skip (they don't
        # access future data, they just compute derived columns)

    # Check that all columns with max_offset < 0 are handled
    _check_unhandled(rules, handled_patterns, violations)

    if violations:
        msg = "Pipeline leakage violations found:\n" + "\n".join(f"  - {v}" for v in violations)
        logger.error(msg)
        raise ValueError(msg)

    logger.info("Pipeline leakage validation passed")


def _check_rolling(step_name, transformer, rules, violations, handled_patterns):
    """Check RollingStatsTransformer windows against availability rules."""
    for col_pattern in transformer.columns:
        rule = _match_rule(col_pattern, rules)
        if rule is None:
            continue

        for window in transformer.windows:
            # end_day must be <= max_offset (both are negative or zero)
            if window.end_day > rule.max_offset:
                violations.append(
                    f"[{step_name}] '{col_pattern}' window end_day={window.end_day} "
                    f"exceeds max_offset={rule.max_offset} (rule: {rule.pattern})"
                )

            # If end_day == max_offset and rule has cutoff_hour,
            # the window must also have end_hour <= cutoff_hour
            if (
                window.end_day == rule.max_offset
                and rule.cutoff_hour is not None
                and window.end_hour is not None
                and window.end_hour > rule.cutoff_hour
            ):
                violations.append(
                    f"[{step_name}] '{col_pattern}' window end_hour={window.end_hour} "
                    f"exceeds cutoff_hour={rule.cutoff_hour} (rule: {rule.pattern})"
                )

        if transformer.overwrite:
            handled_patterns.add(col_pattern)


def _check_ewma(step_name, transformer, rules, violations):
    """Check EWMATransformer cutoff against availability rules."""
    for col_pattern in transformer.columns:
        rule = _match_rule(col_pattern, rules)
        if rule is None:
            continue

        if transformer.cutoff_day > rule.max_offset:
            violations.append(
                f"[{step_name}] '{col_pattern}' cutoff_day={transformer.cutoff_day} "
                f"exceeds max_offset={rule.max_offset} (rule: {rule.pattern})"
            )

        if (
            transformer.cutoff_day == rule.max_offset
            and rule.cutoff_hour is not None
            and transformer.cutoff_hour is not None
            and transformer.cutoff_hour > rule.cutoff_hour
        ):
            violations.append(
                f"[{step_name}] '{col_pattern}' cutoff_hour={transformer.cutoff_hour} "
                f"exceeds cutoff_hour={rule.cutoff_hour} (rule: {rule.pattern})"
            )


def _check_pivot(step_name, transformer, rules, violations, handled_patterns):
    """Check DailyPivotTransformer pivot specs against availability rules."""
    for spec in transformer.pivot_specs:
        prefix = spec.prefix or spec.column
        handled_patterns.add(f"{prefix}_*")
        handled_patterns.add(spec.column)

        # Target output pivots (y_*) are what we predict — day_offset=0 is expected
        output_rule = _match_rule(f"{prefix}_*", rules)
        if output_rule is not None and output_rule.tier == "target":
            continue

        rule = _match_rule(spec.column, rules)
        if rule is None:
            continue

        if spec.day_offset > rule.max_offset:
            violations.append(
                f"[{step_name}] pivot '{spec.column}' day_offset={spec.day_offset} "
                f"exceeds max_offset={rule.max_offset} (rule: {rule.pattern})"
            )


def _check_unhandled(rules, handled_patterns, violations):
    """Check that columns needing lag treatment are not just aggregated for day D."""
    # Columns from ROLLING_FEATURE_SPECS are handled
    for spec in ROLLING_FEATURE_SPECS.values():
        if spec.get("overwrite"):
            for pattern in spec["columns"]:
                handled_patterns.add(pattern)

    # Derived columns that inherit lag from parent columns (created by transformers
    # from already-lagged inputs, e.g. pct_* from overwritten stromerzeugung_*)
    derived_patterns = {"pct_*"}
    handled_patterns.update(derived_patterns)

    for rule in rules:
        if rule.max_offset >= 0:
            continue  # No lag needed
        if rule.tier in ("target", "static"):
            continue

        # Check if this rule's pattern is covered by handled patterns
        covered = False
        for hp in handled_patterns:
            if rule.pattern == hp:
                covered = True
                break
            # Check prefix overlap
            if "*" in hp and "*" in rule.pattern:
                hp_prefix = hp.replace("*", "")
                rule_prefix = rule.pattern.replace("*", "")
                if rule_prefix.startswith(hp_prefix) or hp_prefix.startswith(rule_prefix):
                    covered = True
                    break
            elif "*" in hp:
                hp_prefix = hp.replace("*", "")
                if rule.pattern.startswith(hp_prefix):
                    covered = True
                    break
            elif "*" in rule.pattern:
                if fnmatch(hp, rule.pattern):
                    covered = True
                    break

        if not covered:
            violations.append(
                f"Column '{rule.pattern}' (max_offset={rule.max_offset}, "
                f"tier={rule.tier}) has no lag treatment in pipeline"
            )
