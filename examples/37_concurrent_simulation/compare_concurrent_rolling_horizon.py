"""Compare existing concurrent execution with rolling-horizon execution.

Both cases use the same validated Example 37 configuration. The only rolling-horizon
change is ``enabled=False`` for the existing ``n_steps_per_compute`` implementation and
``enabled=True`` for the rolling-horizon implementation. Results are compared using both
shared OpenMDAO output paths and explicit mappings for outputs moved behind
``RollingHorizonComponent``.
"""

import os
import argparse
from copy import deepcopy
from time import perf_counter
from pathlib import Path
from statistics import mean, stdev
from dataclasses import dataclass

import yaml
import numpy as np


os.environ.setdefault("OPENMDAO_REPORTS", "none")

from h2integrate import H2IntegrateModel, load_tech_yaml, load_plant_yaml, load_driver_yaml


@dataclass(frozen=True)
class Timings:
    """Wall-clock timings for one model execution."""

    construction: float
    setup: float
    execution: float

    @property
    def total(self):
        """Return total construction, setup, and execution time."""
        return self.construction + self.setup + self.execution


@dataclass(frozen=True)
class Comparison:
    """Numerical difference statistics for one result."""

    name: str
    exact: bool
    close: bool
    max_abs: float
    max_rel: float
    rmse: float
    size: int


SEMANTIC_OUTPUTS = {
    "solar electricity": ("solar.electricity_out", "solar.electricity_out"),
    "battery electricity": (
        "battery.electricity_out",
        "rolling_horizon.battery_electricity_out",
    ),
    "battery final SOC (%)": (
        "battery.SOC",
        "rolling_horizon.final_battery_soc",
    ),
    "battery/solar combined electricity": (
        "bat_combiner.electricity_out",
        "rolling_horizon.bat_combiner_electricity_out",
    ),
    "grid electricity bought": (
        "grid_buy.electricity_out",
        "rolling_horizon.grid_buy_electricity_out",
    ),
    "grid electricity sold": (
        "grid_sell.electricity_sold",
        "rolling_horizon.grid_sell_electricity_sold",
    ),
    "final combined electricity": ("combiner.electricity_out", "combiner.electricity_out"),
    "grid buy variable cost": ("grid_buy.VarOpEx", "grid_buy.VarOpEx"),
    "grid sell variable cost": ("grid_sell.VarOpEx", "grid_sell.VarOpEx"),
    "LCOE": (
        "plant.finance_subgroup_renewables.electricity_finance_profast_model.LCOE",
        "plant.finance_subgroup_renewables.electricity_finance_profast_model.LCOE",
    ),
}


def load_example_config(config_path):
    """Load and validate all Example 37 configuration files."""
    config_path = config_path.resolve()
    config_root = config_path.parent
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    config["driver_config"] = load_driver_yaml(config_root / config["driver_config"])
    config["technology_config"] = load_tech_yaml(config_root / config["technology_config"])
    config["plant_config"] = load_plant_yaml(config_root / config["plant_config"])
    return config


def run_case(base_config, rolling_horizon_enabled, rolling_control_interval=None):
    """Construct, set up, and execute one comparison case."""
    config = deepcopy(base_config)
    rolling_config = config["plant_config"]["rolling_horizon"]
    rolling_config["enabled"] = rolling_horizon_enabled
    if rolling_horizon_enabled and rolling_control_interval is not None:
        rolling_config["control_interval"] = rolling_control_interval

    start = perf_counter()
    model = H2IntegrateModel(config)
    constructed = perf_counter()
    model.setup()
    setup = perf_counter()
    model.run()
    finished = perf_counter()

    return model, Timings(
        construction=constructed - start,
        setup=setup - constructed,
        execution=finished - setup,
    )


def _numeric_outputs(model):
    """Return numeric OpenMDAO outputs keyed by absolute variable path."""
    outputs = {}
    for name, metadata in model.model.list_outputs(out_stream=None):
        value = np.asarray(metadata["val"])
        if np.issubdtype(value.dtype, np.number):
            outputs[name] = value
    return outputs


def _compare(name, reference, candidate, rtol, atol):
    """Calculate robust elementwise difference statistics."""
    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    if reference.shape != candidate.shape:
        return None

    difference = np.abs(candidate - reference)
    scale = np.maximum(np.maximum(np.abs(reference), np.abs(candidate)), atol)
    relative = np.divide(difference, scale, out=np.zeros_like(difference), where=scale > 0)
    return Comparison(
        name=name,
        exact=np.array_equal(reference, candidate, equal_nan=True),
        close=np.allclose(reference, candidate, rtol=rtol, atol=atol, equal_nan=True),
        max_abs=float(np.nanmax(difference)) if difference.size else 0.0,
        max_rel=float(np.nanmax(relative)) if relative.size else 0.0,
        rmse=float(np.sqrt(np.nanmean(np.square(difference)))) if difference.size else 0.0,
        size=reference.size,
    )


def compare_semantic_outputs(concurrent_model, rolling_model, rtol, atol):
    """Compare user-facing results whose OpenMDAO paths may differ by mode."""
    comparisons = []
    missing = []
    for name, (concurrent_path, rolling_path) in SEMANTIC_OUTPUTS.items():
        try:
            concurrent_value = concurrent_model.prob.get_val(concurrent_path)
            rolling_value = rolling_model.prob.get_val(rolling_path)
        except KeyError:
            missing.append(name)
            continue

        if name == "battery final SOC (%)":
            concurrent_value = np.asarray(concurrent_value)[-1:]
            rolling_value = np.asarray(rolling_value) * 100.0

        comparison = _compare(name, concurrent_value, rolling_value, rtol, atol)
        if comparison is None:
            missing.append(f"{name} (shape mismatch)")
        else:
            comparisons.append(comparison)
    return comparisons, missing


def compare_shared_outputs(concurrent_model, rolling_model, rtol, atol):
    """Compare outputs that retain the same absolute OpenMDAO path in both modes."""
    concurrent_outputs = _numeric_outputs(concurrent_model)
    rolling_outputs = _numeric_outputs(rolling_model)
    shared_names = sorted(concurrent_outputs.keys() & rolling_outputs.keys())

    comparisons = []
    shape_mismatches = []
    for name in shared_names:
        comparison = _compare(
            name,
            concurrent_outputs[name],
            rolling_outputs[name],
            rtol,
            atol,
        )
        if comparison is None:
            shape_mismatches.append(name)
        else:
            comparisons.append(comparison)
    return comparisons, shape_mismatches


def print_timings(concurrent_runs, rolling_runs):
    """Print timing and speed-ratio results."""
    print(
        f"\nTiming across {len(concurrent_runs)} run(s), "
        "mean +/- sample standard deviation (seconds)"
    )
    print(
        f"{'phase':<16}{'existing concurrent':>26}{'rolling horizon':>24}"
        f"{'concurrent / rolling':>23}"
    )
    for field in ("construction", "setup", "execution", "total"):
        concurrent_values = [getattr(timing, field) for timing in concurrent_runs]
        rolling_values = [getattr(timing, field) for timing in rolling_runs]
        concurrent_time = mean(concurrent_values)
        rolling_time = mean(rolling_values)
        concurrent_std = stdev(concurrent_values) if len(concurrent_values) > 1 else 0.0
        rolling_std = stdev(rolling_values) if len(rolling_values) > 1 else 0.0
        ratio = concurrent_time / rolling_time if rolling_time else float("inf")
        concurrent_text = f"{concurrent_time:.4f} +/- {concurrent_std:.4f}"
        rolling_text = f"{rolling_time:.4f} +/- {rolling_std:.4f}"
        print(f"{field:<16}{concurrent_text:>26}{rolling_text:>24}{ratio:>23.3f}x")


def print_comparisons(title, comparisons, top):
    """Print summary and the largest numerical differences."""
    exact = sum(result.exact for result in comparisons)
    close = sum(result.close for result in comparisons)
    print(f"\n{title}: {len(comparisons)} compared, {exact} exact, {close} within tolerance")
    print(f"{'output':<58}{'exact':>8}{'close':>8}{'max abs':>14}{'max rel':>14}{'RMSE':>14}")
    ranked = sorted(comparisons, key=lambda result: (result.max_rel, result.max_abs), reverse=True)
    for result in ranked[:top]:
        print(
            f"{result.name:<58}{result.exact!s:>8}{result.close!s:>8}"
            f"{result.max_abs:>14.6g}{result.max_rel:>14.6g}{result.rmse:>14.6g}"
        )


def parse_args():
    """Parse command-line options."""
    default_config = Path(__file__).with_name("solar_battery_grid.yaml")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument(
        "--rolling-control-interval",
        type=int,
        default=None,
        help="Override rolling_horizon.control_interval for the enabled case.",
    )
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument("--top", type=int, default=20, help="Maximum rows per result table.")
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of timing repetitions. Run order alternates to reduce ordering bias.",
    )
    return parser.parse_args()


def main():
    """Run both methods and report accuracy and computational efficiency."""
    args = parse_args()
    base_config = load_example_config(args.config)
    simulation = base_config["plant_config"]["plant"]["simulation"]
    rolling_config = base_config["plant_config"]["rolling_horizon"]
    control_interval = args.rolling_control_interval or rolling_config["control_interval"]

    print(f"Configuration: {args.config.resolve()}")
    print(f"Full-simulation timesteps: {simulation['n_timesteps']}")
    print(f"Existing concurrent n_steps_per_compute: {simulation.get('n_steps_per_compute')}")
    print(f"Rolling prediction horizon: {rolling_config['prediction_horizon']}")
    print(f"Rolling control interval: {control_interval}")

    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")

    concurrent_timings = []
    rolling_timings = []
    concurrent_model = None
    rolling_model = None
    for repetition in range(args.repeats):
        modes = (False, True) if repetition % 2 == 0 else (True, False)
        for rolling_enabled in modes:
            model, timings = run_case(
                base_config,
                rolling_enabled,
                rolling_control_interval=args.rolling_control_interval,
            )
            if rolling_enabled:
                rolling_model = model
                rolling_timings.append(timings)
            else:
                concurrent_model = model
                concurrent_timings.append(timings)

    semantic, missing_semantic = compare_semantic_outputs(
        concurrent_model, rolling_model, args.rtol, args.atol
    )
    shared, shape_mismatches = compare_shared_outputs(
        concurrent_model, rolling_model, args.rtol, args.atol
    )

    print_timings(concurrent_timings, rolling_timings)
    print_comparisons("Mapped semantic outputs", semantic, args.top)
    print_comparisons("Shared OpenMDAO outputs", shared, args.top)

    if missing_semantic:
        print(f"\nUnavailable mapped outputs: {', '.join(missing_semantic)}")
    if shape_mismatches:
        print(f"Shape-mismatched shared outputs: {len(shape_mismatches)}")

    all_close = all(result.close for result in semantic) and all(result.close for result in shared)
    print(f"\nOverall within rtol={args.rtol:g}, atol={args.atol:g}: {all_close}")


if __name__ == "__main__":
    main()
