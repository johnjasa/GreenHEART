import shutil

import numpy as np
import pytest
import pyomo.environ as pyo

from h2integrate.core.h2integrate_model import H2IntegrateModel


EXAMPLE = "35_system_level_control/lp_arbitrage"

requires_glpk = pytest.mark.skipif(
    shutil.which("glpsol") is None,
    reason="GLPK executable 'glpsol' is not available in PATH",
)


def _make_controller(example_folder):
    """Set up the example and return its system-level controller."""
    model = H2IntegrateModel(example_folder / "solar_battery_arbitrage.yaml")
    model.setup()
    model.prob.final_setup()
    return model.prob.model.plant.system_level_controller


def _fill_window(
    lp,
    controller,
    sell_price,
    must_run=0.0,
    demand=0.0,
    marginal_cost=0.02,
    rated=50_000.0,
    soc_init=None,
    terminal_price=0.0,
    export_limit=None,
):
    """Populate the mutable parameters of a built LP window model."""
    sell_price = np.broadcast_to(np.asarray(sell_price, dtype=float), (controller.window_len,))
    must_run = np.broadcast_to(np.asarray(must_run, dtype=float), (controller.window_len,))
    demand = np.broadcast_to(np.asarray(demand, dtype=float), (controller.window_len,))
    if export_limit is None:
        export_limit = controller.export_limit
    export_limit = np.broadcast_to(np.asarray(export_limit, dtype=float), (controller.window_len,))

    for t in lp.T:
        lp.must_run[t] = float(must_run[t])
        lp.demand[t] = float(demand[t])
        lp.sell_price[t] = float(sell_price[t])
        lp.export_limit[t] = float(export_limit[t])
        for d in lp.D:
            lp.marginal_cost[d, t] = marginal_cost
    for d in lp.D:
        lp.rated[d] = rated
    for s in lp.S:
        params = controller.storage_params[s]
        lp.soc_init[s] = (
            params["min_soc_fraction"] * params["capacity"] if soc_init is None else soc_init
        )
        lp.terminal_price[s] = terminal_price


def _series(var, index, window_len):
    return np.array([pyo.value(var[index, t]) for t in range(window_len)])


@pytest.mark.unit
@pytest.mark.parametrize("example_folder,resource_example_folder", [(EXAMPLE, None)])
def test_lp_arbitrage_setup(subtests, temp_copy_of_example):
    """The controller reads its topology and storage sizing from configuration."""
    controller = _make_controller(temp_copy_of_example)

    with subtests.test("Export technology resolved from export_component"):
        assert controller.export_tech == "grid_sell"

    with subtests.test("Export limit is the interconnection size"):
        assert controller.export_limit == pytest.approx(100_000.0)

    with subtests.test("Window length from n_control_window_hours"):
        assert controller.window_len == 24

    with subtests.test("Technology classification"):
        assert controller.lp_storage_techs == ["battery"]
        assert controller.lp_dispatchable_techs == ["grid_buy"]
        assert controller.lp_must_run_techs == ["solar"]

    with subtests.test("Sell price input is added"):
        assert "grid_sell_sell_price" in controller._var_rel_names["input"]

    with subtests.test("Storage sizing read from tech config, not connected inputs"):
        params = controller.storage_params["battery"]
        assert params["capacity"] == pytest.approx(200_000.0)
        assert params["max_charge_rate"] == pytest.approx(50_000.0)
        assert params["max_discharge_rate"] == pytest.approx(50_000.0)
        # Round-trip efficiency of 0.88 is split evenly across charge and discharge.
        assert params["charge_efficiency"] == pytest.approx(np.sqrt(0.88))
        assert params["discharge_efficiency"] == pytest.approx(np.sqrt(0.88))


@requires_glpk
@pytest.mark.unit
@pytest.mark.parametrize("example_folder,resource_example_folder", [(EXAMPLE, None)])
def test_lp_arbitrage_buys_low_sells_high(subtests, temp_copy_of_example):
    """The LP charges during the cheap half of the window and discharges in the expensive half."""
    controller = _make_controller(temp_copy_of_example)
    lp = controller._build_lp_model()

    window_len = controller.window_len
    half = window_len // 2
    price = np.concatenate([np.full(half, 0.01), np.full(window_len - half, 0.10)])
    _fill_window(lp, controller, sell_price=price)

    controller._solve_window(lp, 0)

    charge = _series(lp.charge, "battery", window_len)
    discharge = _series(lp.discharge, "battery", window_len)
    soc = _series(lp.soc, "battery", window_len)

    with subtests.test("Charging happens only while the price is low"):
        assert charge[:half].sum() > 0.0
        assert charge[half:].sum() == pytest.approx(0.0, abs=1e-6)

    with subtests.test("Discharging happens only while the price is high"):
        assert discharge[half:].sum() > 0.0
        assert discharge[:half].sum() == pytest.approx(0.0, abs=1e-6)

    with subtests.test("Never charges and discharges simultaneously"):
        assert np.all((charge < 1e-6) | (discharge < 1e-6))

    with subtests.test("Rate limits respected"):
        params = controller.storage_params["battery"]
        assert charge.max() <= params["max_charge_rate"] + 1e-6
        assert discharge.max() <= params["max_discharge_rate"] + 1e-6

    with subtests.test("State of charge stays within its bounds"):
        params = controller.storage_params["battery"]
        assert soc.min() >= params["min_soc_fraction"] * params["capacity"] - 1e-6
        assert soc.max() <= params["max_soc_fraction"] * params["capacity"] + 1e-6


@requires_glpk
@pytest.mark.unit
@pytest.mark.parametrize("example_folder,resource_example_folder", [(EXAMPLE, None)])
def test_lp_arbitrage_respects_export_limit(subtests, temp_copy_of_example):
    """Production beyond the interconnection limit is curtailed rather than exported."""
    controller = _make_controller(temp_copy_of_example)
    lp = controller._build_lp_model()

    window_len = controller.window_len
    oversupply = 5.0 * controller.export_limit
    _fill_window(lp, controller, sell_price=0.05, must_run=oversupply)

    controller._solve_window(lp, 0)

    export = np.array([pyo.value(lp.export[t]) for t in range(window_len)])
    curtail = np.array([pyo.value(lp.curtail[t]) for t in range(window_len)])

    with subtests.test("Export never exceeds the interconnection size"):
        assert export.max() <= controller.export_limit + 1e-6

    with subtests.test("Export is saturated"):
        assert export.min() == pytest.approx(controller.export_limit, rel=1e-6)

    with subtests.test("Surplus is curtailed"):
        assert curtail.min() > 0.0


@requires_glpk
@pytest.mark.unit
@pytest.mark.parametrize("example_folder,resource_example_folder", [(EXAMPLE, None)])
def test_lp_arbitrage_charge_limited_by_availability(temp_copy_of_example):
    """Storage cannot charge from commodity that is not physically on the bus.

    This mirrors the ``charge_available`` clip inside the storage performance
    model. With no must-run production and no import capacity there is nothing
    to charge from, so the commanded charge must be zero even though the price
    spread would otherwise make charging attractive.
    """
    controller = _make_controller(temp_copy_of_example)
    lp = controller._build_lp_model()

    window_len = controller.window_len
    half = window_len // 2
    price = np.concatenate([np.full(half, 0.01), np.full(window_len - half, 0.10)])
    _fill_window(lp, controller, sell_price=price, must_run=0.0, rated=0.0)

    controller._solve_window(lp, 0)

    charge = _series(lp.charge, "battery", window_len)
    assert charge.sum() == pytest.approx(0.0, abs=1e-6)


@requires_glpk
@pytest.mark.unit
@pytest.mark.parametrize("example_folder,resource_example_folder", [(EXAMPLE, None)])
def test_lp_arbitrage_time_varying_export_limit(subtests, temp_copy_of_example):
    """A per-timestep export ceiling is honored, and is re-read on every solve.

    The window model is built once and re-solved with updated mutable
    parameters, so this also guards the assumption that Pyomo re-evaluates a
    mutable ``Param`` used in a variable bound rather than freezing it at
    construction time.
    """
    controller = _make_controller(temp_copy_of_example)
    lp = controller._build_lp_model()

    window_len = controller.window_len
    half = window_len // 2
    oversupply = 5.0 * controller.export_limit
    ceiling = np.concatenate([np.zeros(half), np.full(window_len - half, 20_000.0)])
    _fill_window(lp, controller, sell_price=0.05, must_run=oversupply, export_limit=ceiling)

    controller._solve_window(lp, 0)
    export = np.array([pyo.value(lp.export[t]) for t in range(window_len)])

    with subtests.test("Export tracks the time-varying ceiling"):
        assert np.allclose(export, ceiling, atol=1e-6)

    with subtests.test("Blocked production is curtailed"):
        curtail = np.array([pyo.value(lp.curtail[t]) for t in range(window_len)])
        assert curtail[:half].min() > 0.0

    # Re-solve the same model object with a different ceiling.
    relaxed = np.full(window_len, 60_000.0)
    _fill_window(lp, controller, sell_price=0.05, must_run=oversupply, export_limit=relaxed)
    controller._solve_window(lp, 0)
    export = np.array([pyo.value(lp.export[t]) for t in range(window_len)])

    with subtests.test("A rebuilt ceiling takes effect without rebuilding the model"):
        assert np.allclose(export, relaxed, atol=1e-6)


@pytest.mark.unit
@pytest.mark.parametrize("example_folder,resource_example_folder", [(EXAMPLE, None)])
def test_lp_arbitrage_headroom_components(subtests, temp_copy_of_example):
    """Headroom keys add controller inputs and resolve to the named component."""
    controller = _make_controller(temp_copy_of_example)

    with subtests.test("Headroom inputs are added"):
        assert "existing_load_demand_unmet_demand" in controller._var_rel_names["input"]
        assert "existing_load_demand_surplus" in controller._var_rel_names["input"]

    with subtests.test("Components are recorded on the controller"):
        assert controller.export_limit_component == "existing_load_demand"
        assert controller.surplus_source_component == "existing_load_demand"

    with subtests.test("The uncontrolled existing plant is not dispatched"):
        assert "existing_solar" not in controller.lp_must_run_techs
        assert "existing_solar" not in controller.lp_dispatchable_techs


@pytest.mark.unit
@pytest.mark.parametrize("example_folder,resource_example_folder", [(EXAMPLE, None)])
def test_lp_arbitrage_rejects_unknown_headroom_component(temp_copy_of_example):
    """A headroom key naming a missing technology is reported as a config error."""
    config_path = temp_copy_of_example / "plant_config.yaml"
    text = config_path.read_text()
    config_path.write_text(
        text.replace(
            "    export_limit_component: existing_load_demand\n",
            "    export_limit_component: not_a_tech\n",
        )
    )

    # The controller is wired up during construction, so this fails before setup().
    with pytest.raises(ValueError, match="not a configured technology"):
        H2IntegrateModel(temp_copy_of_example / "solar_battery_arbitrage.yaml")


@requires_glpk
@pytest.mark.unit
@pytest.mark.parametrize("example_folder,resource_example_folder", [(EXAMPLE, None)])
def test_lp_arbitrage_requires_export_component(temp_copy_of_example):
    """A missing ``export_component`` is reported as a configuration error."""
    config_path = temp_copy_of_example / "plant_config.yaml"
    text = config_path.read_text()
    config_path.write_text(text.replace("  export_component: grid_sell\n", ""))

    model = H2IntegrateModel(temp_copy_of_example / "solar_battery_arbitrage.yaml")
    with pytest.raises(ValueError, match="requires an export technology"):
        model.setup()


@requires_glpk
@pytest.mark.integration
@pytest.mark.parametrize("example_folder,resource_example_folder", [(EXAMPLE, None)])
def test_lp_arbitrage_example(subtests, temp_copy_of_example):
    """The full example dispatches a schedule the storage model can follow exactly."""
    model = H2IntegrateModel(temp_copy_of_example / "solar_battery_arbitrage.yaml")
    model.setup()

    # A deterministic diurnal price: cheap overnight, expensive in the evening.
    hour_of_day = np.arange(8760) % 24
    price = 0.03 + 0.025 * np.sin((hour_of_day - 9) * np.pi / 12)
    model.prob.set_val("grid_sell.electricity_sell_price", price, units="USD/(kW*h)")
    model.prob.set_val("grid_buy.electricity_buy_price", price + 0.004, units="USD/(kW*h)")

    model.run()

    get = model.prob.get_val
    commanded = get("system_level_controller.battery_electricity_set_point", units="kW")
    actual = get("battery.electricity_out", units="kW")
    charge = -get("battery.storage_electricity_charge", units="kW")
    discharge = get("battery.storage_electricity_discharge", units="kW")
    soc = get("battery.SOC", units="percent")
    solar = get("solar.electricity_out", units="kW")
    imported = get("grid_buy.electricity_out", units="kW")
    exported = get("grid_sell.electricity_sold", units="kW")
    curtailed = get("grid_sell.electricity_excess", units="kW")
    spill = get("existing_load_demand.unused_electricity_out", units="kW")
    headroom = get("existing_load_demand.unmet_electricity_demand_out", units="kW")
    unmet = get("electrical_load_demand.unmet_electricity_demand_out", units="kW")

    with subtests.test("Storage follows the commanded schedule exactly"):
        # Nothing is clipped by the storage model's charge-availability limit.
        assert np.allclose(commanded, actual, rtol=1e-6, atol=1e-6)

    with subtests.test("State of charge stays within its bounds"):
        assert soc.min() >= 10.0 - 1e-6
        assert soc.max() <= 100.0 + 1e-6

    with subtests.test("Export respects the interconnection limit"):
        assert exported.max() <= 100_000.0 + 1e-6

    with subtests.test("Export never exceeds the existing plant's unmet demand"):
        # The controller plans against this ceiling and the grid model enforces it.
        assert exported.max() <= headroom.max() + 1e-6
        assert np.all(exported <= np.minimum(100_000.0, headroom) + 1e-6)

    with subtests.test("No unmet demand"):
        assert unmet.sum() == pytest.approx(0.0, abs=1e-6)

    with subtests.test("Commodity balance closes"):
        supply = solar + imported + spill + discharge - charge
        assert np.allclose(supply, exported + curtailed, rtol=1e-6, atol=1e-6)

    with subtests.test("Round-trip efficiency is applied"):
        # Not exactly the round-trip efficiency because the battery does not end
        # the year at its starting state of charge.
        assert discharge.sum() / charge.sum() == pytest.approx(0.88, rel=5e-3)

    with subtests.test("Charges cheaper than it discharges"):
        assert np.average(price, weights=charge) < np.average(price, weights=discharge)
