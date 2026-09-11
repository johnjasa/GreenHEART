from pathlib import Path

import numpy as np
import pytest
import openmdao.api as om

from h2integrate import H2IntegrateModel
from h2integrate.converters.grid.grid import GridCostModel
from h2integrate.core.rolling_horizon import (
    StorageSOCStateAdapter,
    RollingHorizonComponent,
    RollingHorizonController,
    RollingHorizonInputAlias,
    RollingHorizonOutputAlias,
    create_rolling_horizon_plan,
    create_boundary_input_aliases,
    create_boundary_output_aliases,
    create_h2integrate_window_problem,
    create_window_model_configuration,
)
from h2integrate.core.inputs.validation import load_tech_yaml, load_plant_yaml, load_driver_yaml


@pytest.fixture
def rolling_horizon_configs():
    plant_config = {
        "plant": {"simulation": {"n_timesteps": 8760}},
        "rolling_horizon": {
            "enabled": False,
            "prediction_horizon": 24,
            "control_interval": 1,
            "steppable_technologies": ["battery", "demand", "grid"],
            "initial_states": {"battery": {"soc": 0.25}},
        },
        "technology_interconnections": [
            ["solar", "battery", "electricity"],
            ["battery", "demand", "electricity"],
            ["demand", "grid", ["unmet_electricity_demand_out", "electricity_set_point"]],
        ],
        "system_level_control": {
            "control_parameters": {"cost_per_tech": {"grid": "buy_price", "battery": "VarOpEx"}}
        },
    }
    technology_config = {
        "technologies": {
            "solar": {
                "performance_model": {"model": "SolarPerformance"},
                "cost_model": {"model": "SolarCost"},
            },
            "battery": {
                "control_strategy": {"model": "BatteryControl"},
                "performance_model": {"model": "BatteryPerformance"},
                "cost_model": {"model": "BatteryCost"},
            },
            "demand": {"performance_model": {"model": "DemandPerformance"}},
            "grid": {
                "performance_model": {"model": "GridPerformance"},
                "cost_model": {"model": "GridCost"},
            },
        }
    }
    return plant_config, technology_config


@pytest.mark.unit
def test_rolling_horizon_plan_preserves_technology_ownership(rolling_horizon_configs):
    plan = create_rolling_horizon_plan(*rolling_horizon_configs)

    assert plan.steppable_technologies == ("battery", "demand", "grid")
    assert plan.initial_states == {"battery": {"soc": 0.25}}
    assert plan.window_model_roles["solar"] == ()
    assert plan.full_sim_model_roles["solar"] == ("performance_model", "cost_model")
    assert plan.window_model_roles["battery"] == (
        "control_strategy",
        "performance_model",
    )
    assert plan.full_sim_model_roles["battery"] == ("cost_model",)


@pytest.mark.unit
def test_initial_state_must_belong_to_steppable_technology(rolling_horizon_configs):
    plant_config, technology_config = rolling_horizon_configs
    plant_config["rolling_horizon"]["initial_states"] = {"solar": {"soc": 0.25}}

    with pytest.raises(ValueError, match="initial states must belong"):
        create_rolling_horizon_plan(plant_config, technology_config)


@pytest.mark.unit
def test_combined_model_is_rejected(rolling_horizon_configs):
    plant_config, technology_config = rolling_horizon_configs
    technology_config["technologies"]["battery"]["performance_model"]["model"] = (
        "WOMBATElectrolyzerModel"
    )

    with pytest.raises(ValueError, match="unsupported combined model"):
        create_rolling_horizon_plan(plant_config, technology_config)


@pytest.mark.unit
def test_window_configuration_filters_annual_roles_and_connections(rolling_horizon_configs):
    plant_config, technology_config = rolling_horizon_configs
    plan = create_rolling_horizon_plan(plant_config, technology_config)

    window_config = create_window_model_configuration(
        {"driver": {"optimization": True}, "recorder": {}},
        plant_config,
        technology_config,
        plan,
    )

    assert window_config["driver_config"] == {}
    assert window_config["plant_config"]["plant"]["simulation"]["n_timesteps"] == 24
    assert "rolling_horizon" not in window_config["plant_config"]
    assert "solar" not in window_config["technology_config"]["technologies"]
    assert "cost_model" not in window_config["technology_config"]["technologies"]["battery"]
    assert window_config["plant_config"]["technology_interconnections"] == [
        ["battery", "demand", "electricity"],
        ["demand", "grid", ["unmet_electricity_demand_out", "electricity_set_point"]],
    ]


@pytest.mark.unit
def test_boundary_aliases_support_one_source_to_multiple_window_inputs():
    plant_config = {
        "plant": {"plant_life": 30, "simulation": {"n_timesteps": 8760}},
        "rolling_horizon": {
            "enabled": False,
            "prediction_horizon": 24,
            "control_interval": 1,
            "steppable_technologies": ["battery", "bat_combiner"],
        },
        "technology_interconnections": [
            ["solar", "battery", "electricity", "cable"],
            ["solar", "bat_combiner", "electricity", "cable"],
            ["battery", "bat_combiner", "electricity", "cable"],
        ],
    }
    technology_config = {
        "technologies": {
            "solar": {"performance_model": {"model": "Solar"}},
            "battery": {"performance_model": {"model": "Battery"}},
            "bat_combiner": {"performance_model": {"model": "Combiner"}},
        }
    }
    plan = create_rolling_horizon_plan(plant_config, technology_config)

    aliases = {alias.name: alias for alias in create_boundary_input_aliases(plan, plant_config)}

    assert aliases["solar_electricity_out"].targets == (
        "battery.electricity_in",
        "bat_combiner.electricity_in1",
    )
    assert aliases["solar_rated_electricity_production"].targets == (
        "bat_combiner.rated_electricity_production1",
    )
    assert not aliases["solar_rated_electricity_production"].windowed
    assert aliases["solar_capacity_factor"].shape == 30


@pytest.mark.unit
def test_boundary_output_aliases_feed_full_sim_topology_and_costs():
    example_dir = Path(__file__).parents[3] / "examples" / "37_concurrent_simulation"
    plant_config = load_plant_yaml(example_dir / "plant_config.yaml")
    technology_config = load_tech_yaml(example_dir / "tech_config.yaml")
    plan = create_rolling_horizon_plan(plant_config, technology_config)

    aliases = {
        alias.name: alias
        for alias in create_boundary_output_aliases(plan, plant_config, technology_config)
    }

    assert aliases["bat_combiner_electricity_out"].targets == ("combiner.electricity_in1",)
    assert aliases["grid_buy_electricity_out"].targets == (
        "combiner.electricity_in2",
        "grid_buy.electricity_out",
    )
    assert aliases["grid_sell_electricity_sold"].targets == ("grid_sell.electricity_sold",)
    assert aliases["battery_electricity_out"].targets == ()
    assert aliases["battery_SOC"].targets == ()
    assert aliases["battery_replacement_schedule"].shape == 30
    assert aliases["battery_replacement_schedule"].targets == (
        "finance_subgroup_renewables.replacement_schedule_battery",
    )


class _FakeWindowProblem:
    def __init__(self):
        self.values = {}
        self.setup_count = 0
        self.run_count = 0

    def setup(self):
        self.setup_count += 1

    def set_val(self, name, value):
        self.values[name] = np.asarray(value)

    def get_val(self, name):
        return self.values[name]

    def run_model(self):
        self.run_count += 1
        self.values["window.output"] = self.values["window.forecast"] * 2


class _FakeStorageComponent:
    def __init__(self):
        self.soc_init = 0.1
        self._soc_timeseries = np.array([0.1, 0.2, 0.3])


class _FakeModel:
    def __init__(self, component):
        self.component = component

    def _get_subsystem(self, path):
        return self.component if path == "plant.battery.storage" else None


@pytest.mark.unit
def test_storage_soc_adapter_uses_committed_state_without_model_changes():
    component = _FakeStorageComponent()
    problem = type("Problem", (), {"model": _FakeModel(component)})()
    adapter = StorageSOCStateAdapter("battery", ("plant.battery.storage",))

    adapter.set_initial_state(problem, 0.4)
    np.testing.assert_array_equal(component._soc_timeseries, [0.4, 0.4, 0.4])
    component._soc_timeseries[:] = [0.45, 0.5, 0.55]

    assert adapter.capture_final_state(problem, committed_steps=2) == pytest.approx(0.5)


@pytest.mark.unit
def test_controller_reuses_problem_and_commits_only_control_interval(rolling_horizon_configs):
    plant_config, technology_config = rolling_horizon_configs
    plant_config["plant"]["simulation"]["n_timesteps"] = 5
    plant_config["rolling_horizon"]["prediction_horizon"] = 3
    plant_config["rolling_horizon"]["control_interval"] = 2
    plan = create_rolling_horizon_plan(plant_config, technology_config)
    fake_problem = _FakeWindowProblem()
    controller = RollingHorizonController(
        plan,
        object(),
        problem_factory=lambda _: fake_problem,
        input_mappings={"forecast": "window.forecast"},
        output_mappings={"dispatch": "window.output"},
    )

    result = controller.run({"forecast": np.arange(5, dtype=float)})

    assert fake_problem.setup_count == 1
    assert fake_problem.run_count == 3
    outputs, _ = result
    np.testing.assert_array_equal(outputs["dispatch"], [0, 2, 4, 6, 8])
    np.testing.assert_array_equal(fake_problem.values["window.forecast"], [4, 4, 4])

    repeated_result = controller.run({"forecast": np.arange(5, dtype=float)})
    assert fake_problem.setup_count == 1
    repeated_outputs, _ = repeated_result
    np.testing.assert_array_equal(repeated_outputs["dispatch"], outputs["dispatch"])


@pytest.mark.unit
def test_rolling_horizon_component_exposes_aliases_and_runs_controller(
    rolling_horizon_configs,
):
    plant_config, technology_config = rolling_horizon_configs
    plant_config["plant"]["simulation"]["n_timesteps"] = 5
    plant_config["rolling_horizon"]["prediction_horizon"] = 3
    plant_config["rolling_horizon"]["control_interval"] = 2
    plant_config["rolling_horizon"]["initial_states"] = {}
    plan = create_rolling_horizon_plan(plant_config, technology_config)
    fake_problem = _FakeWindowProblem()
    controller = RollingHorizonController(
        plan,
        object(),
        problem_factory=lambda _: fake_problem,
    )
    component = RollingHorizonComponent(
        controller=controller,
        input_aliases=(
            RollingHorizonInputAlias(
                "forecast", ("window.forecast", "window.duplicate_forecast"), units="kW"
            ),
        ),
        output_aliases=(RollingHorizonOutputAlias("dispatch", "window.output", units="kW"),),
    )
    problem = om.Problem()
    problem.model.add_subsystem("rolling_horizon", component)
    problem.setup()
    problem.set_val("rolling_horizon.forecast", np.arange(5, dtype=float), units="kW")

    problem.run_model()

    np.testing.assert_array_equal(problem.get_val("rolling_horizon.dispatch"), [0, 2, 4, 6, 8])
    np.testing.assert_array_equal(fake_problem.values["window.duplicate_forecast"], [4, 4, 4])


@pytest.mark.integration
def test_annual_grid_cost_runs_from_assembled_rolling_horizon_output(
    rolling_horizon_configs,
):
    plant_config, technology_config = rolling_horizon_configs
    plant_config["plant"] = {
        "plant_life": 1,
        "simulation": {"n_timesteps": 5, "dt": 3600},
    }
    plant_config["rolling_horizon"]["prediction_horizon"] = 3
    plant_config["rolling_horizon"]["control_interval"] = 2
    plant_config["rolling_horizon"]["initial_states"] = {}
    plan = create_rolling_horizon_plan(plant_config, technology_config)
    fake_problem = _FakeWindowProblem()
    controller = RollingHorizonController(
        plan,
        object(),
        problem_factory=lambda _: fake_problem,
    )
    rolling_component = RollingHorizonComponent(
        controller=controller,
        input_aliases=(RollingHorizonInputAlias("forecast", ("window.forecast",), "kW"),),
        output_aliases=(RollingHorizonOutputAlias("electricity_out", "window.output", "kW"),),
    )
    grid_tech_config = {
        "model_inputs": {
            "shared_parameters": {"interconnection_size": 1000.0},
            "cost_parameters": {
                "cost_year": 2022,
                "interconnection_capex_per_kw": 0.0,
                "interconnection_opex_per_kw": 0.0,
                "fixed_interconnection_cost": 0.0,
                "electricity_buy_price": 0.1,
                "buy_price_mode": "constant",
            },
        }
    }
    problem = om.Problem()
    problem.model.add_subsystem("rolling_horizon", rolling_component)
    problem.model.add_subsystem(
        "grid_cost",
        GridCostModel(
            driver_config={},
            plant_config=plant_config,
            tech_config=grid_tech_config,
        ),
    )
    problem.model.connect("rolling_horizon.electricity_out", "grid_cost.electricity_out")
    problem.setup()
    problem.set_val("rolling_horizon.forecast", np.arange(5, dtype=float), units="kW")

    problem.run_model()

    np.testing.assert_allclose(problem.get_val("grid_cost.VarOpEx"), [2.0])


@pytest.mark.integration
def test_example_37_window_builds_and_runs_without_model_changes():
    example_dir = Path(__file__).parents[3] / "examples" / "37_concurrent_simulation"
    driver_config = load_driver_yaml(example_dir / "driver_config.yaml")
    plant_config = load_plant_yaml(example_dir / "plant_config.yaml")
    technology_config = load_tech_yaml(example_dir / "tech_config.yaml")
    plan = create_rolling_horizon_plan(plant_config, technology_config)
    configuration = create_window_model_configuration(
        driver_config, plant_config, technology_config, plan
    )
    problem = create_h2integrate_window_problem(configuration)
    problem.setup()
    adapter = StorageSOCStateAdapter(
        "battery",
        (
            "plant.battery.DemandOpenLoopStorageController",
            "plant.battery.StoragePerformanceModel",
        ),
    )

    adapter.set_initial_state(problem, 0.25)
    problem.set_val("battery.electricity_in", np.full(plan.prediction_horizon, 100000.0))
    problem.set_val("bat_combiner.electricity_in1", np.full(plan.prediction_horizon, 100000.0))
    problem.run_model()

    assert problem.get_val("grid_buy.electricity_out").shape == (plan.prediction_horizon,)
    assert adapter.capture_final_state(problem, 1) == pytest.approx(0.25)


@pytest.mark.integration
def test_example_37_runs_with_rolling_horizon_enabled():
    example_dir = Path(__file__).parents[3] / "examples" / "37_concurrent_simulation"
    plant_config = load_plant_yaml(example_dir / "plant_config.yaml")
    plant_config["rolling_horizon"]["enabled"] = True
    plant_config["rolling_horizon"]["control_interval"] = 24
    config = {
        "name": "rolling_horizon_example_37",
        "driver_config": load_driver_yaml(example_dir / "driver_config.yaml"),
        "technology_config": load_tech_yaml(example_dir / "tech_config.yaml"),
        "plant_config": plant_config,
    }

    model = H2IntegrateModel(config)
    model.run()

    assert model.rolling_horizon_enabled
    assert model.prob.get_val("grid_buy.VarOpEx").shape == (30,)
    assert np.sum(model.prob.get_val("rolling_horizon.grid_buy_electricity_out")) > 0.0
    assert model.prob.get_val("rolling_horizon.final_battery_soc")[0] == pytest.approx(0.1)
