"""Rolling-horizon planning and reusable window execution.

This module keeps rolling-horizon construction and execution separate from the
full-simulation ``H2IntegrateModel`` lifecycle. Existing technology APIs remain unchanged;
model-specific state is accessed through adapters until public state ports are
proven necessary.
"""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import openmdao.api as om

from h2integrate.core.supported_models import supported_models, no_replacement_schedule_models
from h2integrate.control.control_strategies.passthrough_controller import PassthroughController


WINDOW_MODEL_ROLES = ("dispatch_rule_set", "control_strategy", "performance_model")
FULL_SIM_MODEL_ROLES = ("cost_model", "finance_model")
UNSUPPORTED_COMBINED_MODELS = frozenset(
    {"WOMBATElectrolyzerModel", "IronComponent", "ArdWindPlantModel"}
)


@dataclass(frozen=True)
class RollingHorizonExecutionPlan:
    """Declarative boundary between full- and part-simulation execution."""

    enabled: bool
    n_timesteps: int
    prediction_horizon: int
    control_interval: int
    initial_states: dict[str, dict[str, float]]
    steppable_technologies: tuple[str, ...]
    window_model_roles: dict[str, tuple[str, ...]]
    full_sim_model_roles: dict[str, tuple[str, ...]]


def create_rolling_horizon_plan(plant_config, technology_config):
    """Create and validate a rolling-horizon execution plan from existing configs.

    Only one new plant-level block is consumed. Technology model definitions stay
    in the existing technology YAML and are assigned behind the scenes by role:
    controller, dispatch-rule, and performance models enter the window problem;
    cost and finance models remain in the full-simulation parent problem.

    Args:
        plant_config (dict): Validated plant configuration.
        technology_config (dict): Validated technology configuration.

    Returns:
        RollingHorizonExecutionPlan | None: Plan when configured, otherwise None.
    """
    config = plant_config.get("rolling_horizon")
    if config is None:
        return None

    n_timesteps = int(plant_config["plant"]["simulation"]["n_timesteps"])
    prediction_horizon = int(config["prediction_horizon"])
    control_interval = int(config["control_interval"])
    steppable = tuple(config["steppable_technologies"])
    initial_states = {
        tech_name: dict(states) for tech_name, states in config.get("initial_states", {}).items()
    }
    technologies = technology_config["technologies"]

    if control_interval > prediction_horizon:
        raise ValueError("rolling_horizon control_interval cannot exceed prediction_horizon")
    if prediction_horizon > n_timesteps:
        raise ValueError("rolling_horizon prediction_horizon cannot exceed n_timesteps")
    if len(set(steppable)) != len(steppable):
        raise ValueError("rolling_horizon steppable_technologies contains duplicates")
    unknown = sorted(set(steppable) - set(technologies))
    if unknown:
        raise ValueError(f"Unknown rolling-horizon technologies: {unknown}")
    invalid_state_owners = sorted(set(initial_states) - set(steppable))
    if invalid_state_owners:
        raise ValueError(
            "Rolling-horizon initial states must belong to steppable technologies: "
            f"{invalid_state_owners}"
        )

    window_model_roles = {}
    full_sim_model_roles = {}
    for tech_name, tech_config in technologies.items():
        if tech_name in steppable:
            window_roles = tuple(role for role in WINDOW_MODEL_ROLES if role in tech_config)
            if not window_roles:
                raise ValueError(
                    f"Rolling-horizon technology '{tech_name}' has no operational model role"
                )
            model_names = {
                tech_config[role]["model"] for role in window_roles if "model" in tech_config[role]
            }
            unsupported = sorted(model_names & UNSUPPORTED_COMBINED_MODELS)
            if unsupported:
                raise ValueError(
                    f"Rolling-horizon technology '{tech_name}' uses unsupported combined "
                    f"model(s): {unsupported}"
                )
            full_sim_role_names = FULL_SIM_MODEL_ROLES
        else:
            window_roles = ()
            full_sim_role_names = WINDOW_MODEL_ROLES + FULL_SIM_MODEL_ROLES
        full_sim_roles = tuple(role for role in full_sim_role_names if role in tech_config)
        window_model_roles[tech_name] = window_roles
        full_sim_model_roles[tech_name] = full_sim_roles

    return RollingHorizonExecutionPlan(
        enabled=bool(config.get("enabled", False)),
        n_timesteps=n_timesteps,
        prediction_horizon=prediction_horizon,
        control_interval=control_interval,
        initial_states=initial_states,
        steppable_technologies=steppable,
        window_model_roles=window_model_roles,
        full_sim_model_roles=full_sim_model_roles,
    )


@dataclass(frozen=True)
class RollingHorizonInputAlias:
    """Map one full-simulation input to one or more part-simulation inputs."""

    name: str
    targets: tuple[str, ...]
    units: str | None = None
    source: str | None = None
    windowed: bool = True
    shape: int | tuple[int, ...] | None = None


@dataclass(frozen=True)
class RollingHorizonOutputAlias:
    """Map one part-simulation timeseries to a full-simulation output."""

    name: str
    source: str
    units: str | None = None
    targets: tuple[str, ...] = ()
    windowed: bool = True
    shape: int | tuple[int, ...] | None = None
    aggregation: str | None = None


def _alias_name(variable_path):
    """Return a stable OpenMDAO variable name for an absolute source path."""
    return variable_path.replace(".", "_").replace(":", "_")


def _alias_units(variable_path):
    """Return units for the initial supported rolling-horizon boundary variables."""
    parameter = variable_path.rsplit(".", 1)[-1]
    if parameter == "capacity_factor" or parameter == "replacement_schedule":
        return "unitless"
    if "electricity" in parameter:
        return "kW"
    return None


def create_boundary_input_aliases(plan, plant_config):
    """Create full-to-part aliases from crossing technology connections."""
    steppable = set(plan.steppable_technologies)
    aliases = {}
    combiner_counts = {}
    plant_life = int(plant_config["plant"]["plant_life"])

    for connection in plant_config.get("technology_interconnections", []):
        source_tech, dest_tech, connected_parameter = connection[:3]
        combiner_index = None
        if "combiner" in dest_tech:
            combiner_counts[dest_tech] = combiner_counts.get(dest_tech, 0) + 1
            combiner_index = combiner_counts[dest_tech]

        if source_tech in steppable or dest_tech not in steppable:
            continue

        if isinstance(connected_parameter, list | tuple):
            source_parameter, destination_parameter = connected_parameter
        elif len(connection) == 3:
            source_parameter = destination_parameter = connected_parameter
        else:
            source_parameter = f"{connected_parameter}_out"
            destination_parameter = f"{connected_parameter}_in"
            if combiner_index is not None:
                destination_parameter += str(combiner_index)

        source = f"{source_tech}.{source_parameter}"
        aliases.setdefault((source, True), []).append(f"{dest_tech}.{destination_parameter}")

        if combiner_index is not None and not isinstance(connected_parameter, list | tuple):
            commodity = connected_parameter
            rated_source = f"{source_tech}.rated_{commodity}_production"
            aliases.setdefault((rated_source, False), []).append(
                f"{dest_tech}.rated_{commodity}_production{combiner_index}"
            )
            capacity_source = f"{source_tech}.capacity_factor"
            aliases.setdefault((capacity_source, False), []).append(
                f"{dest_tech}.{commodity}_capacity_factor{combiner_index}"
            )

    result = []
    for (source, windowed), targets in aliases.items():
        shape = plant_life if not windowed and source.endswith(".capacity_factor") else None
        result.append(
            RollingHorizonInputAlias(
                name=_alias_name(source),
                targets=tuple(targets),
                units=_alias_units(source),
                source=source,
                windowed=windowed,
                shape=shape,
            )
        )
    return tuple(result)


def create_boundary_output_aliases(plan, plant_config, technology_config):
    """Create part-to-full aliases for topology boundaries and cost inputs."""
    steppable = set(plan.steppable_technologies)
    aliases = {}
    combiner_counts = {}
    plant_life = int(plant_config["plant"]["plant_life"])

    def add(source, target=None, *, windowed=True, shape=None, aggregation=None):
        key = (source, windowed, shape, aggregation)
        targets = aliases.setdefault(key, [])
        if target is not None:
            targets.append(target)

    for connection in plant_config.get("technology_interconnections", []):
        source_tech, dest_tech, connected_parameter = connection[:3]
        combiner_index = None
        if "combiner" in dest_tech:
            combiner_counts[dest_tech] = combiner_counts.get(dest_tech, 0) + 1
            combiner_index = combiner_counts[dest_tech]

        if source_tech not in steppable or dest_tech in steppable:
            continue

        if isinstance(connected_parameter, list | tuple):
            source_parameter, destination_parameter = connected_parameter
        elif len(connection) == 3:
            source_parameter = destination_parameter = connected_parameter
        else:
            source_parameter = f"{connected_parameter}_out"
            destination_parameter = f"{connected_parameter}_in"
            if combiner_index is not None:
                destination_parameter += str(combiner_index)

        source = f"{source_tech}.{source_parameter}"
        add(source, f"{dest_tech}.{destination_parameter}")

        if combiner_index is not None and not isinstance(connected_parameter, list | tuple):
            commodity = connected_parameter
            add(
                f"{source_tech}.rated_{commodity}_production",
                f"{dest_tech}.rated_{commodity}_production{combiner_index}",
                windowed=False,
            )
            add(
                f"{source_tech}.capacity_factor",
                f"{dest_tech}.{commodity}_capacity_factor{combiner_index}",
                windowed=False,
                shape=plant_life,
                aggregation="capacity_factor",
            )

    for tech_name in steppable:
        tech_config = technology_config["technologies"][tech_name]
        if tech_name in plan.initial_states:
            model_inputs = tech_config.get("model_inputs", {})
            commodity = model_inputs.get("shared_parameters", {}).get("commodity")
            if commodity is not None:
                add(f"{tech_name}.{commodity}_out")
            if "soc" in plan.initial_states[tech_name]:
                add(f"{tech_name}.SOC")

        cost_model = tech_config.get("cost_model", {}).get("model")
        cost_parameters = tech_config.get("model_inputs", {}).get("cost_parameters", {})
        if cost_model != "GridCostModel":
            continue
        if cost_parameters.get("electricity_buy_price") is not None:
            add(f"{tech_name}.electricity_out", f"{tech_name}.electricity_out")
        if cost_parameters.get("electricity_sell_price") is not None:
            add(f"{tech_name}.electricity_sold", f"{tech_name}.electricity_sold")

    finance_subgroups = plant_config.get("finance_parameters", {}).get("finance_subgroups", {})
    for subgroup_name, subgroup_config in finance_subgroups.items():
        for tech_name in set(subgroup_config.get("technologies", ())) & steppable:
            perf_model = (
                technology_config["technologies"][tech_name]
                .get("performance_model", {})
                .get("model")
            )
            if perf_model not in no_replacement_schedule_models:
                add(
                    f"{tech_name}.replacement_schedule",
                    f"finance_subgroup_{subgroup_name}.replacement_schedule_{tech_name}",
                    windowed=False,
                    shape=plant_life,
                )

    return tuple(
        RollingHorizonOutputAlias(
            name=_alias_name(source),
            source=source,
            units=_alias_units(source),
            targets=tuple(targets),
            windowed=windowed,
            shape=shape,
            aggregation=aggregation,
        )
        for (source, windowed, shape, aggregation), targets in aliases.items()
    )


def create_window_model_configuration(driver_config, plant_config, technology_config, plan):
    """Create an operations-only configuration without mutating full-simulation inputs."""
    window_driver_config = deepcopy(driver_config)
    window_driver_config.pop("driver", None)
    window_driver_config.pop("recorder", None)

    window_plant_config = deepcopy(plant_config)
    window_plant_config.pop("rolling_horizon", None)
    window_plant_config.pop("finance_parameters", None)
    window_plant_config.pop("sites", None)
    window_plant_config.pop("resource_to_tech_connections", None)
    simulation = window_plant_config["plant"]["simulation"]
    simulation["n_timesteps"] = plan.prediction_horizon
    simulation["n_steps_per_compute"] = plan.prediction_horizon

    steppable = set(plan.steppable_technologies)
    window_plant_config["technology_interconnections"] = [
        deepcopy(connection)
        for connection in plant_config.get("technology_interconnections", [])
        if connection[0] in steppable and connection[1] in steppable
    ]
    window_plant_config["tech_to_dispatch_connections"] = [
        deepcopy(connection)
        for connection in plant_config.get("tech_to_dispatch_connections", [])
        if connection[0] in steppable and connection[1] in steppable
    ]

    window_technologies = {}
    for tech_name in plan.steppable_technologies:
        tech_config = deepcopy(technology_config["technologies"][tech_name])
        allowed_roles = set(plan.window_model_roles[tech_name])
        for role in WINDOW_MODEL_ROLES + FULL_SIM_MODEL_ROLES:
            if role not in allowed_roles:
                tech_config.pop(role, None)
        window_technologies[tech_name] = tech_config

    window_technology_config = deepcopy(technology_config)
    window_technology_config["technologies"] = window_technologies
    return {
        "driver_config": window_driver_config,
        "plant_config": window_plant_config,
        "technology_config": window_technology_config,
        "all_interconnections": tuple(
            tuple(connection) for connection in plant_config.get("technology_interconnections", [])
        ),
    }


def _add_window_passthrough_controller(tech_group, perf_component, tech_config, n_timesteps):
    """Add the standard passthrough controller when a window technology needs one."""
    if "control_strategy" in tech_config:
        return
    if getattr(perf_component, "_control_classifier", None) not in (
        "flexible",
        "dispatchable",
        "storage",
    ):
        return

    model_inputs = tech_config.get("model_inputs", {})
    shared = model_inputs.get("shared_parameters", {})
    performance = model_inputs.get("performance_parameters", {})
    commodity = getattr(perf_component, "commodity", None) or performance.get(
        "commodity", shared.get("commodity")
    )
    units = getattr(perf_component, "commodity_rate_units", None) or performance.get(
        "commodity_rate_units", shared.get("commodity_rate_units")
    )
    if commodity is None or units is None:
        return

    controller = PassthroughController(
        commodity=commodity,
        n_timesteps=n_timesteps,
        commodity_rate_units=units,
    )
    tech_group.add_subsystem("controller", controller, promotes=["*"])
    order = list(tech_group._static_subsystems_allprocs)
    tech_group.set_order(["controller", *[name for name in order if name != "controller"]])


def _validate_window_component_timestep(model_name, model_class, plant_config):
    """Apply the existing component timestep bounds to a window component."""
    dt = int(plant_config["plant"]["simulation"]["dt"])
    minimum, maximum = model_class._time_step_bounds
    if not minimum <= dt <= maximum:
        raise ValueError(
            f"Model {model_name} is incompatible with the rolling-horizon timestep {dt}"
        )


def create_h2integrate_window_problem(configuration, model_registry=None):
    """Build an operations-only OpenMDAO problem for the initial supported topology.

    The first implementation supports ordinary technology groups, direct
    interconnections, cable transport, and combiners. More complex topology is
    rejected explicitly until its full-simulation/part-simulation port contract is defined.
    """
    if "system_level_control" in configuration["plant_config"]:
        raise NotImplementedError("System-level control window wiring is not implemented")

    registry = supported_models if model_registry is None else model_registry
    problem = om.Problem(reports=False)
    plant = problem.model.add_subsystem("plant", om.Group(), promotes=["*"])
    n_timesteps = int(configuration["plant_config"]["plant"]["simulation"]["n_timesteps"])

    performance_components = {}
    for tech_name, tech_config in configuration["technology_config"]["technologies"].items():
        tech_group = plant.add_subsystem(tech_name, om.Group())
        for role in WINDOW_MODEL_ROLES:
            if role not in tech_config:
                continue
            model_name = tech_config[role]["model"]
            model_class = registry[model_name]
            _validate_window_component_timestep(
                model_name, model_class, configuration["plant_config"]
            )
            component = model_class(
                driver_config=configuration["driver_config"],
                plant_config=configuration["plant_config"],
                tech_config=tech_config,
            )
            tech_group.add_subsystem(model_name, component, promotes=["*"])
            if role == "performance_model":
                performance_components[tech_name] = component

        performance_component = performance_components.get(tech_name)
        if performance_component is not None:
            _add_window_passthrough_controller(
                tech_group, performance_component, tech_config, n_timesteps
            )

    included = set(configuration["technology_config"]["technologies"])
    combiner_counts = {}
    for connection in configuration["all_interconnections"]:
        if len(connection) not in (3, 4):
            raise ValueError(f"Invalid rolling-horizon connection: {connection}")
        source_tech, dest_tech, connected_parameter = connection[:3]
        if "splitter" in source_tech or "splitter" in dest_tech:
            raise NotImplementedError("Splitter window connections are not implemented")

        combiner_index = None
        if "combiner" in dest_tech:
            combiner_counts[dest_tech] = combiner_counts.get(dest_tech, 0) + 1
            combiner_index = combiner_counts[dest_tech]
        if source_tech not in included or dest_tech not in included:
            continue

        if isinstance(connected_parameter, list | tuple):
            source_parameter, destination_parameter = connected_parameter
            plant.connect(
                f"{source_tech}.{source_parameter}",
                f"{dest_tech}.{destination_parameter}",
            )
            continue

        if len(connection) == 3:
            plant.connect(
                f"{source_tech}.{connected_parameter}",
                f"{dest_tech}.{connected_parameter}",
            )
            continue

        transport_type = connection[3]
        if transport_type != "cable":
            raise NotImplementedError(
                f"Rolling-horizon transport '{transport_type}' is not implemented"
            )
        connection_name = f"{connected_parameter}_{source_tech}_to_{dest_tech}_{transport_type}"
        transport_class = registry[transport_type]
        transport = transport_class(
            transport_item=connected_parameter,
            plant_config=configuration["plant_config"],
        )
        plant.add_subsystem(connection_name, transport)
        plant.connect(
            f"{source_tech}.{connected_parameter}_out",
            f"{connection_name}.{connected_parameter}_in",
        )
        destination_input = f"{connected_parameter}_in"
        if combiner_index is not None:
            destination_input += str(combiner_index)
        plant.connect(
            f"{connection_name}.{connected_parameter}_out",
            f"{dest_tech}.{destination_input}",
        )
        if combiner_index is not None:
            plant.connect(
                f"{source_tech}.rated_{connected_parameter}_production",
                f"{dest_tech}.rated_{connected_parameter}_production{combiner_index}",
            )
            plant.connect(
                f"{source_tech}.capacity_factor",
                f"{dest_tech}.{connected_parameter}_capacity_factor{combiner_index}",
            )

    plant.options["auto_order"] = True
    return problem


@dataclass(frozen=True)
class StorageSOCStateAdapter:
    """Adapt current storage SOC attributes without changing storage model APIs.

    This adapter is intentionally isolated around existing private storage state.
    It can be replaced by public OpenMDAO state ports after that contract has been
    proven without forcing all storage implementations to change now.
    """

    technology_name: str
    component_paths: tuple[str, ...]
    state_name: str = "soc"

    @staticmethod
    def _get_component(problem, path):
        component = problem.model._get_subsystem(path)
        if component is None:
            raise ValueError(f"Storage state component '{path}' was not found")
        return component

    def set_initial_state(self, problem, value):
        """Seed all participating storage components with one SOC fraction."""
        state = float(value)
        for path in self.component_paths:
            component = self._get_component(problem, path)
            component.soc_init = state
            if hasattr(component, "_soc_timeseries"):
                component._soc_timeseries.fill(state)

    def capture_final_state(self, problem, committed_steps):
        """Read SOC after the committed interval from the authoritative component."""
        if committed_steps < 1:
            raise ValueError("committed_steps must be at least one")
        component = self._get_component(problem, self.component_paths[-1])
        return float(component._soc_timeseries[committed_steps - 1])


class RollingHorizonController:
    """Own a reusable part-simulation problem and advance it over a full simulation.

    Intended lifecycle::

        build_window_problem()
        for start in control intervals:
            set_window_inputs(start)
            set_initial_state()
            run the reusable window problem
            commit_control_interval()
            capture_final_state()
        return assembled full-simulation outputs

    Input and output mappings are full-simulation names mapped to inner OpenMDAO
    variable names. Model-specific state is isolated behind adapters.
    """

    def __init__(
        self,
        plan,
        window_configuration,
        problem_factory=create_h2integrate_window_problem,
        input_mappings=None,
        static_input_mappings=None,
        output_mappings=None,
        static_output_mappings=None,
        state_adapters=(),
    ):
        self.plan = plan
        self.window_configuration = window_configuration
        self.problem_factory = problem_factory
        self.input_mappings = {
            full_sim_name: (targets,) if isinstance(targets, str) else tuple(targets)
            for full_sim_name, targets in (input_mappings or {}).items()
        }
        self.static_input_mappings = {
            full_sim_name: (targets,) if isinstance(targets, str) else tuple(targets)
            for full_sim_name, targets in (static_input_mappings or {}).items()
        }
        self.output_mappings = dict(output_mappings or {})
        self.static_output_mappings = dict(static_output_mappings or {})
        self.state_adapters = tuple(state_adapters)
        self.problem = None
        self._states = deepcopy(plan.initial_states)

    def build_window_problem(self):
        """Build and set up the reusable window problem once."""
        self.problem = self.problem_factory(self.window_configuration)
        self.problem.setup()
        return self.problem

    @staticmethod
    def _slice_and_pad(values, start, horizon):
        """Return one fixed-size forecast, repeating the final available value."""
        values = np.asarray(values)
        window = values[start : start + horizon]
        if window.size == 0:
            raise ValueError("Cannot create a rolling-horizon window from an empty forecast")
        if window.shape[0] < horizon:
            padding = [(0, horizon - window.shape[0])] + [(0, 0)] * (window.ndim - 1)
            window = np.pad(window, padding, mode="edge")
        return window

    def set_window_inputs(self, start, full_sim_inputs):
        """Slice full-simulation inputs over the prediction horizon."""
        for full_sim_name, window_names in self.input_mappings.items():
            if full_sim_name not in full_sim_inputs:
                raise KeyError(f"Missing full-simulation rolling-horizon input '{full_sim_name}'")
            values = self._slice_and_pad(
                full_sim_inputs[full_sim_name], start, self.plan.prediction_horizon
            )
            for window_name in window_names:
                self.problem.set_val(window_name, values)
        for full_sim_name, window_names in self.static_input_mappings.items():
            if full_sim_name not in full_sim_inputs:
                raise KeyError(f"Missing static rolling-horizon input '{full_sim_name}'")
            for window_name in window_names:
                self.problem.set_val(window_name, full_sim_inputs[full_sim_name])

    def set_initial_state(self, state):
        """Set explicit state at the beginning of the committed interval."""
        for adapter in self.state_adapters:
            try:
                value = state[adapter.technology_name][adapter.state_name]
            except KeyError as exc:
                state_key = f"{adapter.technology_name}.{adapter.state_name}"
                raise KeyError(f"Missing initial rolling-horizon state '{state_key}'") from exc
            adapter.set_initial_state(self.problem, value)

    def commit_control_interval(self, start, full_sim_outputs, committed_steps):
        """Copy only the implemented interval into full-simulation outputs."""
        for full_sim_name, window_name in self.output_mappings.items():
            window_values = np.asarray(self.problem.get_val(window_name))
            if window_values.ndim == 0:
                raise ValueError(f"Rolling-horizon output '{window_name}' must be a timeseries")
            if full_sim_name not in full_sim_outputs:
                full_sim_outputs[full_sim_name] = np.zeros(
                    (self.plan.n_timesteps, *window_values.shape[1:]), dtype=window_values.dtype
                )
            full_sim_outputs[full_sim_name][start : start + committed_steps] = window_values[
                :committed_steps
            ]

    def capture_final_state(self, committed_steps):
        """Return state after the committed interval, not after the forecast horizon."""
        for adapter in self.state_adapters:
            tech_states = self._states.setdefault(adapter.technology_name, {})
            tech_states[adapter.state_name] = adapter.capture_final_state(
                self.problem, committed_steps
            )
        return deepcopy(self._states)

    def _run_once(self, full_sim_inputs):
        """Run one complete full-simulation rolling-horizon evaluation."""
        if self.problem is None:
            self.build_window_problem()

        self._states = deepcopy(self.plan.initial_states)
        full_sim_outputs = {}
        for start in range(0, self.plan.n_timesteps, self.plan.control_interval):
            committed_steps = min(self.plan.control_interval, self.plan.n_timesteps - start)
            self.set_window_inputs(start, full_sim_inputs)
            self.set_initial_state(self._states)
            self.problem.run_model()
            self.commit_control_interval(start, full_sim_outputs, committed_steps)
            self.capture_final_state(committed_steps)

        for full_sim_name, window_name in self.static_output_mappings.items():
            full_sim_outputs[full_sim_name] = np.array(self.problem.get_val(window_name), copy=True)

        return full_sim_outputs, deepcopy(self._states)

    def run(self, full_sim_inputs):
        """Run one complete rolling-horizon simulation."""
        inputs = {name: np.array(value, copy=True) for name, value in full_sim_inputs.items()}
        return self._run_once(inputs)


class RollingHorizonComponent(om.ExplicitComponent):
    """Expose a rolling-horizon controller as one full-simulation OpenMDAO component.

    Cost and finance models remain ordinary full-simulation subsystems. They run
    after this component and consume its assembled operational outputs.
    """

    def initialize(self):
        self.options.declare("controller", types=RollingHorizonController, recordable=False)
        self.options.declare("input_aliases", types=tuple)
        self.options.declare("output_aliases", types=tuple)

    def setup(self):
        self.controller = self.options["controller"]
        self.input_aliases = self.options["input_aliases"]
        self.output_aliases = self.options["output_aliases"]
        n_timesteps = self.controller.plan.n_timesteps

        self.controller.input_mappings = {
            alias.name: alias.targets for alias in self.input_aliases if alias.windowed
        }
        self.controller.static_input_mappings = {
            alias.name: alias.targets for alias in self.input_aliases if not alias.windowed
        }
        self.controller.output_mappings = {
            alias.name: alias.source for alias in self.output_aliases if alias.windowed
        }
        self.controller.static_output_mappings = {
            alias.name: alias.source for alias in self.output_aliases if not alias.windowed
        }

        for alias in self.input_aliases:
            if alias.windowed:
                self.add_input(alias.name, val=np.zeros(n_timesteps), units=alias.units)
            elif alias.shape is None:
                self.add_input(alias.name, val=0.0, units=alias.units)
            else:
                self.add_input(alias.name, val=np.zeros(alias.shape), units=alias.units)
        for alias in self.output_aliases:
            if alias.windowed:
                self.add_output(alias.name, val=np.zeros(n_timesteps), units=alias.units)
            elif alias.shape is None:
                self.add_output(alias.name, val=0.0, units=alias.units)
            else:
                self.add_output(alias.name, val=np.zeros(alias.shape), units=alias.units)

        for technology_name, states in self.controller.plan.initial_states.items():
            for state_name, value in states.items():
                self.add_output(
                    f"final_{technology_name}_{state_name}",
                    val=value,
                    units="unitless",
                )

    def compute(self, inputs, outputs):
        """Run all windows and publish assembled full-simulation trajectories."""
        full_sim_inputs = {
            alias.name: np.asarray(inputs[alias.name]) for alias in self.input_aliases
        }
        full_sim_outputs, final_states = self.controller.run(full_sim_inputs)
        for alias in self.output_aliases:
            if alias.aggregation == "capacity_factor":
                prefix = alias.source.rsplit(".", 1)[0]
                timeseries_alias = next(
                    candidate
                    for candidate in self.output_aliases
                    if candidate.windowed
                    and candidate.source.startswith(f"{prefix}.")
                    and candidate.source.endswith("_out")
                )
                rated_alias = next(
                    candidate
                    for candidate in self.output_aliases
                    if not candidate.windowed and candidate.source.startswith(f"{prefix}.rated_")
                )
                rated_production = float(np.asarray(full_sim_outputs[rated_alias.name]).item())
                if rated_production > 0:
                    capacity_factor = (
                        np.mean(full_sim_outputs[timeseries_alias.name]) / rated_production
                    )
                else:
                    capacity_factor = 0.0
                outputs[alias.name] = np.full(outputs[alias.name].shape, capacity_factor)
            else:
                outputs[alias.name] = full_sim_outputs[alias.name]
        for technology_name, states in final_states.items():
            for state_name, value in states.items():
                outputs[f"final_{technology_name}_{state_name}"] = value
