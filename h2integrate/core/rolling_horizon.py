"""Planning scaffold for a future rolling-horizon execution mode.

This module deliberately separates architecture planning from execution. It can
describe which parts of each technology belong in a reusable window problem
without changing the existing H2Integrate OpenMDAO hierarchy. The runtime hooks
document the intended ``SubmodelComp`` boundary but remain unimplemented until
state and window interfaces are defined for participating models.
"""

from dataclasses import dataclass


WINDOW_MODEL_ROLES = ("dispatch_rule_set", "control_strategy", "performance_model")
ANNUAL_MODEL_ROLES = ("cost_model", "finance_model")
SUPPORTED_ECONOMIC_SIGNALS = ("buy_price",)


@dataclass(frozen=True)
class TechnologyPlacement:
    """Planned placement of one technology's configured model roles."""

    name: str
    window_model_roles: tuple[str, ...]
    annual_model_roles: tuple[str, ...]


@dataclass(frozen=True)
class RollingHorizonExecutionPlan:
    """Declarative boundary between annual and rolling-horizon execution."""

    enabled: bool
    prediction_horizon: int
    control_interval: int
    technology_placements: tuple[TechnologyPlacement, ...]
    boundary_interconnections: tuple[tuple, ...]
    economic_signals: tuple[tuple[str, str], ...]
    economic_feedback_signals: tuple[tuple[str, str], ...]

    @property
    def steppable_technologies(self):
        """Return technology names assigned to the future window problem."""
        return tuple(
            placement.name
            for placement in self.technology_placements
            if placement.window_model_roles
        )


def create_rolling_horizon_plan(plant_config, technology_config):
    """Create and validate a rolling-horizon execution plan from existing configs.

    Only one new plant-level block is consumed. Technology model definitions stay
    in the existing technology YAML and are assigned behind the scenes by role:
    controller, dispatch-rule, and performance models enter the window problem;
    cost and finance models remain in the annual parent problem.

    Args:
        plant_config (dict): Validated plant configuration.
        technology_config (dict): Validated technology configuration.

    Returns:
        RollingHorizonExecutionPlan | None: Plan when configured, otherwise None.
    """
    config = plant_config.get("rolling_horizon")
    if config is None:
        return None

    enabled = config.get("enabled", False)
    prediction_horizon = int(config["prediction_horizon"])
    control_interval = int(config["control_interval"])
    n_timesteps = int(plant_config["plant"]["simulation"]["n_timesteps"])

    if control_interval > prediction_horizon:
        raise ValueError("rolling_horizon control_interval cannot exceed prediction_horizon")
    if prediction_horizon > n_timesteps:
        raise ValueError("rolling_horizon prediction_horizon cannot exceed n_timesteps")

    technologies = technology_config["technologies"]
    steppable = tuple(config["steppable_technologies"])
    unknown = sorted(set(steppable) - set(technologies))
    if unknown:
        raise ValueError(f"Unknown rolling-horizon technologies: {unknown}")
    if len(set(steppable)) != len(steppable):
        raise ValueError("rolling_horizon steppable_technologies contains duplicates")

    placements = []
    for tech_name, tech_config in technologies.items():
        if tech_name in steppable:
            window_roles = tuple(role for role in WINDOW_MODEL_ROLES if role in tech_config)
            if not window_roles:
                raise ValueError(
                    f"Rolling-horizon technology '{tech_name}' has no operational model role"
                )
            annual_role_names = ANNUAL_MODEL_ROLES
        else:
            window_roles = ()
            annual_role_names = WINDOW_MODEL_ROLES + ANNUAL_MODEL_ROLES
        annual_roles = tuple(role for role in annual_role_names if role in tech_config)
        placements.append(TechnologyPlacement(tech_name, window_roles, annual_roles))

    steppable_set = set(steppable)
    boundaries = [
        tuple(connection)
        for connection in plant_config.get("technology_interconnections", [])
        if len(connection) >= 2
        and ((connection[0] in steppable_set) != (connection[1] in steppable_set))
    ]

    economic_signals = []
    economic_feedback_signals = []
    control_parameters = plant_config.get("system_level_control", {}).get("control_parameters", {})
    for tech_name, signal in control_parameters.get("cost_per_tech", {}).items():
        if isinstance(signal, int | float) or signal in SUPPORTED_ECONOMIC_SIGNALS:
            economic_signals.append((tech_name, str(signal)))
        elif signal in ("VarOpEx", "feedstock"):
            economic_feedback_signals.append((tech_name, signal))

    return RollingHorizonExecutionPlan(
        enabled=enabled,
        prediction_horizon=prediction_horizon,
        control_interval=control_interval,
        technology_placements=tuple(placements),
        boundary_interconnections=tuple(boundaries),
        economic_signals=tuple(economic_signals),
        economic_feedback_signals=tuple(economic_feedback_signals),
    )


class RollingHorizonRunner:
    """Future owner of one reusable, window-sized OpenMDAO problem.

    Intended lifecycle::

        build_window_problem()
        expose_window_problem()  # optionally wrap with om.SubmodelComp
        for start in control intervals:
            set_window_inputs(start)
            set_initial_state()
            run_window()
            commit_control_interval()
            capture_final_state()
        return assembled annual outputs

    ``SubmodelComp`` should expose only design parameters, forecasts, explicit
    initial/final states, and committed operational results. Cost and finance
    models remain in the annual parent problem. Endogenous annual economic
    feedback, when required, belongs in a separate supervisory iteration around
    the complete rolling-horizon run.
    """

    def __init__(self, plan):
        self.plan = plan

    def build_window_problem(self):
        """Create one reusable OpenMDAO Problem sized to the prediction horizon.

        The future implementation should call the existing technology factory
        with each placement's ``window_model_roles`` instead of introducing a
        second technology YAML format.
        """
        raise NotImplementedError("Rolling-horizon window construction is an architecture stub")

    def connect_window_technologies(self):
        """Recreate applicable plant interconnections inside the window problem."""
        raise NotImplementedError("Rolling-horizon connections are an architecture stub")

    def add_window_controller(self):
        """Place system-level control inside the window problem when configured."""
        raise NotImplementedError("Rolling-horizon control is an architecture stub")

    def expose_window_problem(self):
        """Map selected inner variables through a future ``om.SubmodelComp``.

        Outer inputs should be limited to design values, forecast slices, and
        initial state. Outer outputs should be committed operational trajectories
        and final state. The mapping should use promoted names generated by the
        normal H2Integrate connection helpers.
        """
        raise NotImplementedError("Rolling-horizon variable mapping is an architecture stub")

    def set_window_inputs(self, start, annual_inputs):
        """Slice annual forecasts over the prediction horizon and set inner inputs."""
        raise NotImplementedError("Rolling-horizon input slicing is an architecture stub")

    def set_initial_state(self, state):
        """Set explicit state at the beginning of the committed interval."""
        raise NotImplementedError("Rolling-horizon state input is an architecture stub")

    def commit_control_interval(self, start, annual_outputs):
        """Copy only the implemented interval from window to annual outputs."""
        raise NotImplementedError("Rolling-horizon output assembly is an architecture stub")

    def capture_final_state(self):
        """Return state after the committed interval, not after the forecast horizon."""
        raise NotImplementedError("Rolling-horizon state output is an architecture stub")

    def run(self, annual_inputs):
        """Advance the reusable window problem and assemble annual outputs."""
        raise NotImplementedError("Rolling-horizon execution is an architecture stub")


def require_implemented_runtime(plan):
    """Prevent an enabled scaffold from silently using annual execution."""
    if plan is not None and plan.enabled:
        raise NotImplementedError(
            "rolling_horizon is an architecture scaffold only. Set enabled to false to "
            "inspect the generated execution plan while using the existing annual runner."
        )
