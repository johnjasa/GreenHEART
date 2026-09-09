"""Sweep the size of a merchant battery added to an existing solar plant.

A synthetic hourly locational marginal price (LMP) series drives both the
export price and the import price. The system-level controller solves a rolling
24-hour linear program that decides, simultaneously, when to charge the battery
(from the existing plant's spill or from the grid), when to discharge, and how
much to export. Export is capped at the existing plant's unmet demand, so only
the value of the addition is counted.

The battery is the only new asset. Its power rating is fixed by
``tech_config.yaml`` and the sweep varies its energy capacity, so each case asks
how many hours of storage are worth building behind a given inverter. The driver
evaluates a full year of dispatch at each capacity and records the resulting net
present value.

Export revenue reaches the net present value through the grid cost model, which
values every exported kilowatt-hour at that hour's LMP and reports the total as
negative variable OpEx. ProFAST treats that as a coproduct, so
``commodity_sell_price`` is set to zero in ``plant_config.yaml`` to keep the
same energy from being paid for twice.

The run writes three figures: the value curve over the swept capacities, and
then the dispatch and the economics of whichever capacity came out best.

Each case is a year of hourly linear programs, so the whole sweep takes a while.
"""

from pathlib import Path

import numpy as np
import openmdao.api as om
import matplotlib.pyplot as plt

from h2integrate.core.h2integrate_model import H2IntegrateModel


def make_lmp_profile(n_timesteps=8760, seed=0):
    """Build a synthetic hourly LMP series in $/kWh.

    Combines a diurnal shape (an evening peak and a midday solar-driven trough
    that dips negative), a seasonal summer premium, and lognormal noise with
    occasional scarcity spikes. The series averages about 30 USD/MWh and its
    scarcity events reach several hundred USD/MWh, which is the spread the
    battery earns against.

    Args:
        n_timesteps (int): Number of hourly timesteps.
        seed (int): Seed for the random generator.

    Returns:
        np.ndarray: Price series of shape ``(n_timesteps,)`` in ``$/kWh``.
    """
    rng = np.random.default_rng(seed)
    hours = np.arange(n_timesteps)
    hour_of_day = hours % 24
    day_of_year = hours // 24

    # Evening ramp peak around hour 19, midday solar trough around hour 13.
    diurnal = 0.030 + 0.026 * np.sin((hour_of_day - 9) * np.pi / 12)
    midday_trough = -0.022 * np.exp(-(((hour_of_day - 13) / 2.6) ** 2))
    evening_peak = 0.030 * np.exp(-(((hour_of_day - 19) / 1.9) ** 2))

    seasonal = 0.010 * np.sin((day_of_year - 100) * 2 * np.pi / 365)
    noise = rng.lognormal(mean=0.0, sigma=0.25, size=n_timesteps) - 1.0

    price = diurnal + midday_trough + evening_peak + seasonal + 0.012 * noise

    # Scarcity spikes on a small number of hours.
    spike_hours = rng.choice(n_timesteps, size=n_timesteps // 400, replace=False)
    price[spike_hours] += rng.uniform(0.15, 0.60, size=spike_hours.size)

    return price


# -- Create the model --
n_timesteps = 8760
dt_h = 1.0
lmp = make_lmp_profile(n_timesteps)

h2i = H2IntegrateModel("solar_battery_arbitrage.yaml")
h2i.setup()

# -- Apply the LMP series to both grid connections --
# The export price feeds the controller (input-to-input) and the revenue term of
# the grid cost model. The import price feeds the controller's marginal cost for
# grid_buy and the purchase term of the cost model. A small adder on the import
# side represents transmission and ancillary charges.
h2i.prob.set_val("grid_sell.electricity_sell_price", lmp, units="USD/(kW*h)")
h2i.prob.set_val("grid_buy.electricity_buy_price", lmp + 0.004, units="USD/(kW*h)")

h2i.run()

# -- Read the value curve the sweep traced out --
sql_path = Path(__file__).parent / "outputs" / "battery_sizing_sweep.sql"
cases = list(om.CaseReader(sql_path).get_cases())
sweep_capacity = np.array(
    [case.get_design_vars()["battery.storage_capacity"].item() for case in cases]
)
sweep_npv = np.array(
    [case.get_objectives()["finance_subgroup_electricity.NPV_electricity"].item() for case in cases]
)
order = np.argsort(sweep_capacity)
sweep_capacity = sweep_capacity[order]
sweep_npv = sweep_npv[order]
best_capacity = sweep_capacity[np.argmax(sweep_npv)]


def sweep_output(name):
    """Stack one recorded output across the sweep cases, ordered by battery capacity."""
    return np.array([np.atleast_1d(case.outputs[name]) for case in cases])[order]


sweep_capex = sweep_output("battery.CapEx").ravel()
sweep_duration = sweep_output("battery.storage_duration").ravel()
sweep_sold = sweep_output("grid_sell.electricity_sold")
sweep_bought = sweep_output("grid_buy.electricity_out")

sweep_revenue = (sweep_sold * lmp).sum(axis=1) * dt_h
sweep_margin = sweep_revenue - (sweep_bought * (lmp + 0.004)).sum(axis=1) * dt_h

print("\nBattery sizing sweep")
header = ("Capacity (MWh)", "Duration (h)", "CapEx (MUSD)", "Margin (MUSD/yr)", "NPV (MUSD)")
print("".join(f"{col:>18}" for col in header))
for i, capacity in enumerate(sweep_capacity):
    marker = "  <- best" if capacity == best_capacity else ""
    print(
        f"{capacity / 1e3:>18,.1f}{sweep_duration[i]:>18,.1f}{sweep_capex[i] / 1e6:>18,.1f}"
        f"{sweep_margin[i] / 1e6:>18,.2f}{sweep_npv[i] / 1e6:>18,.2f}{marker}"
    )

# -- Plot the value curve --
fig0, ax0 = plt.subplots(figsize=(8, 5))
ax0.plot(sweep_capacity / 1e3, sweep_npv / 1e6, "o-", color="tab:blue")
ax0.axhline(0.0, color="k", linewidth=0.8, linestyle=":")
ax0.set_xlabel("Battery energy capacity (MWh), at a fixed power rating")
ax0.set_ylabel("NPV (MUSD)")
ax0.set_title("Value of the addition against battery capacity")
ax0.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("lp_arbitrage_sizing.png", dpi=150)
print("Plot saved to lp_arbitrage_sizing.png")

# The driver leaves the model on whichever case ran last, so put it back on the
# best capacity before pulling out the hourly schedule the figures below describe.
h2i.prob.set_val("battery.storage_capacity", best_capacity, units="kW*h")
h2i.prob.run_model()
h2i.post_process()

# -- Extract results --
battery_discharge = h2i.prob.get_val("plant.battery.storage_electricity_discharge", units="kW")
# The storage model reports charging as a negative rate; flip it to a magnitude.
battery_charge = -h2i.prob.get_val("plant.battery.storage_electricity_charge", units="kW")
battery_soc = h2i.prob.get_val("plant.battery.SOC", units="percent")
grid_import = h2i.prob.get_val("plant.grid_buy.electricity_out", units="kW")
grid_export = h2i.prob.get_val("plant.grid_sell.electricity_sold", units="kW")
# The existing plant's shortfall is the export ceiling; its spill is chargeable supply.
export_ceiling = h2i.prob.get_val(
    "plant.existing_load_demand.unmet_electricity_demand_out", units="kW"
)
existing_spill = h2i.prob.get_val("plant.existing_load_demand.unused_electricity_out", units="kW")
interconnection_limit = h2i.prob.get_val("plant.grid_sell.interconnection_size", units="kW").item()

charge_rate = h2i.prob.get_val("plant.battery.max_charge_rate", units="kW").item()
storage_capacity = h2i.prob.get_val("plant.battery.storage_capacity", units="kW*h").item()
battery_capex = h2i.prob.get_val("plant.battery.CapEx", units="USD").item()
utilization = h2i.prob.get_val("plant.battery.standard_capacity_factor")[0]

export_revenue = float(np.sum(grid_export * lmp) * dt_h)
import_cost = float(np.sum(grid_import * (lmp + 0.004)) * dt_h)

# Volume-weighted price the schedule captured, versus the flat market average.
realized_price = export_revenue / max(grid_export.sum() * dt_h, 1.0)
time_average_price = float(lmp.mean())
npv = h2i.prob.get_val("finance_subgroup_electricity.NPV_electricity", units="USD").item()

print("\nBest capacity in the sweep")
print(f"Capacity:               {storage_capacity / 1e3:>12,.1f} MWh")
print(f"Charge rate:            {charge_rate / 1e3:>12,.1f} MW")
print(f"Storage duration:       {storage_capacity / charge_rate:>12,.2f} h")
print(f"Battery CapEx:          {battery_capex / 1e6:>12,.1f} MUSD")
print(f"Utilization factor:     {utilization:>12,.3f}")
print(f"Annual export:          {grid_export.sum() * dt_h / 1e3:>12,.0f} MWh")
print(f"Annual import:          {grid_import.sum() * dt_h / 1e3:>12,.0f} MWh")
print(f"Battery throughput:     {battery_discharge.sum() * dt_h / 1e3:>12,.0f} MWh discharged")
print(f"Equivalent full cycles: {battery_discharge.sum() * dt_h / storage_capacity:>12,.1f}")
print(f"Gross export revenue:   {export_revenue:>12,.0f} USD")
print(f"Gross import cost:      {import_cost:>12,.0f} USD")
print(f"Gross energy margin:    {export_revenue - import_cost:>12,.0f} USD")
print(f"Time-average LMP:       {time_average_price * 1e3:>12,.2f} USD/MWh")
print(f"Realized export price:  {realized_price * 1e3:>12,.2f} USD/MWh")
print(f"Capture rate:           {100 * realized_price / time_average_price:>12,.1f}%")
print(f"NPV:                    {npv / 1e6:>12,.1f} MUSD")
charge_in_cheap_hours = battery_charge[lmp < np.quantile(lmp, 0.2)].sum() / battery_charge.sum()
discharge_in_costly_hours = (
    battery_discharge[lmp > np.quantile(lmp, 0.8)].sum() / battery_discharge.sum()
)
print(f"Charging while price < 20th pct: {100 * charge_in_cheap_hours:.1f}%")
print(f"Discharging while price > 80th pct: {100 * discharge_in_costly_hours:.1f}%")

# -- Plot a representative week --
start = 24 * 180  # mid-summer
n_hours = 168
window = slice(start, start + n_hours)
hours = np.arange(n_hours)

fig, axes = plt.subplots(5, 1, figsize=(14, 15), sharex=True)

axes[0].plot(hours, lmp[window] * 100, color="tab:red")
axes[0].axhline(0.0, color="k", linewidth=0.8, linestyle=":")
axes[0].set_ylabel("LMP (\u00a2/kWh)")
axes[0].set_title(
    f"LP Arbitrage, Representative Summer Week  |  best capacity "
    f"{storage_capacity / 1e3:,.0f} MWh at {charge_rate / 1e3:,.0f} MW"
)

axes[1].bar(
    hours,
    existing_spill[window] / 1e3,
    width=1.0,
    color="tab:orange",
    align="edge",
    label="Existing plant spill",
)
axes[1].bar(
    hours,
    grid_import[window] / 1e3,
    width=1.0,
    bottom=existing_spill[window] / 1e3,
    color="tab:gray",
    align="edge",
    label="Grid import",
)
axes[1].set_ylabel("Chargeable supply (MW)")
axes[1].legend(loc="upper right")

axes[2].bar(
    hours,
    battery_discharge[window] / 1e3,
    width=1.0,
    color="tab:green",
    align="edge",
    label="Discharge",
)
axes[2].bar(
    hours,
    -battery_charge[window] / 1e3,
    width=1.0,
    color="tab:purple",
    align="edge",
    label="Charge",
)
axes[2].axhline(0.0, color="k", linewidth=0.8)
axes[2].set_ylabel("Battery (MW)")
axes[2].legend(loc="upper right")

ax_soc = axes[2].twinx()
ax_soc.plot(hours, battery_soc[window], color="k", linewidth=1.2, label="SOC")
ax_soc.set_ylabel("SOC (%)")

axes[3].bar(
    hours,
    grid_export[window] / 1e3,
    width=1.0,
    color="tab:blue",
    align="edge",
    label="Export",
)
axes[3].step(
    hours,
    export_ceiling[window] / 1e3,
    where="post",
    color="tab:red",
    linewidth=1.6,
    label="Export ceiling (existing plant's unmet demand)",
)
axes[3].fill_between(
    hours,
    export_ceiling[window] / 1e3,
    interconnection_limit / 1e3,
    step="post",
    color="tab:red",
    alpha=0.08,
    label="Blocked by the ceiling",
)
axes[3].axhline(
    interconnection_limit / 1e3,
    color="k",
    linewidth=1.0,
    linestyle="--",
    label="Interconnection limit",
)
axes[3].set_ylabel("Export (MW)")
axes[3].legend(loc="upper right", fontsize=8)

# The existing plant alternates between spilling and falling short, and those two
# signals are what the addition is allowed to charge from and sell into.
axes[4].bar(
    hours,
    export_ceiling[window] / 1e3,
    width=1.0,
    color="tab:red",
    alpha=0.6,
    align="edge",
    label="Unmet demand (sellable headroom)",
)
axes[4].bar(
    hours,
    -existing_spill[window] / 1e3,
    width=1.0,
    color="tab:orange",
    alpha=0.8,
    align="edge",
    label="Spill (chargeable surplus)",
)
axes[4].axhline(0.0, color="k", linewidth=0.8)
axes[4].set_ylabel("Existing plant (MW)")
axes[4].set_xlabel("Hour of week")
axes[4].legend(loc="upper right", fontsize=8)

plt.tight_layout()
plt.savefig("lp_arbitrage_results.png", dpi=150)
print("Plot saved to lp_arbitrage_results.png")

# -- Plot the plant economics over the year --
hourly_revenue = grid_export * lmp * dt_h
hourly_cost = grid_import * (lmp + 0.004) * dt_h
hourly_margin = hourly_revenue - hourly_cost

fig2, ax2 = plt.subplots(2, 2, figsize=(15, 10))

# Cumulative cash flow: where the money actually accrues through the year.
days = np.arange(n_timesteps) / 24.0
ax2[0, 0].plot(days, np.cumsum(hourly_revenue) / 1e6, color="tab:blue", label="Export revenue")
ax2[0, 0].plot(days, np.cumsum(hourly_cost) / 1e6, color="tab:red", label="Import cost")
ax2[0, 0].plot(days, np.cumsum(hourly_margin) / 1e6, color="k", linewidth=2, label="Net margin")
ax2[0, 0].set_xlabel("Day of year")
ax2[0, 0].set_ylabel("Cumulative (MUSD)")
ax2[0, 0].set_title("Cumulative plant cash flow")
ax2[0, 0].legend(loc="upper left")
ax2[0, 0].grid(alpha=0.3)

# Monthly breakdown. Month boundaries for a non-leap year.
month_edges = np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]) * 24
month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
monthly_revenue = np.array(
    [hourly_revenue[month_edges[m] : month_edges[m + 1]].sum() for m in range(12)]
)
monthly_cost = np.array([hourly_cost[month_edges[m] : month_edges[m + 1]].sum() for m in range(12)])
x = np.arange(12)
ax2[0, 1].bar(x - 0.2, monthly_revenue / 1e6, width=0.4, color="tab:blue", label="Revenue")
ax2[0, 1].bar(x + 0.2, monthly_cost / 1e6, width=0.4, color="tab:red", label="Import cost")
ax2[0, 1].plot(x, (monthly_revenue - monthly_cost) / 1e6, "ko-", linewidth=1.5, label="Net margin")
ax2[0, 1].set_xticks(x)
ax2[0, 1].set_xticklabels(month_labels)
ax2[0, 1].set_ylabel("MUSD")
ax2[0, 1].set_title("Monthly revenue and cost")
ax2[0, 1].legend(loc="upper left")
ax2[0, 1].grid(alpha=0.3, axis="y")

# Average day: when the battery buys and sells relative to the price shape.
hour_of_day = np.arange(n_timesteps) % 24
mean_charge = np.array([battery_charge[hour_of_day == h].mean() for h in range(24)])
mean_discharge = np.array([battery_discharge[hour_of_day == h].mean() for h in range(24)])
mean_price = np.array([lmp[hour_of_day == h].mean() for h in range(24)])
ax2[1, 0].bar(range(24), mean_discharge / 1e3, width=0.8, color="tab:green", label="Discharge")
ax2[1, 0].bar(range(24), -mean_charge / 1e3, width=0.8, color="tab:purple", label="Charge")
ax2[1, 0].axhline(0.0, color="k", linewidth=0.8)
ax2[1, 0].set_xlabel("Hour of day")
ax2[1, 0].set_ylabel("Mean battery power (MW)")
ax2[1, 0].set_title("Average daily arbitrage cycle")
ax2[1, 0].legend(loc="upper left")
ax_price = ax2[1, 0].twinx()
ax_price.plot(range(24), mean_price * 1e3, color="tab:red", linewidth=2, label="Mean LMP")
ax_price.set_ylabel("Mean LMP (USD/MWh)")

# Revenue concentration: merchant plants earn a large share in very few hours.
sorted_revenue = np.sort(hourly_revenue)[::-1]
revenue_share = np.cumsum(sorted_revenue) / sorted_revenue.sum()
hour_share = np.arange(1, n_timesteps + 1) / n_timesteps
ax2[1, 1].plot(hour_share * 100, revenue_share * 100, color="tab:blue", linewidth=2)
ax2[1, 1].plot([0, 100], [0, 100], "k:", linewidth=1, label="Uniform revenue")
top_decile = revenue_share[int(0.1 * n_timesteps)] * 100
ax2[1, 1].axvline(10, color="tab:red", linestyle="--", linewidth=1)
ax2[1, 1].annotate(
    f"Top 10% of hours\n= {top_decile:.0f}% of revenue",
    xy=(10, top_decile),
    xytext=(25, top_decile - 25),
    arrowprops={"arrowstyle": "->", "color": "tab:red"},
    color="tab:red",
)
ax2[1, 1].set_xlabel("Share of hours, ranked by revenue (%)")
ax2[1, 1].set_ylabel("Share of annual revenue (%)")
ax2[1, 1].set_title("Revenue concentration")
ax2[1, 1].legend(loc="lower right")
ax2[1, 1].grid(alpha=0.3)

fig2.suptitle(
    f"LP Arbitrage Economics  |  {charge_rate / 1e3:,.0f} MW / "
    f"{storage_capacity / 1e3:,.0f} MWh battery  |  capture rate "
    f"{100 * realized_price / time_average_price:.0f}%  |  NPV {npv / 1e6:,.1f} MUSD",
    fontsize=13,
)
plt.tight_layout()
plt.savefig("lp_arbitrage_economics.png", dpi=150)
print("Plot saved to lp_arbitrage_economics.png")
