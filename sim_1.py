import json
from functools import partial, reduce

import matplotlib.pyplot as plt
import numpy as np
import torch


def create_input_spike(t, duration=100):
    return torch.zeros(duration).scatter_(0, torch.tensor([t]), 1.0)


def create_spike_train(times, duration=100):
    return reduce(torch.add, map(partial(create_input_spike, duration=duration), times))


def lif_step(state: dict, input_val: float, beta: float, threshold: float) -> dict:
    voltage = state["voltage"] * beta + input_val
    spike = 1.0 if voltage >= threshold else 0.0
    voltage = 0.0 if spike > 0 else voltage
    return {"voltage": voltage, "spike": spike}


def run_lif_simulation(inputs: torch.Tensor, beta: float = 0.8, threshold: float = 0.5) -> dict:
    initial_state = {"voltage": 0.0, "spike": 0.0}

    def simulate_step(acc, inp):
        new_state = lif_step(acc[-1], inp.item(), beta, threshold)
        return acc + [new_state]

    states = reduce(simulate_step, inputs, [initial_state])[1:]

    return {
        "spikes": torch.tensor([s["spike"] for s in states]),
        "voltages": torch.tensor([s["voltage"] for s in states]),
        "spike_times": [i for i, s in enumerate(states) if s["spike"] > 0],
    }


def stdp_window(
    dt: float, A_plus: float = 0.01, A_minus: float = 0.012, tau_plus: float = 20.0, tau_minus: float = 20.0
) -> float:
    return A_plus * np.exp(-dt / tau_plus) if dt > 0 else -A_minus * np.exp(dt / tau_minus)


def find_spike_pairs(pre_spikes: torch.Tensor, post_spikes: torch.Tensor, max_dt: int = 10) -> list[tuple]:
    pre_times = torch.nonzero(pre_spikes, as_tuple=False).flatten().tolist()
    post_times = torch.nonzero(post_spikes, as_tuple=False).flatten().tolist()
    pairs = []
    for pre_t in pre_times:
        for post_t in post_times:
            dt = post_t - pre_t
            if 0 < abs(dt) <= max_dt:
                pairs.append((pre_t, post_t, dt))
    return pairs


def apply_stdp_rule(pairs: list[tuple], initial_weight: float = 2.0) -> dict:
    weight_changes = [stdp_window(dt) for _, _, dt in pairs]
    final_weight = initial_weight + sum(weight_changes)
    return {
        "initial_weight": initial_weight,
        "final_weight": final_weight,
        "weight_change": final_weight - initial_weight,
        "pairs": pairs,
        "changes": weight_changes,
    }


def task1_experiment() -> dict:
    print("TASK 1: LIF Analysis")
    input_spikes = create_spike_train([5], duration=20)
    synaptic_current = torch.cat([torch.zeros(1), input_spikes[:-1] * 3.0])  # Stronger current
    results = run_lif_simulation(synaptic_current, beta=0.8, threshold=0.4)  # Lower threshold

    print(f"  Max voltage: {torch.max(results['voltages']):.3f}")
    print(f"  Spike times: {results['spike_times']}")

    return {"input_times": [5], "current": synaptic_current, "results": results, "duration": 20}


def stdp_potentiation_experiment() -> dict:
    print("STDP Potentiation: Pre → Post (+2ms)")
    pre_spikes = create_spike_train([10, 30, 50])
    post_spikes = create_spike_train([12, 32, 52])
    pairs = find_spike_pairs(pre_spikes, post_spikes)
    stdp_results = apply_stdp_rule(pairs)
    return {
        "type": "potentiation",
        "pre_spikes": pre_spikes,
        "post_spikes": post_spikes,
        "pre_times": [10, 30, 50],
        "post_times": [12, 32, 52],
        "stdp_results": stdp_results,
    }


def stdp_depression_experiment() -> dict:
    print("STDP Depression: Post → Pre (-2ms)")
    pre_spikes = create_spike_train([12, 32, 52])
    post_spikes = create_spike_train([10, 30, 50])
    pairs = find_spike_pairs(pre_spikes, post_spikes)
    stdp_results = apply_stdp_rule(pairs)
    return {
        "type": "depression",
        "pre_spikes": pre_spikes,
        "post_spikes": post_spikes,
        "pre_times": [12, 32, 52],
        "post_times": [10, 30, 50],
        "stdp_results": stdp_results,
    }


def plot_task1(data: dict) -> None:
    plt.style.use("dark_background")
    plt.rcParams["axes.facecolor"] = "#121212"
    plt.rcParams["figure.facecolor"] = "#121212"
    plt.rcParams["savefig.facecolor"] = "#121212"

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    time_axis = np.arange(data["duration"])

    if data["results"]["spike_times"]:
        axes[0].eventplot([data["results"]["spike_times"]], colors=["red"], linewidths=4)
        axes[0].set_title(f"Task 1: Spike Raster - {len(data['results']['spike_times'])} spike(s)")
    else:
        axes[0].text(10, 0.5, "No spikes detected", ha="center")
        axes[0].set_title("Task 1: No spikes detected")
    axes[0].set_ylabel("LIF Neuron")
    axes[0].set_xlim(0, data["duration"])

    axes[1].plot(time_axis, data["results"]["voltages"], "cyan", linewidth=2)
    axes[1].axhline(y=0.4, color="red", linestyle="--", alpha=0.8, label="Threshold (0.4)")
    axes[1].axvline(x=5, color="orange", linestyle=":", alpha=0.7, label="Input (5ms)")
    axes[1].axvline(x=6, color="yellow", linestyle=":", alpha=0.7, label="Current (6ms)")
    axes[1].set_title("Membrane Potential")
    axes[1].set_ylabel("Voltage")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(time_axis, data["current"], "lime", linewidth=2)
    axes[2].axvline(x=6, color="orange", linestyle=":", alpha=0.7, label="Current peak (6ms)")
    axes[2].set_title("Synaptic Current (3.0 nA)")
    axes[2].set_ylabel("Current (nA)")
    axes[2].set_xlabel("Time (ms)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("task1.svg", bbox_inches="tight")
    plt.close()
    print("✓ Task 1 plot saved")


def plot_stdp_comparison(pot_data: dict, dep_data: dict) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].eventplot(
        [pot_data["pre_times"]], colors=["blue"], linewidths=6, lineoffsets=-0.2, linelengths=0.4, label="Pre"
    )
    axes[0, 0].eventplot(
        [pot_data["post_times"]], colors=["red"], linewidths=6, lineoffsets=0.2, linelengths=0.4, label="Post"
    )
    axes[0, 0].set_title("Potentiation: Pre→Post")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[1, 0].eventplot(
        [dep_data["post_times"]], colors=["red"], linewidths=6, lineoffsets=0.2, linelengths=0.4, label="Post"
    )
    axes[1, 0].eventplot(
        [dep_data["pre_times"]], colors=["blue"], linewidths=6, lineoffsets=-0.2, linelengths=0.4, label="Pre"
    )
    axes[1, 0].set_title("Depression: Post→Pre")
    axes[1, 0].set_xlabel("Time (ms)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    weights = [2.0, pot_data["stdp_results"]["final_weight"], dep_data["stdp_results"]["final_weight"]]
    conditions = ["Initial", "Potentiation", "Depression"]
    colors = ["gray", "green", "red"]
    bars = axes[0, 1].bar(conditions, weights, color=colors, alpha=0.7)
    axes[0, 1].set_title("Final Weights")
    axes[0, 1].set_ylabel("Weight (nA)")
    for bar, weight in zip(bars, weights, strict=False):
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width() / 2.0, height + 0.01, f"{weight:.4f}", ha="center", va="bottom")

    changes = [0, pot_data["stdp_results"]["weight_change"], dep_data["stdp_results"]["weight_change"]]
    change_bars = axes[1, 1].bar(conditions, changes, color=colors, alpha=0.7)
    axes[1, 1].set_title("Weight Changes (Δw)")
    axes[1, 1].set_ylabel("Change (nA)")
    axes[1, 1].axhline(y=0, color="black", linestyle="-", alpha=0.5)
    axes[1, 1].set_xlabel("Experiment")
    for bar, change in zip(change_bars, changes, strict=False):
        height = bar.get_height()
        axes[1, 1].text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.0005 if height >= 0 else height - 0.001,
            f"{change:.4f}",
            ha="center",
            va="bottom" if height >= 0 else "top",
        )

    dt_range = np.linspace(-10, 10, 200)
    stdp_curve = np.array([stdp_window(dt) for dt in dt_range])
    axes[0, 2].plot(dt_range, stdp_curve, "b-", linewidth=3)
    axes[0, 2].axhline(y=0, color="k", linestyle="-", alpha=0.3)
    axes[0, 2].axvline(x=0, color="k", linestyle="-", alpha=0.3)
    axes[0, 2].scatter([2], [stdp_window(2)], color="green", s=100, label="+2ms")
    axes[0, 2].scatter([-2], [stdp_window(-2)], color="red", s=100, label="-2ms")
    axes[0, 2].set_title("STDP Learning Window")
    axes[0, 2].set_xlabel("Δt (ms)")
    axes[0, 2].set_ylabel("Weight Change")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 2].axis("off")
    summary = f"""STDP Results

Potentiation (+2ms):
  Weight: {pot_data["stdp_results"]["initial_weight"]:.3f} → {pot_data["stdp_results"]["final_weight"]:.4f} nA
  Change: +{pot_data["stdp_results"]["weight_change"]:.4f} nA
  Pairs: {len(pot_data["stdp_results"]["pairs"])}

Depression (-2ms):
  Weight: {dep_data["stdp_results"]["initial_weight"]:.3f} → {dep_data["stdp_results"]["final_weight"]:.4f} nA
  Change: {dep_data["stdp_results"]["weight_change"]:.4f} nA
  Pairs: {len(dep_data["stdp_results"]["pairs"])}
"""
    axes[1, 2].text(
        0.05,
        0.95,
        summary,
        transform=axes[1, 2].transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox={"boxstyle": "round", "facecolor": "lightblue", "alpha": 0.8},
    )

    plt.tight_layout()
    plt.savefig("stdp_analysis.svg", bbox_inches="tight")
    plt.close()
    print("✓ STDP plot saved")


def run_experiments() -> dict:
    experiments = {
        "task1": task1_experiment(),
        "potentiation": stdp_potentiation_experiment(),
        "depression": stdp_depression_experiment(),
    }
    return experiments


def generate_visualisations(experiments: dict) -> dict:
    plot_task1(experiments["task1"])
    plot_stdp_comparison(experiments["potentiation"], experiments["depression"])
    return experiments


def save_results(experiments: dict) -> dict:
    serialisable_data = {
        "task1": {
            "spike_times": experiments["task1"]["results"]["spike_times"],
            "max_voltage": float(torch.max(experiments["task1"]["results"]["voltages"])),
            "max_current": float(torch.max(experiments["task1"]["current"])),
        },
        "potentiation": {
            "initial_weight": experiments["potentiation"]["stdp_results"]["initial_weight"],
            "final_weight": experiments["potentiation"]["stdp_results"]["final_weight"],
            "weight_change": experiments["potentiation"]["stdp_results"]["weight_change"],
        },
        "depression": {
            "initial_weight": experiments["depression"]["stdp_results"]["initial_weight"],
            "final_weight": experiments["depression"]["stdp_results"]["final_weight"],
            "weight_change": experiments["depression"]["stdp_results"]["weight_change"],
        },
        "framework": "snnTorch",
        "python": "3.13+",
    }
    with open("lab_results.json", "w") as f:
        json.dump(serialisable_data, f, indent=2)
    return experiments


def main():
    experiments = run_experiments()
    experiments = generate_visualisations(experiments)
    experiments = save_results(experiments)


if __name__ == "__main__":
    main()
