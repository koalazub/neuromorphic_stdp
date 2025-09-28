import brian2
import matplotlib.pyplot as plt
import numpy as np
import pyNN.brian2 as sim
from pyNN.parameters import Sequence

brian2.prefs.codegen.target = "numpy"
brian2.prefs.core.default_float_dtype = np.float64


dt = 0.1  # Use 0.1ms timestep (100 μs) - matches Brian2's internal expectations
min_delay = 1.0  # Set minimum delay to multiple of dt

sim.setup(
    timestep=dt,
    min_delay=min_delay,
    max_delay=10.0,
    threads=1,  # Single thread for deterministic results
    spike_precision="off_grid",  # Allow off-grid spike times
)

print(f"PyNN setup complete: dt={dt}ms, min_delay={min_delay}ms")


neuron_params = {
    "tau_m": 20.0,  # membrane time constant (ms)
    "tau_refrac": 2.0,  # refractory period (ms)
    "v_rest": -65.0,  # resting potential (mV)
    "v_reset": -70.0,  # reset potential (mV)
    "v_thresh": -50.0,  # threshold potential (mV)
    "cm": 1.0,  # membrane capacitance (nF)
    "tau_syn_E": 5.0,  # excitatory synaptic time constant (ms)
    "tau_syn_I": 5.0,  # inhibitory synaptic time constant (ms)
    "i_offset": 0.0,  # offset current (nA)
}


print("Creating neural populations...")

input_spike_times = Sequence([5.0])
input_source = sim.Population(1, sim.SpikeSourceArray(spike_times=input_spike_times))

lif_neuron = sim.Population(1, sim.IF_curr_exp(**neuron_params))


print("Setting up synaptic connections...")

delay_value = min_delay  # 1.0 ms = 10 timesteps at 0.1ms
weight_value = 30.0  # Strong enough to cause threshold crossing

synapse = sim.StaticSynapse(weight=weight_value, delay=delay_value)

connection = sim.Projection(
    input_source,
    lif_neuron,
    sim.AllToAllConnector(),
    synapse_type=synapse,
    receptor_type="excitatory",
)


print("Configuring data recording...")
input_source.record("spikes")
lif_neuron.record(["spikes", "v"])


simtime = 20.0
print(f"Running simulation for {simtime} ms...")
sim.run(simtime)
print("Simulation completed successfully!")


print("Extracting simulation data...")

input_data = input_source.get_data().segments[0]
input_spike_times_recorded = input_data.spiketrains[0].times if input_data.spiketrains else []

output_data = lif_neuron.get_data().segments[0]
output_spike_times = output_data.spiketrains[0].times if output_data.spiketrains else []

membrane_potential = output_data.analogsignals[0]
time_axis = np.array(membrane_potential.times)
voltage = np.array(membrane_potential).flatten()

tau_m = neuron_params["tau_m"]
v_rest = neuron_params["v_rest"]
cm = neuron_params["cm"]

dt_calc = time_axis[1] - time_axis[0] if len(time_axis) > 1 else dt / 1000
dv_dt = np.gradient(voltage, dt_calc)
synaptic_current = cm * (dv_dt + (voltage - v_rest) / tau_m)

threshold_crossings = []
for i in range(1, len(voltage)):
    if voltage[i - 1] < neuron_params["v_thresh"] and voltage[i] >= neuron_params["v_thresh"]:
        threshold_crossings.append((time_axis[i], voltage[i]))

print(f"Data extraction complete. Found {len(threshold_crossings)} threshold crossings.")


print("Generating visualization...")

fig = plt.figure(figsize=(14, 12))
gs = fig.add_gridspec(4, 1, height_ratios=[1, 2, 2, 1.5], hspace=0.4, top=0.95, bottom=0.08, left=0.08, right=0.78)

ax1 = fig.add_subplot(gs[0])
if len(input_spike_times_recorded) > 0:
    ax1.eventplot(
        [input_spike_times_recorded],
        colors=["orange"],
        linewidths=4,
        linelengths=0.4,
        lineoffsets=-0.3,
        label="Input Spikes",
    )
    for spike_t in input_spike_times_recorded:
        ax1.text(spike_t, -0.3, f"{float(spike_t):.1f}ms", ha="center", va="top", fontsize=8, color="orange")

if len(output_spike_times) > 0:
    ax1.eventplot(
        [output_spike_times], colors=["red"], linewidths=4, linelengths=0.4, lineoffsets=0.3, label="Output Spikes"
    )
    for spike_t in output_spike_times:
        ax1.text(spike_t, 0.3, f"{float(spike_t):.1f}ms", ha="center", va="bottom", fontsize=8, color="red")

ax1.set_title(f"Input/Output Spike Comparison - {len(output_spike_times)} output spike(s)", fontsize=12, pad=10)
ax1.set_ylabel("Spikes", fontsize=11)
ax1.set_xlim(0, simtime)
ax1.set_ylim(-0.8, 0.8)
ax1.grid(True, alpha=0.3)
ax1.legend(loc="upper right", fontsize=9)

ax2 = fig.add_subplot(gs[1])
ax2.plot(time_axis, voltage, "b-", linewidth=2, label="Membrane Potential")
ax2.axhline(y=neuron_params["v_thresh"], color="r", linestyle="--", linewidth=2, alpha=0.8, label="Threshold (-50mV)")
ax2.axhline(y=neuron_params["v_rest"], color="g", linestyle="--", linewidth=2, alpha=0.8, label="Rest (-65mV)")
ax2.axhline(y=neuron_params["v_reset"], color="orange", linestyle="--", linewidth=2, alpha=0.8, label="Reset (-70mV)")

for i, (cross_time, cross_voltage) in enumerate(threshold_crossings):
    ax2.plot(
        cross_time,
        cross_voltage,
        "ro",
        markersize=10,
        markeredgecolor="black",
        markeredgewidth=1,
        label="Threshold Crossing" if i == 0 else "",
    )

if len(output_spike_times) > 0:
    for i, spike_t in enumerate(output_spike_times):
        ax2.axvline(
            x=spike_t, color="red", linestyle=":", linewidth=2, alpha=0.7, label="Spike Event" if i == 0 else ""
        )

max_voltage = np.max(voltage)
max_time = time_axis[np.argmax(voltage)]
ax2.plot(
    max_time,
    max_voltage,
    "go",
    markersize=12,
    markeredgecolor="black",
    markeredgewidth=1,
    label=f"Peak: {max_voltage:.1f}mV",
)

ax2.set_ylabel("Voltage (mV)", fontsize=11)
ax2.set_title("Membrane Potential - Full View", fontsize=12, pad=15)
ax2.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, fontsize=10)
ax2.set_xlim(0, simtime)
ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(gs[2])
if len(output_spike_times) > 0:
    spike_time = float(output_spike_times[0])
    zoom_start = max(0, spike_time - 3)
    zoom_end = min(simtime, spike_time + 4)

    zoom_indices = (time_axis >= zoom_start) & (time_axis <= zoom_end)
    zoom_time = time_axis[zoom_indices]
    zoom_voltage = voltage[zoom_indices]

    ax3.plot(zoom_time, zoom_voltage, "b-", linewidth=3, label="Membrane Potential")
    ax3.axhline(y=neuron_params["v_thresh"], color="r", linestyle="--", linewidth=2, alpha=0.8, label="Threshold")
    ax3.axhline(y=neuron_params["v_rest"], color="g", linestyle="--", linewidth=2, alpha=0.8, label="Rest")
    ax3.axhline(y=neuron_params["v_reset"], color="orange", linestyle="--", linewidth=2, alpha=0.8, label="Reset")

    for cross_time, cross_voltage in threshold_crossings:
        if zoom_start <= cross_time <= zoom_end:
            ax3.plot(
                cross_time,
                cross_voltage,
                "ro",
                markersize=15,
                markeredgecolor="black",
                markeredgewidth=2,
                label="Threshold Crossing",
            )

    ax3.axvline(x=spike_time, color="red", linestyle=":", linewidth=3, alpha=0.8, label="Spike Event")
    ax3.axhspan(neuron_params["v_thresh"] - 0.5, neuron_params["v_thresh"] + 0.5, alpha=0.15, color="red")

    ax3.set_ylabel("Voltage (mV)", fontsize=11)
    ax3.set_title(f"Zoomed View: Threshold Crossing at t={spike_time:.2f}ms", fontsize=12, pad=15)
    ax3.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax3.set_xlim(zoom_start, zoom_end)
    ax3.grid(True, alpha=0.3)

    textstr = "RED CIRCLE:\nExact threshold\ncrossing point"
    props = {"boxstyle": "round", "facecolor": "pink", "alpha": 0.9, "edgecolor": "red"}
    ax3.text(
        0.02,
        0.98,
        textstr,
        transform=ax3.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=props,
        fontweight="bold",
    )
else:
    ax3.text(simtime / 2, -60, "No spikes to zoom into", ha="center", va="center", fontsize=12, color="red")
    ax3.set_title("Zoomed View - No Spikes Detected", fontsize=12, pad=15)
    ax3.set_xlim(0, simtime)
    ax3.grid(True, alpha=0.3)

ax4 = fig.add_subplot(gs[3])
ax4.plot(time_axis, synaptic_current, "g-", linewidth=2, label="Synaptic Current")
if len(input_spike_times_recorded) > 0:
    for spike_t in input_spike_times_recorded:
        ax4.axvline(
            x=spike_t,
            color="orange",
            linestyle="--",
            linewidth=3,
            alpha=0.8,
            label="Input Spike" if spike_t == input_spike_times_recorded[0] else "",
        )

ax4.set_ylabel("Current (nA)", fontsize=11)
ax4.set_xlabel("Time (ms)", fontsize=11)
ax4.set_title("Calculated Synaptic Current", fontsize=12, pad=15)
ax4.set_xlim(0, simtime)
ax4.grid(True, alpha=0.3)
ax4.legend(loc="upper right", fontsize=10, framealpha=0.9)

plt.savefig("sdtp.svg", bbox_inches="tight")
plt.show()

print("\nBrian2 Backend Configuration:")
print(f"  Codegen target: {brian2.prefs.codegen.target}")
print(f"  Float precision: {brian2.prefs.core.default_float_dtype}")
print(f"  Timestep: {dt} ms (100 μs)")
print(f"  Min delay: {min_delay} ms")
print("  Spike precision: off_grid")

print("\nSimulation Parameters:")
print(f"  Synaptic weight: {weight_value} nA")
print(f"  Synaptic delay: {delay_value} ms")
print(f"  Simulation time: {simtime} ms")

print("\nNeuron Parameters:")
for param, value in neuron_params.items():
    print(f"  {param}: {value}")

print("\nResults:")
print(f"  Input spikes recorded: {len(input_spike_times_recorded)}")
if len(input_spike_times_recorded) > 0:
    print(f"  Input spike times: {[f'{float(t):.3f}' for t in input_spike_times_recorded]} ms")

print(f"  Output spikes: {len(output_spike_times)}")
if len(output_spike_times) > 0:
    print(f"  Output spike times: {[f'{float(t):.3f}' for t in output_spike_times]} ms")

    input_time = float(input_spike_times_recorded[0]) if len(input_spike_times_recorded) > 0 else 5.0
    output_time = float(output_spike_times[0])
    total_delay = output_time - input_time
    print(f"  Total delay (input→output): {total_delay:.3f} ms")

print(f"  Maximum voltage: {max_voltage:.3f} mV")
print(f"  Threshold crossings detected: {len(threshold_crossings)}")

if len(threshold_crossings) > 0:
    for i, (cross_time, cross_voltage) in enumerate(threshold_crossings):
        print(f"  Crossing {i + 1}: t={cross_time:.3f}ms, V={cross_voltage:.3f}mV")

if max_voltage >= neuron_params["v_thresh"]:
    print("  ✓ SUCCESS: Visible threshold crossing achieved!")
    overshoot = max_voltage - neuron_params["v_thresh"]
    print(f"    Overshoot: {overshoot:.3f} mV")
else:
    print("  ✗ No threshold crossing detected")
    shortfall = neuron_params["v_thresh"] - max_voltage
    print(f"    Shortfall: {shortfall:.3f} mV - increase weight")

sim.end()
