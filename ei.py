import matplotlib.pyplot as plt
import pyNN.brian2 as sim

# import pyNN.spiNNaker as sim
import pyNN.utility.plotting as plot
from pyNN.random import RandomDistribution

n_neurons = 1000
n_exc = int(0.8 * n_neurons)
n_inh = int(0.2 * n_neurons)
simtime = 1000

sim.setup(timestep=1.0)
# sim.setup(timestep=0.1, **{"simulator": "nest"})


pop_exc = sim.Population(n_exc, sim.IF_curr_exp(), label="Excitatory")
pop_inh = sim.Population(n_inh, sim.IF_curr_exp(), label="Inhibitory")

stim_exc = sim.Population(n_exc, sim.SpikeSourcePoisson(rate=1000.0), label="Stim_Exc")
stim_inh = sim.Population(n_inh, sim.SpikeSourcePoisson(rate=1000.0), label="Stim_Inh")

synapse_exc = sim.StaticSynapse(
    weight=sim.RandomDistribution("normal_clipped", mu=0.1, sigma=0.1, low=0.0, high=0.2),
    delay=sim.RandomDistribution("normal_clipped", mu=1.5, sigma=0.75, low=0.1, high=14.4),
)

synapse_inh = sim.StaticSynapse(
    weight=sim.RandomDistribution("normal_clipped", mu=-0.4, sigma=0.1, low=-0.5, high=0.0),
    delay=sim.RandomDistribution("normal_clipped", mu=0.75, sigma=0.375, low=0.1, high=14.4),
)

sim.Projection(stim_exc, pop_exc, sim.OneToOneConnector(), synapse_type=synapse_exc)
sim.Projection(stim_inh, pop_inh, sim.OneToOneConnector(), synapse_type=synapse_exc)
sim.Projection(pop_exc, pop_inh, sim.FixedProbabilityConnector(p_connect=0.1), synapse_type=synapse_exc)
sim.Projection(pop_inh, pop_exc, sim.FixedProbabilityConnector(p_connect=0.1), synapse_type=synapse_inh)

pop_exc.initialize(v=RandomDistribution("uniform", low=-65.0, high=-55.0))
pop_inh.initialize(v=RandomDistribution("uniform", low=-65.0, high=-55.0))

pop_exc.record("spikes")

sim.run(simtime)

spikes = pop_exc.get_data("spikes").segments[0].spiketrains
plot.Figure(
    plot.Panel(spikes, yticks=True, markersize=1, xlim=(0, simtime)),
    title="Asynchronous Irregular Spiking Activity",
)
plt.show()

sim.end()
