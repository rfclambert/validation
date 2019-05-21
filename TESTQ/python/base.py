from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import execute
from qiskit import BasicAer
from qiskit.tools.visualization import plot_state_city
from qiskit.tools.visualization import plot_histogram

#Grover (with control bit)
init = QuantumRegister(3, 'init')
circinit = QuantumCircuit(init)

#Initialization
circinit.h(init[0])
circinit.h(init[1])
circinit.x(init[2])
circinit.h(init[2])

#Grover Operator
gro = QuantumRegister(3, 'gro')
circGO = QuantumCircuit(init)
#Oracle is just Toffoli so we search 11
circGO.ccx(init[0], init[1], init[2])
#Mean-Flip circuit
circGO.h(init[0])
circGO.h(init[1])
circGO.x(init[0])
circGO.x(init[1])
circGO.h(init[1])
circGO.cx(init[0],init[1])
circGO.h(init[1])
circGO.barrier(init)
circGO.x(init[0])
circGO.x(init[1])
circGO.h(init[0])
circGO.h(init[1])

#We can concatenate two circuits with the same registers
qc = circinit+circGO

#The grover operator with initialization
qc.draw(output='mpl', plot_barriers = False)

#We can choose our backend (here the simulator for the states)
backend = BasicAer.get_backend('statevector_simulator')
job = execute(qc, backend)

#The result here is the complete superposition, so we can plot it that way.
result = job.result()
outputstate = result.get_statevector(qc, decimals = 3)
print(outputstate)

plot_state_city(outputstate)

# We want to measure it so we create a classical register
c = ClassicalRegister(2, 'c')

# Create a Quantum Circuit
meas = QuantumCircuit(init, c)
meas.barrier(init)

# map the quantum measurement to the classical bits
meas.measure(init[0],c[0])
meas.measure(init[1],c[1])

# We add the measure part to the end of the circuit
qc_m = qc+meas

#drawing the circuit
qc_m.draw(output='mpl', plot_barriers = False)

# We want to use the simulator for the measures
backend_sim = BasicAer.get_backend('qasm_simulator')

# Because it can be random, we can choose the number of shots we want to do
job_sim = execute(qc_m, backend_sim, shots=1024)

# Here the results are measures not states
result_sim = job_sim.result()

counts = result_sim.get_counts(qc_m)
print(counts)

plot_histogram(counts)
