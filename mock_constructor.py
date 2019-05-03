from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
import numpy as np

def constructor_function(x, qr, inverse=False, depth=2, entangler_map=None):
    """A mock constructor function to test the CustomExpansion class.
    
    Args:
        x (numpy.ndarray): 1D to-be-transformed data
        qr (QuantumRegister)
        inverse (bool): whether or not to invert the circuit
        depth (int): number of times to repeat circuit
        entangler_map (dict): describe the connectivity of qubits
    
    Returns:
        qc (QuantumCircuit): layers of Rx gates interleaved with ZZ gates
    """
    
    if entangler_map is None:
        entangler_map = {i: [j for j in range(i, len(x)) if j != i] for i in range(len(x) - 1)}
    
    qc = QuantumCircuit(qr)

    for _ in range(depth):
        for i in range(len(x)):
            qc.rx(x[i], qr[i])
        for source in entangler_map:
            for target in entangler_map[source]:
                qc.cx(qr[source], qr[target])
                qc.u1(x[source] * x[target], qr[target])
                qc.cx(qr[source], qr[target])
    return qc