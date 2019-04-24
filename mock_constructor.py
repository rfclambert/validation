from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
import numpy as np


def constructor_function(x, qr, inverse=False, depth=2):
    """A mock constructor function to test the class,
    it only places H and u1 gates"""
    qc = QuantumCircuit(qr)
    for _ in range(depth):
        qc.h(qr)
        for i in range(len(x)):
            qc.u1(x[i], qr[i])
    return qc
