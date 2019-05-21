from General import *
# import numpy as np


def deutsh(n):
    """Deutsch-Josza algorithm for n qbits, the function is secret and randomly generated"""
    q = QuantumRegister(n, 'q')
    circ = QuantumCircuit(q)

    circ.h(q)
    print(boite_noire(circ, q, n))
    circ.barrier(q)
    circ.h(q)

    circ_m = measure(circ, q, [i for i in range(n)])
    return circ_m


# Tests

circ_m = deutsh(4)
launch(2048, circ_m)

