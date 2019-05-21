from General import *
# import numpy as np


def bernsteinv(n, nbr):
    q = QuantumRegister(n, 'q')
    circ = QuantumCircuit(q)
    circ.h(q)
    print(oracle_BV(circ, q, n, nbr))
    circ.barrier(q)
    circ.h(q)

    circ_m = measure(circ, q, [i for i in range(n)])
    return circ_m


def bernsteinv_o(n, nbr):
    q = QuantumRegister(n + 1, 'q')
    circ = QuantumCircuit(q)
    for i in range(n):
        circ.h(q[i])
    print(oracle_BV_o(circ, q, n, nbr))
    circ.barrier(q)
    for i in range(n):
        circ.h(q[i])
    # circ.h(q)

    circ_m = measure(circ, q, [i for i in range(n)])
    return circ_m


# Tests
circ_m = bernsteinv_o(4, 5)
launch2(circ_m)
launch(2048, circ_m)

circ_m = bernsteinv(4, 5)
launch2(circ_m)
launch(2048, circ_m)
