from General import *
import numpy as np


def grover(n):
    """Return a circuit with a grover algorithm on n qbits, with 11..1 search"""
    q = QuantumRegister(n, 'q')
    circ = QuantumCircuit(q)
    controls = [q[i] for i in range(n - 2, -1, -1)]
    print(controls)

    circ.h(q)

    stop = 1 * (n < 3) + int(np.sqrt(n) + 1) * (n >= 3)

    for _ in range(stop):
        circ.barrier(q)
        circ.h(q[n - 1])
        cnx(circ, q, controls, q[n - 1])
        circ.h(q[n - 1])
        circ.barrier(q)
        circ.h(q)
        circ.x(q)
        circ.h(q[n - 1])
        cnx(circ, q, controls, q[n - 1])
        circ.h(q[n - 1])
        circ.barrier(q)
        circ.x(q)
        circ.h(q)

    circ_m = measure(circ, q, [i for i in range(n)])
    return circ_m


def better_grover(n):
    """Return a circuit with a grover algorithm on n qbits, with secret search"""
    q = QuantumRegister(n, 'q')
    circ = QuantumCircuit(q)
    controls = [q[i] for i in range(n - 2, -1, -1)]
    secret = bina(n)  # [ [0,1,2], [0,1], [0,2], [0], [1,2], [1], [2], [] ]
    r.shuffle(secret)
    search = secret[0]
    print(search)
    circ.h(q)

    stop = 1 * (n < 3) + int(np.sqrt(n) + 1) * (n >= 3)

    for _ in range(stop):
        oracle(circ, q, n, search)
        circ.barrier(q)
        circ.h(q)
        circ.x(q)
        circ.h(q[n - 1])
        cnx(circ, q, controls, q[n - 1])
        circ.h(q[n - 1])
        circ.barrier(q)
        circ.x(q)
        circ.h(q)

    circ_m = measure(circ, q, [i for i in range(n)])
    return circ_m


# We try it here


circ_m = grover(5)
launch(2048, circ_m)
# circ_m.draw(output='mpl', plot_barriers = False)

# We try it here
circ_m = better_grover(7)
launch(2048, circ_m)
