from General import *
import numpy as np
from qiskit import IBMQ
IBMQ.load_accounts()


def shor_quant(A, N, maxQ):
    """Builds the quantum circuit for the period estimation part of shor"""
    binn = [int(x) for x in bin(N)[2:]] # The binlist of N
    binn.reverse()
    bits_N = len(binn)  # number of bits of N
    bits_X = maxQ - (4 * bits_N + 3) - (maxQ - (4*bits_N + 3)) % 2  # number of bits of X, must be even
    print("Qbits for N:", bits_N)
    print("Qbits for X:", bits_X)
    if bits_X < 1 * bits_N:  # here we don't want to send requests when the result won't be satisfying
        print("Not enough qbits")
        return 0
    print("Launching quantum procedure...")
    # Dans l'ordre:X(m)|Y(n+1)|N(n)|A(n)|APOW(n)|lost|lost2
    # nombre total de qbits: 7*n+3+1*(3*n%2)

    nX = bits_X  # q[nX] debut de Y
    nXY = nX + bits_N + 1  # q[nXY] debut de N
    nXYN = nXY + bits_N  # q[nXYN] debut de A
    nXYNA = nXYN + bits_N  # q[nXYNA] debut de APOW
    n = nXYNA + bits_N  # Total de qbit chargés

    q = QuantumRegister(n + 2, 'q')  # +lost+lost2=+2
    circ = QuantumCircuit(q)

    # We have all the quantum registers of qbits.

    RegX = [q[i] for i in range(bits_X)]
    RegY = [q[i + nX] for i in range(nXY - nX)]
    RegN = [q[i + nXY] for i in range(nXYN - nXY)]
    RegA = [q[i + nXYN] for i in range(nXYNA - nXYN)]
    RegAPOW = [q[i + nXYNA] for i in range(n - nXYNA)]

    # Uniform superposition on X
    for r in RegX:
        circ.h(r)
    # Setting of N
    for r in [RegN[i] for i in range(bits_N) if binn[i]]:
        circ.x(r)  # Set N

    # Main call: the modular exponentiation

    expmod(circ, q,
           RegX,  # X
           A,  # a
           RegA,  # A
           RegAPOW,  # APOW
           RegY,  # Y
           N,  # n
           RegN,  # N
           binn,  # binn
           q[n],  # lost
           q[n + 1])  # lost2

    # Comme bits_X est pair, le resultat A**x%n est dans RegAPOW

    # Reset of N to 0 (useless here)
    for r in [RegN[i] for i in range(bits_N) if binn[i]]:
        circ.x(r)  # Reset N

    QFTn(circ, q, RegX)  # Fourier transform de X
    circ_m = measure_direct(circ, q, RegX + RegY)

    return circ_m


def periodfinder(A, n):
    """The algorithm to find the period in the modular exponentiation. Makes a call to the quantum algorithm but
    can switch back to classical algorithm if it fails"""
    prop = qperiodfinder(A, n)  # Quantum period finder, can fail
    if prop:  # No apriori failure
        res = prop
        Aprop = powmod(A, prop, n)  # Eventuellement on ne fait pas cette verification et on test tout à la fin
        fun = Aprop
        if fun != 1:
            print("Quantum process failed, surely because k|L. Back to classical.")
        while fun != 1:  # Maybe infinite loop, if quantum completely failed
            fun = (fun*Aprop) % n
            res += prop
        return res
    # Quantum process failed, classical algorithm
    res = 1
    fun = A
    while fun != 1:
        fun = (fun * A) % n
        res += 1
    return res


def qperiodfinder(A, n):
    """The quantum algorithm to find the period in the modular exponentiation"""
    maxQ = 25  # Here we set the number of qbits we have available
    circ_m = shor_quant(A, n, maxQ)  # The quantum circuit is generated
    if circ_m is 0:
        print("Quantum circuit generation failed. Classical circuit engaged")
        return 0
    print("Quantum circuit generated...")
    print("Total Qbit usage:{}, Total depth: {}".format(sum(reg.size for reg in circ_m.qregs), circ_m.depth()))
    #Call to the simulator
    name = 'ibmq_qasm_simulator'
    backend_sim = IBMQ.get_backend(name, hub=None)  # BasicAer.get_backend('qasm_simulator')
    print("Sending quantum circuit to hardware "+name+", waiting for the result...")
    job_sim = execute(circ_m, backend_sim, shots=1, max_credits=3)
    result_sim = job_sim.result()
    print("Result get!")
    counts = result_sim.get_counts(circ_m)
    print(counts, type(counts))
    bits_N = len(bin(n)) - 2
    bits_X = maxQ - 4 * (bits_N + 1) + (maxQ - 4 * (bits_N + 1)) % 2
    for key in counts:
        print(key)
        print(type(key))
        S = int('0b' + key[bits_N:], 2)  # S = int(k*2**bits_X/L), we want to find L with it. It can fail if k|L
        # here N is not the number we ant to factorize, it's 2**bits_X, the "magnitude" of our period fider algorithm
        N = 2 ** bits_X
        print("S:", S, "N:", N)
        # Call to the fraction finder, given N and S = k*N/L, will return (k, L), as a simplified fraction.
        k, L = fracf(N, S)
        print(k, L)
    return L


def shor(n):
    """The global algorithm to find the factorization of n into its prime components."""
    print("Starting...")
    # A random integer is picked
    A = 7#r.randrange(2, n)
    print("A:", A)
    # If we're lucky it's already the prime number or a multiple of it
    res = np.gcd(A, n)
    if res != 1:
        print("Lucky find!")
        return res, n // res
    # DEBUT PROCEDURE QUANTIQUE
    res = periodfinder(A, n)
    print("Res:", res)
    # fin
    if res % 2:  # Unlucky, we have to pick an other A and redo everything
        print("Period not even...")
        return shor(n)
    else:
        L = res // 2
        R = powmod(A, L, n)
        if R == 1 or R + 1 == n:
            print("Trivial root...")
            return shor(n)
        P = np.gcd(R - 1, n)
        print("Result found!")
        return P, n // P


print(shor(15))
