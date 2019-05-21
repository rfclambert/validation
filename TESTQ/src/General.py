from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import execute
from qiskit import BasicAer
import numpy as np
from qiskit.tools.visualization import plot_histogram
from qiskit.providers.aer import UnitarySimulator

import random as r

#C:\Users\RaphaelLambert\AppData\Local\Packages\CanonicalGroupLimited.UbuntuonWindows_79rhkp1fndgsc\LocalState\rootfs

# Then we translate matrix into gates for Qiskit
def rootx(circuit, reg, target):
    """Adds a SQRT(X) gate to circ, on reg at qbit target"""
    circuit.u1(np.pi / 4, target)
    circuit.x(target)
    circuit.u1(np.pi / 4, target)
    circuit.x(target)
    circuit.u3(np.pi / 2, -np.pi / 2, np.pi / 2, target)
    return 1


def crootx(circuit, reg, control, target):
    """Adds a Control-SQRT(X) gate to circ, on reg controlled by control at qbit target"""
    circuit.cu1(np.pi / 4, control, target)
    circuit.cx(control, target)
    circuit.cu1(np.pi / 4, control, target)
    circuit.cx(control, target)
    circuit.cu3(np.pi / 2, -np.pi / 2, np.pi / 2, control, target)
    return 1


def crootnx(circuit, reg, control, target, n, d):
    """Adds a Control-n-th-root(X) gate to circuit, on reg, controlled by control at qbit target, dagger if d"""
    if d:
        circuit.cu1(-np.pi / (2 * n), control, target)
        circuit.cx(control, target)
        circuit.cu1(-np.pi / (2 * n), control, target)
        circuit.cx(control, target)
        circuit.cu3(np.pi / n, np.pi / 2, -np.pi / 2, control, target)
    else:
        circuit.cu1(np.pi / (2 * n), control, target)
        circuit.cx(control, target)
        circuit.cu1(np.pi / (2 * n), control, target)
        circuit.cx(control, target)
        circuit.cu3(np.pi / n, -np.pi / 2, np.pi / 2, control, target)
    return 1


def cnrootnx(c, r, cont, target, n, d=False):
    """Adds a len(cont)-qbits-Control-n-th-root(X) gate to c, on r, controlled by the List cont at qbit target,
    dagger if d"""
    if len(cont) == 1:
        crootnx(c, r, cont[0], target, n, d)
    else:
        cnrootnx(c, r, [cont[0]], target, 2 * n, d)
        cnx(c, r, cont[1:], cont[0])  # 2?
        cnrootnx(c, r, [cont[0]], target, 2 * n, not d)
        cnx(c, r, cont[1:], cont[0])  # 2?
        cnrootnx(c, r, cont[1:], target, 2 * n, d)
    return 1


def cnx(c, r, cont, target):
    """Adds a len(cont)-qbits-Control(X) gate to c, on r, controlled by the List cont at qbit target"""
    #c.barrier(r)
    if len(cont) == 1:
        c.cx(cont[0], target)
    elif len(cont) == 2:
        c.ccx(cont[0], cont[1], target)
    else:
        cnrootnx(c, r, [cont[0]], target, 2)
        cnx(c, r, cont[1:], cont[0])  # 2?
        cnrootnx(c, r, [cont[0]], target, 2, True)
        cnx(c, r, cont[1:], cont[0])  # 2?
        cnrootnx(c, r, cont[1:], target, 2)
    return 1


def cnxd(c, r, cont, target):
    """Adds a len(cont)-qbits-Control(X) dagger before gate to c, on r, controlled by the List cont at qbit target"""
    if len(cont) == 1:
        c.cx(cont[0], target)
    elif len(cont) == 2:
        c.ccx(cont[0], cont[1], target)
    else:
        cnrootnx(c, r, cont[:-1], target, 2, True)
        cnx(c, r, cont[1:], cont[0])  # 2?
        cnrootnx(c, r, cont[:-1], target, 2)
        cnx(c, r, cont[1:], cont[0])  # 2?
        cnrootnx(c, r, cont[1:], target, 2, True)
    return 1


# Then we have some quality of life function to create circuits.
def measure(circ, reg, targ):
    """Returns circ with measures added on reg on each qbit index specified in targ"""
    c = ClassicalRegister(len(targ), 'c')
    meas = QuantumCircuit(reg, c)
    circ.barrier(reg)
    for i in range(len(targ)):
        meas.measure(reg[targ[i]], c[i])  # ici
    return circ + meas


def launch(n, circ):
    """Create a backend and launch circ n times on it"""
    backend_sim = BasicAer.get_backend('qasm_simulator')
    job_sim = execute(circ, backend_sim, shots=n)
    result_sim = job_sim.result()
    counts = result_sim.get_counts(circ)
    #print(counts)
    # plot_histogram(counts)
    return counts


def launch2(circ):
    """Create a backend and return the state vectors"""
    backend = BasicAer.get_backend('statevector_simulator')
    job = execute(circ, backend)
    result = job.result()
    outputstate = result.get_statevector(circ, decimals=3)
    print(outputstate)
    return outputstate


def affiche(mat):
    for ligne in mat:
        a = ''
        for case in ligne:
            a+=str(np.around(case.real,3))+' '
        print(a)


def launch3(circ):
    """Create a backend and return the state vectors"""
    backend = BasicAer.get_backend('unitary_simulator')
    job = execute(circ, backend)
    result = job.result()
    outputuni = result.get_unitary(circ, decimals=3)
    print(outputuni)
    affiche(outputuni)
    return outputuni


def plot(circ):
    """Plots the result (doesn't work?)"""
    circ.draw(output='mpl')


def limite_prec(n):
    """Test for the precision of cnx"""
    q = QuantumRegister(n, 'q')
    circ = QuantumCircuit(q)
    controls = [q[i] for i in range(n - 2, -1, -1)]
    print(controls)
    circ.x(q)
    cnx(circ, q, controls, q[n - 1])
    circ_m = measure(circ, q, [i for i in range(n)])
    return circ_m


def oracle(circ, reg, n, bi):
    """Oracle for the binary bi"""
    for i in bi:
        circ.x(reg[i])
    circ.h(reg[n - 1])
    cnx(circ, reg, [reg[i] for i in range(n - 2, -1, -1)], reg[n - 1])
    circ.h(reg[n - 1])
    for i in bi:
        circ.x(reg[i])


def bina(n):
    """List of all binary in the form of a list of list of index.
     Ex: bina(3) = [[2, 1, 0], [2, 1], [2, 0], [2], [1, 0], [1], [0], []]"""
    if n == 1:
        return [[0], []]
    b = bina(n - 1)
    return [[n - 1] + i for i in b] + b


def boite_noire(circ, q, n):
    """The secret function for Deutsch Josza"""
    secret = bina(n)  # See the example in the doc for more details
    r.shuffle(secret)
    if r.random() < 0.5:
        return 0
    else:
        for i in range(2 ** (n - 1)):
            print(secret[i])
            oracle(circ, q, n, secret[i])
        return 1


def oracle_BV(circ, q, n, nbr):
    """Creates an oracle for the Bernstein Vazirani algorithm (it's just a succession of oracles)"""
    binary = bina(n)
    for i in range(2 ** n):
        if bin(i & nbr).count('1') % 2:
            oracle(circ, q, n, binary[i])
    return 1


def cnx_o_paquet(c, r, cont, target, losts):
    """Adds a len(cont)-qbits-Control(X) gate to c, on r, controlled by the List cont at qbit target,
    losing one qbit in the process"""
    if len(cont) == 1:
        c.cx(cont[0], target)
    elif len(cont) == 2:
        c.ccx(cont[0], cont[1], target)
    elif len(cont) == 3:  # and len(losts) == 1:
        c.ccx(cont[2], cont[1], losts[0])
        c.ccx(cont[0], losts[0], target)
        c.ccx(cont[2], cont[1], losts[0])
        c.ccx(cont[0], losts[0], target)
    else:
        m = len(cont)
        c.ccx(cont[0], losts[0], target)

        for i in range(1, m - 2):
            c.ccx(cont[i], losts[i], losts[i - 1])
        c.ccx(cont[-1], cont[-2], losts[m - 3])
        for i in range(m - 3, 0, -1):
            c.ccx(cont[i], losts[i], losts[i - 1])
        c.ccx(cont[0], losts[0], target)
        for i in range(1, m - 2):
            c.ccx(cont[i], losts[i], losts[i - 1])
        c.ccx(cont[-1], cont[-2], losts[m - 3])
        for i in range(m - 3, 0, -1):
            c.ccx(cont[i], losts[i], losts[i - 1])
    return 1


def cnx_o(c, r, cont, target, lost):
    """Adds a len(cont)-qbits-Control(X) gate to c, on r, controlled by the List cont at qbit target,
    losing one qbit in the process"""
    if len(cont) == 1:
        c.cx(cont[0], target)
    elif len(cont) == 2:
        c.ccx(cont[0], cont[1], target)
    elif len(cont) == 3:
        c.ccx(cont[2], cont[1], lost)
        c.ccx(cont[0], lost, target)
        c.ccx(cont[2], cont[1], lost)
        c.ccx(cont[0], lost, target)
    else:
        m = int(np.ceil(len(cont) / 2 + 1))
        m1 = len(cont) - m
        # A more efficient way to do this would be defining a new circuit
        # and just multiply it instead of doing the same thing twice
        # but we don't really care about that, it gives the same quantum circuit at the end
        cnx_o_paquet(c, r, cont[m1:], lost, [target] + cont[:m1])
        cnx_o_paquet(c, r, [lost] + cont[:m1], target, cont[m1:])
        cnx_o_paquet(c, r, cont[m1:], lost, [target] + cont[:m1])
        cnx_o_paquet(c, r, [lost] + cont[:m1], target, cont[m1:])
    return 1


def oracle_o(circ, reg, n, bi):
    """Oracle for the binary bi, optimized to the cost of the last qbit of the list reg of qbits"""
    for i in bi:
        circ.x(reg[i])
    circ.h(reg[n-1])
    cnx_o(circ, reg, [reg[i] for i in range(n-2,-1,-1)], reg[n-1], reg[n])
    circ.h(reg[n-1])
    for i in bi:
        circ.x(reg[i])


def oracle_BV_o(circ, q, n, nbr):
    """Oracle of BV for the binary bi, optimized to the cost of the last qbit of the list reg of qbits"""
    binary = bina(n)
    for i in range(2**n):
        if bin(i&nbr).count('1')%2:
            oracle_o(circ, q, n, binary[i])
    return 1


def powmod(a, p, n):
    """A**p%n, done in an inefficient way"""
    res = 1
    for _ in range(p):
        res = (res * a) % n
    return res


def inversemod(a, n):
    """Calculates b, a*b = 1 mod N"""
    b = n
    a, sa = abs(a), -1 if a < 0 else 1
    b, sb = abs(b), -1 if b < 0 else 1
    vv, uu, v, u = 1, 0, 0, 1
    e = 1
    q, rr = divmod(a, b)
    while rr:
        a, b = b, rr
        vv, v = q * vv + v, vv
        uu, u = q * uu + u, uu
        e = -e
        q, rr = divmod(a, b)
    res = -sa * e * uu
    return res if res > 0 else n + res


def measure_direct(circ, reg, targ):
    """Returns circ with measures added on reg on each qbit index specified in targ"""
    c = ClassicalRegister(len(targ), 'c')
    meas = QuantumCircuit(reg, c)
    circ.barrier(reg)
    for i in range(len(targ)):
        meas.measure(targ[i], c[i])
    return circ + meas


def MAJ(circ, q, c, b, a):
    """A part of the adder"""
    circ.cx(a, b)
    circ.cx(a, c)
    circ.ccx(b, c, a)


def JAM(circ, q, c, b, a):
    """A part of the subber"""
    circ.ccx(b, c, a)
    circ.cx(a, c)
    circ.cx(a, b)


def UMA(circ, q, c, b, a):
    """A part of the adder"""
    circ.ccx(b, c, a)
    circ.cx(a, c)
    circ.cx(c, b)


def AMU(circ, q, c, b, a):
    """A part of the subber"""
    circ.cx(c, b)
    circ.cx(a, c)
    circ.ccx(b, c, a)


def add(circ, q, A, B, lost, last):
    """A and B are lists of the bits of a and b to be added, lost is ancilla.
     A->A B->A+B. Last is the eventual carry, so the result is in B+[last]"""
    MAJ(circ, q, lost, B[0], A[0])
    for i in range(1, len(A)):
        MAJ(circ, q, A[i - 1], B[i], A[i])
    circ.cx(A[-1], last)
    for i in range(len(A) - 1, 0, -1):
        UMA(circ, q, A[i - 1], B[i], A[i])
    UMA(circ, q, lost, B[0], A[0])
    # now a=a and b = b+a


def UMA_3(circ, q, c, b, a):
    """A part of the adder, an other version with more gates but less depth"""
    circ.x(b)
    circ.cx(c, b)
    circ.ccx(c, b, a)
    circ.x(b)
    circ.cx(a, c)
    circ.cx(a, b)


def AMU_3(circ, q, c, b, a):
    """A part of the subber, an other version with more gates but less depth"""
    circ.cx(a, b)
    circ.cx(a, c)
    circ.x(b)
    circ.ccx(c, b, a)
    circ.cx(c, b)
    circ.x(b)


def add_o(circ, q, A, B, lost, last):
    """A and B are lists of the bits of a and b to be added, lost is ancilla.
     A->A B->A+B. Last is the eventual carry, so the result is in B+[last].
     This version uses more gates but less depth, so is faster"""
    if len(A) < 4:
        MAJ(circ, q, lost, B[0], A[0])
        for i in range(1, len(A)):
            MAJ(circ, q, A[i - 1], B[i], A[i])
        circ.cx(A[-1], last)
        for i in range(len(A) - 1, 0, -1):
            UMA_3(circ, q, A[i - 1], B[i], A[i])
        UMA_3(circ, q, lost, B[0], A[0])
        # now a=a and b = b+a
    else:
        n = len(A)
        for i in range(1, n):
            circ.cx(A[i], B[i])
        circ.cx(A[1], lost)
        circ.ccx(A[0], B[0], lost)
        circ.cx(A[2], A[1])
        circ.ccx(lost, B[1], A[1])
        circ.cx(A[3], A[2])
        for i in range(2, n-2):
            circ.ccx(A[i-1], B[i], A[i])
            circ.cx(A[i+2], A[i+1])
        circ.ccx(A[n-3], B[n-2], A[n-2])
        circ.cx(A[n-1], last)
        circ.ccx(A[n-2], B[n-1], last)
        for i in range(1, n-1):
            circ.x(B[i])
        circ.cx(lost, B[1])
        for i in range(2, n):
            circ.cx(A[i-1], B[i])
        circ.ccx(A[n-3], B[n-2], A[n-2])
        for i in range(n-3, 1, -1):
            circ.ccx(A[i-1], B[i], A[i])
            circ.cx(A[i + 2], A[i+1])
            circ.x(B[i+1])
        circ.ccx(lost, B[1], A[1])
        circ.cx(A[3], A[2])
        circ.x(B[2])
        circ.ccx(A[0], B[0], lost)
        circ.cx(A[2], A[1])
        circ.x(B[1])
        circ.cx(A[1], lost)
        for i in range(n):
            circ.cx(A[i], B[i])


def sub_o(circ, q, A, B, lost, last):
    """A and B are lists of the bits of a and b to be added, lost is ancilla.
         A->A A+B->B. Last is the eventual carry, so the result is in B+[last].
         This version uses more gates but less depth, so is faster"""
    if len(A) < 4:
        AMU_3(circ, q, lost, B[0], A[0])
        for i in range(1, len(A)):
            AMU_3(circ, q, A[i - 1], B[i], A[i])
        circ.cx(A[-1], last)
        for i in range(len(A) - 1, 0, -1):
            JAM(circ, q, A[i - 1], B[i], A[i])
        JAM(circ, q, lost, B[0], A[0])
    else:
        n = len(A)
        for i in range(n-1, -1, -1):
            circ.cx(A[i], B[i])
        circ.cx(A[1], lost)
        circ.x(B[1])
        circ.cx(A[2], A[1])
        circ.ccx(A[0], B[0], lost)
        circ.x(B[2])
        circ.cx(A[3], A[2])
        circ.ccx(lost, B[1], A[1])
        for i in range(2, n-2):
            circ.x(B[i + 1])
            circ.cx(A[i + 2], A[i + 1])
            circ.ccx(A[i - 1], B[i], A[i])
        circ.ccx(A[n - 3], B[n - 2], A[n - 2])
        for i in range(n-1, 1, -1):
            circ.cx(A[i - 1], B[i])
        circ.cx(lost, B[1])
        for i in range(n - 2, 0, -1):
            circ.x(B[i])
        circ.ccx(A[n - 2], B[n - 1], last)
        circ.cx(A[n - 1], last)
        circ.ccx(A[n - 3], B[n - 2], A[n - 2])
        for i in range(n-3, 1, -1):
            circ.cx(A[i + 2], A[i + 1])
            circ.ccx(A[i - 1], B[i], A[i])
        circ.cx(A[3], A[2])
        circ.ccx(lost, B[1], A[1])
        circ.cx(A[2], A[1])
        circ.ccx(A[0], B[0], lost)
        circ.cx(A[1], lost)
        for i in range(n-1, 0, -1):
            circ.cx(A[i], B[i])


def sub(circ, q, A, B, lost, last):
    """A and B are lists of the bits of a and b to be added, lost is ancilla.
    A->A A+B->B. Last is the eventual carry, so the result is in B+[last]."""
    AMU(circ, q, lost, B[0], A[0])
    for i in range(1, len(A)):
        AMU(circ, q, A[i - 1], B[i], A[i])
    circ.cx(A[-1], last)
    for i in range(len(A) - 1, 0, -1):
        JAM(circ, q, A[i - 1], B[i], A[i])
    JAM(circ, q, lost, B[0], A[0])


def addmod(circ, q, A, B, lost, last, N, lost2, binn):
    """The modular addition. A and B are lists of the bits of a and b to be added, N the modulo, a+b<2N,
    lost is ancilla. A->A B->B+A mod N. Last is the eventual carry, so the result is in B+[last]."""
    # A + B
    add_o(circ, q, A, B, lost, last)
    # A+B-N
    sub_o(circ, q, N, B, lost, last)
    # We check if A+B-N<0
    circ.x(last)
    circ.cx(last, lost2)
    circ.x(last)
    # if yes, we do:
    # control swap N to 0
    for i in range(len(binn)):
        if binn[i]:
            circ.cx(lost2, N[i])
    # A+B-N+(N if A+B-N<0 else 0)=A+B (-N if not negative) (<N so no modulo to be applied)
    add_o(circ, q, N, B, lost, last)
    # control swap 0 to N
    for i in range(len(binn)):
        if binn[i]:
            circ.cx(lost2, N[i])
    # And we do it again to reset the lost qbit
    sub_o(circ, q, A, B, lost, last)
    circ.cx(last, lost2)
    add_o(circ, q, A, B, lost, last)


def submod(circ, q, A, B, lost, last, N, lost2, binn):
    """The modular addition inverse. A and B are lists of the bits of a and b to be added, N the modulo, a+b<2N,
    lost is ancilla. A->A A+B mod N->B. Last is the eventual carry, so the result is in B+[last].
    It is just the addition in reverse order"""
    sub_o(circ, q, A, B, lost, last)
    circ.cx(last, lost2)
    add_o(circ, q, A, B, lost, last)
    for i in range(len(binn) - 1, -1, -1):
        if binn[i]:
            circ.cx(lost2, N[i])
    sub_o(circ, q, N, B, lost, last)
    for i in range(len(binn) - 1, -1, -1):
        if binn[i]:
            circ.cx(lost2, N[i])
    circ.x(last)
    circ.cx(last, lost2)
    circ.x(last)
    add_o(circ, q, N, B, lost, last)
    sub_o(circ, q, A, B, lost, last)


def cmultmod(circ, q, control, X, a, A, Y, n, N, binn, lost, lost2):
    """control the control qbit, X the list of qbits of b, bina the bitlist of a,
    A the list of qbits (at zero) with enough place to put a, Y the res register,
    N the modulo register, binn its bitlist and lost and lost2 two ancillas
    X->X Y->b*y if control else 0 A=0->0 N=0->0"""
    # We have to precompute the different a*2**i to be added if b_i
    binapow = [[int(x) for x in bin((powmod(2, i, n) * a) % n)[2:]] for i in range(len(X))]
    for i in range(len(binapow)):
        binapow[i].reverse()
    # For each bit in X
    for i in range(len(X)):
        # We conditionally load a*2**i%n in A for the addition. if not control, it will be 0
        for j in range(len(binapow[i])):
            if binapow[i][j]:
                circ.ccx(control, X[i], A[j])
        # We add either 0 or a*2**i%n depending on control and the i-th bit of X
        addmod(circ, q, A, Y, lost, Y[-1], N, lost2, binn)
        # We unload what was in A
        for j in range(len(binapow[i])):
            if binapow[i][j]:
                circ.ccx(control, X[i], A[j])
    # if not control, we load X in Y because it would be empty
    circ.x(control)
    for i in range(len(X)):
        circ.ccx(control, X[i], Y[i])
    circ.x(control)
    # note that for it to work, Y need to be empty at the start
    # A=0->0;X->X,Y=0->X if not control else x*a, N->N


def cdivmod(circ, q, control, X, a, A, Y, n, N, binn, lost, lost2):
    """The reverse multiplication. control the control qbit, B the list of qbits of b, bina the bitlist of a,
    A the list of qbits (at zero) with enough place to put a, Y the res register,
    N the modulo register, binn its bitlist and lost and lost2 two ancillas
    B->B b*y mod N->Y if control else 0 A=0->0 N=0->0"""
    binapow = [[int(x) for x in bin((powmod(2, i, n) * a) % n)[2:]] for i in range(len(X))]
    for i in range(len(binapow)):
        binapow[i].reverse()
    circ.x(control)
    for i in range(len(X) - 1, -1, -1):
        circ.ccx(control, X[i], Y[i])
    circ.x(control)
    for i in range(len(X) - 1, -1, -1):
        for j in range(len(binapow[i]) - 1, -1, -1):
            if binapow[i][j]:
                circ.ccx(control, X[i], A[j])
        submod(circ, q, A, Y, lost, Y[-1], N, lost2, binn)  # Y may be too long?
        for j in range(len(binapow[i]) - 1, -1, -1):
            if binapow[i][j]:
                circ.ccx(control, X[i], A[j])


def expmod(circ, q, X, a, A, APOW, Y, n, N, binn, lost, lost2):
    """The modular exponentiation. X is the list of bits of x such that we want a**x mod n. A has to be big enough
    to contain n (it will contain a*2**i%n), APOW is the same (it will contain a**(2**i)%n), Y is the same plus one,
    (it will contain the result at the end), n is the value of the modulo, N its register, binn its bitlist, and
    lost and lost2 two ancillas"""
    # We initialize APOW at 1 (start of successive multiplications)
    circ.x(APOW[0])
    # For each bit in X (the exposant)
    for i in range(len(X)):
        # We do a controlled multiplication if x_i
        control = X[i]
        # Here APOW cointains a**(x_0*2**0)*a**(x_1*2**1)*...*a**(x_k*2**k)%N and Y = 0
        cmultmod(circ, q, control, APOW, powmod(a, 2 ** i, n), A, Y, n, N, binn, lost, lost2)
        # Here APOW cointains a**(x_0*2**0)*a**(x_1*2**1)*...*a**(x_k*2**k)%N and
        # Y = a**(x_0*2**0)*a**(x_1*2**1)*...*a**(x_k*2**k)*a**(x_{k+1}*2**(k+1))%N
        # We switch APOW and Y
        temp = APOW.copy()
        APOW = Y[:-1]
        Y = temp + [Y[-1]]
        # We reset Y (containing the previous step)
        # Here APOW cointains a**(x_0*2**0)*a**(x_1*2**1)*...*a**(x_k*2**k)*a**(x_{k+1}*2**(k+1))%N and
        # Y = a**(x_0*2**0)*a**(x_1*2**1)*...*a**(x_k*2**k)%N
        cdivmod(circ, q, control, APOW, powmod(inversemod(a, n), 2 ** i, n), A, Y, n, N, binn, lost, lost2)
        # Here APOW cointains a**(x_0*2**0)*a**(x_1*2**1)*...*a**(x_k*2**k)*a**(x_{k+1}*2**(k+1))%N and Y = 0
        # Ready to loop again!

    return 1


def fracf(N, S):
    """We suppose S = k*N/L. Return k,L. Ex: N=10000, S = 66667, res = 1,3"""
    L = [N, S]
    Lneg = [-S]
    T = []
    while L[-1] > 10 and -Lneg[-1] > 10:
        L.append(L[-2] % L[-1])
        T.append(L[-3] // L[-2])
        Lneg.append((L[-3] % L[-2]) - L[-2])
        print(L, T, Lneg)
    Backtrack = [(T[-1]+1*(Lneg[-1]>-10)), 1]
    for i in range(len(T)-1):
        Backtrack = [T[-2-i]*Backtrack[0]+Backtrack[1]]+Backtrack
    print(Backtrack)
    return Backtrack[1], Backtrack[0]


def QFTn(circ, q, X):
    """A strange QFT but it works"""
    lamb = [2 * np.pi / (2 ** m) for m in range(2, len(X) + 1)]
    X.reverse()
    for i in range(len(X)):
        circ.h(X[i])
        for j in range(len(lamb) - i):
            circ.cu1(lamb[j], X[1 + j + i], X[i])
    return 1


def iQFTn(circ, q, X):
    """The reverse QFT"""
    lamb = [2 * np.pi / (2 ** m) for m in range(2, len(X) + 1)]
    for i in range(len(X)-1, -1, -1):
        for j in range(len(lamb) - i-1, -1, -1):
            circ.cu1(-lamb[j], X[1 + j + i], X[i])
        circ.h(X[i])
    X.reverse()
    return 1


def Uphi(circ, q, x, RegX):
    """Adds a Uphi(x) gate to circ at RegX. x and RegX must be of size 2"""
    n = len(x)
    Phi = [x[i] for i in range(n)]+[(np.pi-x[i])*(np.pi-x[(i+1)%n]) for i in range(n)]
    for reg in RegX:
        circ.h(reg)
    for i in range(n):
        circ.u1(-2*Phi[i], RegX[i])
    for i in range(n):
        circ.cx(RegX[i], RegX[(i+1) % n])
        circ.u1(-2 * Phi[n+i], RegX[(i+1) % n])
        circ.cx(RegX[i], RegX[(i+1) % n])
    for reg in RegX:
        circ.h(reg)
    for i in range(n):
        circ.u1(-2*Phi[i], RegX[i])
    for i in range(n):
        circ.cx(RegX[i], RegX[(i+1) % n])
        circ.u1(-2 * Phi[n+i], RegX[(i+1) % n])
        circ.cx(RegX[i], RegX[(i+1) % n])
    circ.barrier(q)
    return 1


def iUphi(circ, q, x, RegX):
    """inverted Adds a Uphi(x) gate to circ at RegX. x and RegX must be of size 2"""
    n = len(x)
    Phi = [x[i] for i in range(n)]+[(np.pi-x[i])*(np.pi-x[(i+1)%n]) for i in range(n)]
    for i in range(n-1, -1, -1):
        circ.cx(RegX[i],RegX[(i+1)%n])
        circ.u1(2 * Phi[n+i], RegX[(i+1)%n])
        circ.cx(RegX[i],RegX[(i+1)%n])
    for i in range(n):
        circ.u1(2*Phi[i], RegX[i])
    for reg in RegX:
        circ.h(reg)
    for i in range(n-1, -1, -1):
        circ.cx(RegX[i],RegX[(i+1)%n])
        circ.u1(2 * Phi[n+i], RegX[(i+1)%n])
        circ.cx(RegX[i],RegX[(i+1)%n])
    for i in range(n):
        circ.u1(2*Phi[i], RegX[i])
    for reg in RegX:
        circ.h(reg)
    circ.barrier(q)
    return 1


def W(circ, q, RegX, Theta):
    """Place a Variational Trainable circuit on circ at RegX"""
    for i in range(len(RegX)):
        circ.u3(Theta[i + len(RegX)], 0.0, 0.0, RegX[i])
        circ.u1(Theta[i], RegX[i])
    for i in range(2*len(RegX), len(Theta), 2*len(RegX)):
        circ.barrier(q)
        for k in range(len(RegX)):
            circ.u2(0.0, np.pi, RegX[(k+1) % len(RegX)])
            circ.cx(RegX[k], RegX[(k+1) % len(RegX)])
            circ.u2(0.0, np.pi, RegX[(k+1) % len(RegX)])
        for j in range(len(RegX)):
            circ.u3(Theta[i + j + len(RegX)], 0.0, 0.0, RegX[j])
            circ.u1(Theta[i+j], RegX[j])

    circ.barrier(q)
    return 1


def sum_l_str(L):
    res = ''
    for l in L:
        res += l
    return res


def genere_chains(n, sub):
    assert(sub < n)
    Letters = ['A', 'C', 'T', 'G']
    correct_string = ''
    for _ in range(n):
        nuc = r.choice(Letters)
        correct_string += nuc
    #print(correct_string)
    mutation = [c for c in correct_string]
    for i in range(sub):
        place = r.randrange(n)
        new = r.choice(Letters)
        while new == correct_string[place]:
            new = r.choice(Letters)
        mutation[place] = new
    mutation = sum_l_str(mutation)
    return correct_string, mutation


def mutation(chaine, force):
    """force = [chance de décalage, chance de mutation]"""
    Letters = ['A', 'C', 'T', 'G']
    mutate = ''
    decal = -1
    if r.random() < force[0]:
        decal = r.randrange(len(chaine)-1)
    i = 0
    while len(mutate) < len(chaine):
        if i == decal:
            mutate += r.choice(Letters)
        if r.random() < force[1]:
            mutate += r.choice(Letters)
        else:
            mutate += chaine[i]
        i += 1
    return mutate