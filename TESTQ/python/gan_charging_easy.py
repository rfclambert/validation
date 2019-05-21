from General import *
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

from qiskit_aqua.components.optimizers import SPSA


def sum_l(L):
    res = []
    for l in L:
        res += l
    return res


def norm1(f, g):
    res = 0
    for i in range(len(f)):
        res += np.abs(f[i]-g[i])
    return res


def Variationer_learn(shots, l, m, proba=None, n=4):
    """Learn the theta to find the data. data is of the form list (list (tuple label)).
    Shots is the amount of shots to take for the empirical evaluation of p_y.
    l is the depth of the variationner"""
    print(n)
    N = 10 ** 5  # Taille de la database
    #n = 4  # nombre de qubits (détermine la précision)
    if proba is None:
        Database = np.random.normal(0, 1, N)

        mini = np.min(Database)
        maxi = np.max(Database)
        h = (maxi - mini) / (2 ** n)
        bins = [[k for d in Database if mini + h * k < d < mini + h * (k + 1)] for k in range(2 ** n)]
    else:
        print(2**n, len(proba))
        assert(2**n == len(proba))
        assert(0.98<sum(proba)<1.01)
        mini = 0
        maxi = (2**n) - 1
        h = (maxi - mini) / (2 ** n)
        bins = [[0]*int(p*N) for p in proba]

    interv = [mini + h * k for k in range(2 ** n)]
    #vals = [len(bine) / N for bine in bins]

    # Initial random theta
    Theta = np.random.normal(0, 10**-2, 2*n*(l+1))

    # Initial circuit
    q = QuantumRegister(n, 'q')
    RegX = [q[i] for i in range(n)]

    # To keep track of what's done
    curve = []

    def Remp(theta):
        """The error function"""
        # Part of the circuit common to all data
        circ = QuantumCircuit(q)
        W(circ, q, RegX, theta)
        circ_var = circ
        circ_m = measure_direct(circ_var, q, RegX)
        counts = launch(m, circ_m)
        # We transform the results
        bins_var = [0 for _ in range(2 ** n)]
        for KEY in counts:
            value = int('0b' + KEY, 2)
            bins_var[value] = counts[KEY] / m
        compar = [len(b) / N for b in bins]
        err_theta = norm1(compar, bins_var)
        curve.append(err_theta)
        print(err_theta, len(curve))
        return err_theta
    # The optimizer
    optimizer = SPSA(max_trials=shots, c0=4.0, c1=0.1, c2=0.602,c3=0.101,c4=0.0, skip_calibration=True)
    optimizer.set_options(save_steps=1)
    theta_star = optimizer.optimize(2*n*(l+1), Remp, initial_point=Theta)

    plt.plot([i for i in range(len(curve))], curve)
    plt.show()
    print("Debut du test")
    # We test the trained model
    circ_test = QuantumCircuit(q)
    W(circ_test, q, RegX, theta_star[0])
    circ_m = measure_direct(circ_test, q, RegX)
    prec_here = 8192
    counts = launch(prec_here, circ_m)
    # We transform the results
    bins_var = [0 for _ in range(2**n)]
    for KEY in counts:
        value = int('0b' + KEY, 2)
        bins_var[value] = counts[KEY]/prec_here
    compar = [len(b)/N for b in bins]
    if len(interv) == len(compar):
        plt.plot(interv, compar)
    plt.plot(interv, bins_var)
    plt.show()
    return theta_star[0], norm1(compar, bins_var)

def Variational_prepared(theta, n):
    N = 10**5
    q = QuantumRegister(n, 'q')
    circ_test = QuantumCircuit(q)
    RegX = [q[i] for i in range(n)]
    W(circ_test, q, RegX, theta)
    circ_m = measure_direct(circ_test, q, RegX)
    prec_here = 4096
    counts = launch(prec_here, circ_m)
    bins_var = [0 for _ in range(2 ** n)]
    for KEY in counts:
        value = int('0b' + KEY, 2)
        bins_var[value] = counts[KEY] / prec_here
    Database = np.random.normal(0, 1, N)
    mini = np.min(Database)
    maxi = np.max(Database)
    h = (maxi - mini) / (2 ** n)
    bins = [[k for d in Database if mini + h * k < d < mini + h * (k + 1)] for k in range(2 ** n)]
    compar = [len(b) / N for b in bins]
    interv = [mini + h * k for k in range(2 ** n)]
    if len(interv) == len(compar):
        plt.plot(interv, compar)

    plt.plot(interv, bins_var)

    plt.show()
    return counts


def mosel():
    res = []
    for shots in [500, 1000]:
        for l in range(1, 6):
            for m in [16, 32, 64, 128, 256, 512, 1024]:
                res.append(Variationer_learn(shots, l, m))
                print(res[-1])
    print(res)


#mosel()
print(Variationer_learn(1000, 3, 4096, proba=[1/24]*24+8*[0], n=5))


t_cartes = [ -2.69594966,   2.32087974,  -2.93514564,  -2.40187986,
        -3.12253911,   1.42226258,   1.58419545,  -6.19110692,
         6.28707991,  -4.18947154,  -1.81484303,  -1.87517441,
         2.11192122,   0.95329807,   2.02605144,  -7.81099626,
         4.16123674,  -3.32482814,   6.47763031,  -3.84806068,
         3.60302162,  -3.5275127 ,  -7.66684734, -12.79672765,
        -1.34402021,   3.33807265,  -1.56936059,  -3.11115854,
         5.39177194,   2.9766531 ,  -8.8426481 ,   1.55053037,
        -0.87447843,  -1.30369889,  -3.19885241,   0.28281301,
         1.95656589,  -4.57453334,   0.46110864,   2.94559584]

#Theta_normal_3_qubits_1_l = [-1.67673687, 1.03846931, -1.23430009, -0.62289478, -1.5214651, -1.3836211, -1.46432798,
#                         1.29695935, 0.90557468, 1.51990279, -1.22472484, -1.01901762]
# Theta_normal_4_qubits_2_l = [ -8.71899641,  -0.23749671,  -4.55442361,  -0.37604125,
#          0.08447067,  -3.4566636 ,   8.09786053,   2.22697822,
#          6.01210079,  -6.98933334,  -4.46638232,  -2.94839828,
#         -6.03761162,  -8.63327411,   3.23178755,   1.83183239,
#         -0.05029006,   3.34760167,   0.4856    ,  -0.18212859,
#         -1.59968093,   4.26177009,  -0.15821456, -10.50228961]

#Variational_prepared(Theta_normal_4_qubits_2_l, 4)


