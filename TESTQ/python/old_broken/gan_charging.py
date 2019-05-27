from General import *
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

from qiskit.aqua.components.optimizers import SPSA


def sum_l(L):
    res = []
    for l in L:
        res += l
    return res


def Variationer_learn(shots, l, m):
    """Learn the theta to find the data. data is of the form list (list (tuple label)).
    Shots is the amount of shots to take for the empirical evaluation of p_y.
    l is the depth of the variationner"""

    N = 10 ** 5  # Taille de la database
    n = 5  # nombre de qubits (détermine la précision)

    Database = np.random.normal(0, 1, N)

    mini = np.min(Database)
    maxi = np.max(Database)
    h = (maxi - mini) / (2 ** n)
    bins = [[k for d in Database if mini + h * k < d < mini + h * (k + 1)] for k in range(2 ** n)]
    Database = sum_l(bins)
    interv = [mini + h * k for k in range(2 ** n)]
    #vals = [len(bine) / N for bine in bins]

    clf = MLPClassifier(hidden_layer_sizes=(int(m/2), 8, 5, 2))

    # Initial random theta
    Theta = np.random.normal(0, 1, 2*n*(l+1))

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
        #print(counts, m)
        # We transform the results
        input_var = []
        for KEY in counts:
            value = int('0b' + KEY, 2)
            for _ in range(counts[KEY]):
                input_var.append(value)
        r.shuffle(Database)
        input_db = Database[:m]
        r.shuffle(input_var)

        input = [sorted(input_var)] + [sorted(input_db)]
        print(input)
        target = [0, 1]
        clf.partial_fit(input, target, [0, 1])
        err_theta = -np.log(clf.predict_proba([input[0]]))[0][1]
        curve.append(err_theta)
        print(err_theta)
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
    counts = launch(1024, circ_m)
    # We transform the results
    bins_var = [0 for _ in range(2**n)]
    for KEY in counts:
        value = int('0b' + KEY, 2)
        bins_var[value] = counts[KEY]
    plt.plot(interv, bins_var)
    plt.show()
    return theta_star[1]


Variationer_learn(1000, 3, 64)



