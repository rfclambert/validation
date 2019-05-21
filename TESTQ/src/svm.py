from General import *
import numpy as np
import matplotlib.pyplot as plt
from qiskit import IBMQ
from utils import get_data, split
from qiskit.aqua import run_algorithm, QuantumInstance
from qiskit.aqua.algorithms import QSVM, VQC
from qiskit.aqua.components.optimizers import SPSA
from qiskit.aqua.components.feature_maps import SecondOrderExpansion, FirstOrderExpansion, PauliExpansion, PauliZExpansion
from qiskit.aqua.components.variational_forms import RYRZ
from sklearn import svm, datasets
from sklearn.decomposition import PCA

IBMQ.load_accounts()

def generation(n, delta):
    """We want to generate data hard to learn for a classical machine learning algo.
    We generate couples and then apply a map on it"""
    # The list of -1 labeled data
    N = []
    # The list of 1 labeled data
    P = []
    # The two random SU(2) gates to form the SU(4) gate
    V1 = [r.random() * np.pi, r.random() * 2 * np.pi, r.random() * 2 * np.pi]
    V2 = [r.random() * np.pi, r.random() * 2 * np.pi, r.random() * 2 * np.pi]
    # Just a number to see the advancement
    bt = 0
    while len(N) < n or len(P) < n:

        bt += 1
        if not bt%100:
            print("Try number", bt)

        # We take a random point
        x = (r.random() * 2 * np.pi, r.random() * 2 * np.pi)

        #We create the circuit to map it and label it
        q = QuantumRegister(2, 'q')
        circ = QuantumCircuit(q)
        Uphi(circ, q, x, [q[0], q[1]])
        circ.u3(V1[0], V1[1], V1[2], q[0])
        circ.u3(V2[0], V2[1], V2[2], q[1])
        circ_m = measure_direct(circ, q, [q[0], q[1]])
        counts = launch(2048, circ_m)

        # Now, we will decide it's label
        res = 0
        for key in counts:
            if key[0] == key[1]:
                res += counts[key]
            else:
                res -= counts[key]
        # This is the mean result, the one we will use to decide the label
        res = res/2048
        # delta is the threshold, for the two labeled zones to be clearly separated.
        if res > delta and len(P) < n:
            P.append(x)
            print(res, len(P), len(N))
        if res < -delta and len(N) < n:
            N.append(x)
            print(res, len(P), len(N))

    # We can see the final map
    plt.scatter([p[0] for p in P], [p[1] for p in P])
    plt.scatter([p[0] for p in N], [p[1] for p in N])
    plt.show()
    # We also return the parameters used for V1 and V2 if we want to compare different data using the same map.
    return N, P, V1, V2


def generation_grid(n, delta):
    """We want to generate data hard to learn for a classical machine learning algo.
    We generate all couples and then apply a map on it. n**2 results, then we can sample from it"""
    N = []
    P = []
    V1 = [r.random() * np.pi, r.random() * 2 * np.pi, r.random() * 2 * np.pi]
    V2 = [r.random() * np.pi, r.random() * 2 * np.pi, r.random() * 2 * np.pi]
    for i in range(n):
        for j in range(n):
            x = (2 * np.pi*(i+0.5)/n, 2 * np.pi*(j+0.5)/n)
            q = QuantumRegister(2, 'q')
            circ = QuantumCircuit(q)
            Uphi(circ, q, x, [q[0], q[1]])
            circ.u3(V1[0], V1[1], V1[2], q[0])
            circ.u3(V2[0], V2[1], V2[2], q[1])
            #circ.z(q)
            circ_m = measure_direct(circ, q, [q[0], q[1]])
            counts = launch(2048, circ_m)
            res = 0
            for key in counts:
                if key[0] == key[1]:
                    res += counts[key]
                else:
                    res -= counts[key]
            res = res/2048
            if res > delta:
                P.append(x)
            if res < -delta:
                N.append(x)
            print(res)
        print(res, len(P), len(N), i, j)
    plt.scatter([p[0] for p in P], [p[1] for p in P])
    plt.scatter([p[0] for p in N], [p[1] for p in N])
    plt.show()
    return N, P, V1, V2


def stock(n, sep):
    res = generation(n, sep)

    uni = open('uni_{}_{}.txt'.format(n, sep), 'w')
    doc = open('res_{}_{}.txt'.format(n, sep), 'w')
    for i in range(max(len(res[0]), len(res[1]))):
        if i < len(res[0]) and i < len(res[1]):
            doc.write("{} {}\n".format(res[0][i], res[1][i]))
        elif i >= len(res[0]):
            doc.write("{} {}\n".format((0, 0), res[1][i]))
        else:
            doc.write("{} {}\n".format(res[0][i], (0, 0)))
    for i in range(len(res[2])):
        uni.write("{} {}\n".format(res[2][i], res[3][i]))
    uni.close()
    doc.close()
    return res


def stock_compl(n, sep):
    res = generation_grid(n, sep)

    uni = open('uni_grid.txt', 'w')
    doc = open('res_grid.txt', 'w')
    for i in range(max(len(res[0]), len(res[1]))):
        if i < len(res[0]) and i < len(res[1]):
            doc.write("{} {}\n".format(res[0][i], res[1][i]))
        elif i >= len(res[0]):
            doc.write("{} {}\n".format((0, 0), res[1][i]))
        else:
            doc.write("{} {}\n".format(res[0][i], (0, 0)))
    for i in range(len(res[2])):
        uni.write("{} {}\n".format(res[2][i], res[3][i]))
    uni.close()
    doc.close()
    return res


def stock_get(n, sep):
    doc = open('res_{}_{}.txt'.format(n, sep), 'r')
    nbr = ''
    store= False
    res = []
    resm = []

    for line in doc:
        L = []
        Lm = []
        pmc = 0
        for c in line:
            if c.isnumeric() or c == '.':
                store = True
                nbr += c
            if store and not c.isnumeric() and not c == '.':
                if pmc<2:
                    L.append(float(nbr))
                else:
                    Lm.append(float(nbr))
                store = False
                pmc += 1
                nbr = ''
        res.append(L)
        resm.append(Lm)
        # print(res, resm)

    return [r for r in resm], [r for r in res]


def sig(x):
    """Sigmoid"""
    return 1/(1+np.exp(-x))


def pmdm(R, py, y, b):
    """returns p(m(x)!=mhat(x))"""
    return sig( (np.sqrt(R)*( 0.5*(1-y*b) - py ))/np.sqrt(2*(1-py)*py) )


def clamp(x, a, b):
    """np.clip in fact"""
    return a * (x < a) + x * (x >= a and x <= b) + b * (x > b)


def built_inVar():
    """The aqua var"""
    feature_dim = 3
    shots = 1024
    random_seed = 10598
    gener = stock_get(5, 0.3)  # generation(20, 0.2)
    print(gener)
    #gener_test = generation(5, 0.3)
    training_input = {'-1': [gener[0][i]+[0.2] for i in range(len(gener[0])//2)],
                      '1': [gener[1][i]+[-0.2] for i in range(len(gener[1])//2)]}
    test_input = {'-1': [gener[0][i]+[0.2] for i in range(len(gener[0])//2, len(gener[0]))],
                  '1': [gener[1][i]+[-0.2] for i in range(len(gener[1])//2, len(gener[1]))]}
    backend = BasicAer.get_backend('qasm_simulator')
    optimizer = SPSA(max_trials=100, c0=4.0, skip_calibration=True)
    optimizer.set_options(save_steps=1)
    feature_map = PauliExpansion(feature_dimension=feature_dim, depth=2, paulis=[ 'Z', 'ZZ', 'ZZZ'])
    var_form = RYRZ(num_qubits=feature_dim, depth=3)
    svm_b = VQC(optimizer, feature_map, var_form, training_input, test_input)
    quantum_instance = QuantumInstance(backend, shots=shots, seed=random_seed, seed_transpiler=random_seed)
    result = svm_b.run(quantum_instance)
    print("testing success ratio: ", result['testing_accuracy'])


def Variationer_learn(data, shots, l, epsilon, test=None):
    """Learn the theta to find the data. data is of the form list (list (tuple label)).
    Shots is the amount of shots to take for the empirical evaluation of p_y.
    l is the depth of the variationner"""
    # Number of qubits used
    n = len(data[0])
    # Initial random theta
    Theta = np.random.normal(0, 1, 2*n*(l+1))
    # Initial circuit
    q = QuantumRegister(2, 'q')
    RegX = [q[i] for i in range(n)]
    # To keep track of what's done
    curve = []

    def Remp(theta):
        """The error function"""
        # Part of the circuit common to all data
        circ = QuantumCircuit(q)
        W(circ, q, RegX, theta)
        # For each data, compute the error
        err_theta = 0
        for i in range(len(data)):
            # We create the circuit for this data
            circ_uphi = QuantumCircuit(q)
            Uphi(circ_uphi, q, data[i][0], RegX)
            circ_var = circ_uphi + circ
            circ_m = measure_direct(circ_var, q, RegX)
            counts = launch(shots, circ_m)
            # We count each label
            res = 0
            for KEY in counts:
                if KEY[0] == KEY[1]:
                    res += counts[KEY]
                else:
                    res -= counts[KEY]
            res = res / shots
            # We have the probability of each label
            resm = 0.5 * (1 - res)
            resp = 0.5 * (1 + res)
            # We compute the error of this data (either cross entropy or binomial error)
            if data[i][1] == 1:
                #err_theta += pmdm(shots, resp, 1, 0)#clamp(theta[-1], -0.1, 0.1))
                err_theta -= np.log(clamp(resp, epsilon, 1-epsilon))
            else:
                #err_theta += pmdm(shots, resm, -1, 0)#clamp(theta[-1], -0.1, 0.1))
                err_theta -= np.log(clamp(resm, epsilon, 1-epsilon))
        # The error for this theta
        print(err_theta/len(data))
        curve.append(err_theta/len(data))
        return err_theta/len(data)
    # The optimizer
    optimizer = SPSA(max_trials=200, c0=4.0, c1=0.1, c2=0.602,c3=0.101,c4=0.0, skip_calibration=True)
    optimizer.set_options(save_steps=1)
    theta_star = optimizer.optimize(2*n*(l+1), Remp, initial_point=Theta)

    plt.plot([i for i in range(len(curve))], curve)
    plt.show()

    # We test the trained model
    success = 0
    circ_test = QuantumCircuit(q)
    W(circ_test, q, RegX, theta_star[0])
    for datum in test:
        circ_uphi_test = QuantumCircuit(q)
        Uphi(circ_uphi_test, q, datum[0], RegX)
        circ_var_test = circ_uphi_test + circ_test
        circ_m_test = measure_direct(circ_var_test, q, RegX)
        counts_test = launch(shots, circ_m_test)
        res_test = 0
        for key in counts_test:
            if key[1] == key[0]:
                res_test += counts_test[key]
            else:
                res_test -= counts_test[key]
        res_test = res_test / shots
        resm_test = 0.5 * (1 - res_test)
        resp_test = 0.5 * (1 + res_test)
        if resm_test > resp_test:
            if datum[1] == -1:
                success += 100/len(test)
        else:
            if datum[1] == 1:
                success += 100/len(test)
    print("Pourcentage de succes:", success)
    return theta_star[1]


def kern(X, Y):
    """The kernel function that takes a list of X  and Y and computes K(x,y) for each pair"""
    # Hard coded number of shots
    shots = 4000
    n = len(X)
    n_y = len(Y)
    sym = False
    if n == n_y:
        sym = True
    m = min(len(X[0]), len(Y[0]))
    # Quantum circuit
    q = QuantumRegister(m, 'q')
    RegX = [q[i] for i in range(m)]
    # We will put our results here
    kernel_mat = np.zeros((n, n_y))
    print("Start of the evaluation of K, wait for {} iterations".format(n*n_y))
    for i in range(n):
        if sym:
            for j in range(i, n_y):
                # To keep track of the time
                if (i*n_y+j) % 10 == 0:
                    print("Iteration {}/{}".format((i*n_y+j), n*n_y))
                # We build and run the circuit
                circ_x = QuantumCircuit(q)
                Uphi(circ_x, q, X[i], RegX)
                iUphi(circ_x, q, Y[j], RegX)
                circ_m_x = measure_direct(circ_x, q, RegX)
                counts_test = launch(shots, circ_m_x)
                # We count the probability to get 00
                if '0' * m in counts_test:
                    kernel_mat[i][j] = counts_test['0' * m] / shots
                    kernel_mat[j][i] = counts_test['0' * m] / shots
        else:
            for j in range(n_y):
                # To keep track of the time
                if (i*n_y+j) % 10 == 0:
                    print("Iteration {}/{}".format((i*n_y+j), n*n_y))
                # We build and run the circuit
                circ_x = QuantumCircuit(q)
                Uphi(circ_x, q, X[i], RegX)
                iUphi(circ_x, q, Y[j], RegX)
                circ_m_x = measure_direct(circ_x, q, RegX)
                counts_test = launch(shots, circ_m_x)
                # We count the probability to get 00
                if '0' * m in counts_test:
                    kernel_mat[i][j] = counts_test['0' * m] / shots
    # We check if it was an evaluation of the Gram matrix
    # Because this function can't differenciate the first evaluation and the one for the test.
    # print("The Kernel matrix for this is:")
    # print(kernel_mat)
    return kernel_mat


def kernel_estimation(X, Y, T, Ty):
    #n = len(X[0])
    #m = len(X)
    h = 0.5
    clf = svm.SVC(kernel=kern)
    clf.fit(X, Y)
    print(clf.score(T, Ty))


def classical_kernel_estimation(X, Y, T, Ty):
    clf = svm.SVC(gamma='auto')
    clf.fit(X, Y)
    print(clf.score(T, Ty))


def test_iris():
    iris = datasets.load_iris()
    dejavu = []
    X = []
    Y = []
    T = []
    Ty = []
    n = len(iris.target)
    for i in range(20):
        ind = r.randrange(n)
        while ind in dejavu:
            ind = r.randrange(n)
        X.append(iris.data[ind, :4])
        Y.append(-1*(iris.target[ind]==0)+1*(iris.target[ind]!=0))
        dejavu.append(ind)
        ind = r.randrange(n)
        while ind in dejavu:
            ind = r.randrange(n)
        T.append(iris.data[ind, :4])
        Ty.append(-1*(iris.target[ind]==0)+1*(iris.target[ind]!=0))
        dejavu.append(ind)
        #print(dejavu)
    print(X, Y, T, Ty)
    classical_kernel_estimation(X, Y, T, Ty)
    kernel_estimation(X, Y, T, Ty)

    gener = stock(10, 0.3)
    data_learn = [gener[0][i] for i in range(len(gener[0]) // 2)] + [gener[1][i] for i in range(len(gener[1]) // 2)]
    data_lab = [-1 for i in range(len(gener[0])//2)] + [1 for i in range(len(gener[1])//2)]
    print(data_learn, data_lab)
    test = [gener[0][i] for i in range(len(gener[0])//2, len(gener[0]))] +\
           [gener[1][i] for i in range(len(gener[1])//2, len(gener[1]))]
    test_lab = [-1 for i in range(len(gener[0])//2, len(gener[0]))] +\
           [1 for i in range(len(gener[1])//2, len(gener[1]))]
    print(test, test_lab)
    classical_kernel_estimation(data_learn, data_lab, test, test_lab)
    kernel_estimation(data_learn, data_lab, test, test_lab)


def test_string(n, nbr_qbits, choice, method):
    n_test = 20
    if choice == 'mat':
        X, Y, _ = get_data(0)
        Y = Y[:, 1].ravel()
    if choice == 'raw':
        X, Y, _ = get_data(0, "raw")
        Y = Y[:, 1].ravel()
    if choice == 'genere':
        normal, muta = genere_chains(nbr_qbits, 4)
        X = [normal]*int(n+n_test+1)+[muta]*int(n+n_test+1)
        Y = [1]*int(n+n_test+1)+[0]*int(n+n_test+1)
        ind = [i for i in range(len(X))]
        r.shuffle(ind)
        X = [mutation(X[i], [0.1, 0.2]) for i in ind]
        Y = [Y[i] for i in ind]

    print(X, Y)

    if choice in ['raw', 'genere']:
        X_crash = []
        for seq in X:
            crash = []
            for nucleotid in seq:
                crash.append( (0*(nucleotid=='A')+1*(nucleotid=='C')+2*(nucleotid=='T')+3*(nucleotid=='G'))*np.pi/2)
            X_crash.append(crash)
        X = np.array(X_crash)
        Y = np.array(Y)

    print(X, Y)

    if method == 'pca':
        pca = PCA(n_components=nbr_qbits).fit(X)
        X_red = pca.transform(X)
    if method == 'trunc':
        X_red = X[:, :nbr_qbits]
    Y = 2*(Y-0.5)

    print(X_red, Y)

    X_train, Y_train, X_test, Y_test, _, _ = split(X_red, Y, 0.2, 0)
    print(X_train, Y_train, X_test, Y_test)
    classical_kernel_estimation(X_train, Y_train, X_test, Y_test)
    classical_kernel_estimation(X_train[:n+1], Y_train[:n+1], X_test[:n_test], Y_test[:n_test])
    kernel_estimation(X_train[:n+1], Y_train[:n+1], X_test[:n_test], Y_test[:n_test])





#built_inVar()
# gener = stock_get(20, 0.3)  # generation(20, 0.3)
# data_learn = [[gener[0][i], -1] for i in range(len(gener[0])//2)] + [[gener[1][i], 1] for i in range(len(gener[1])//2)]
# test = [[gener[0][i], -1] for i in range(len(gener[0])//2, len(gener[0]))] +\
#       [[gener[1][i], 1] for i in range(len(gener[1])//2, len(gener[1]))]
# Variationer_learn(data_learn, 100, 3, 10**-10, test)


# gener = stock_get(20, 0.3)
# #gener = stock(20, 0.3)
# print(gener)
# data_learn = [gener[0][i] for i in range(len(gener[0])//2)] + [gener[1][i] for i in range(len(gener[1])//2)]
# data_lab = [-1 for i in range(len(gener[0])//2)] + [1 for i in range(len(gener[1])//2)]
# print(data_learn, data_lab)
# test = [gener[0][i] for i in range(len(gener[0])//2, len(gener[0]))] +\
#        [gener[1][i] for i in range(len(gener[1])//2, len(gener[1]))]
# test_lab = [-1 for i in range(len(gener[0])//2, len(gener[0]))] +\
#        [1 for i in range(len(gener[1])//2, len(gener[1]))]
# print(test, test_lab)
# #kernel_estimation(data_learn, data_lab, test, test_lab)
# classical_kernel_estimation(data_learn, data_lab, test, test_lab)
# test_iris()
# print(genere_chains(15,4))
# est_string(20, 15, 'genere', 'trunc')

#stock_compl()


