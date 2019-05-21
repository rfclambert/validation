from General import *
from svm import *
from utils import get_data
import matplotlib.pyplot as plt
import functools
from custom_map import CustomExpansion
from sklearn import svm
from qiskit.aqua import run_algorithm, QuantumInstance
from qiskit.aqua.components.feature_maps import SecondOrderExpansion, FirstOrderExpansion, PauliExpansion, self_product
from qiskit.aqua.algorithms import QSVM
from qiskit.aqua.algorithms.many_sample.qsvm._qsvm_estimator import _QSVM_Estimator
from qiskit.aqua.components.multiclass_extensions.all_pairs import *
from qsvm_datasets import *
from wavelets import Wavelets

def ccxtest(n):
    """Truth table for cnx"""
    b = bina(n)
    for bi in b:
        print(bi)
        q = QuantumRegister(n, 'q')
        circ = QuantumCircuit(q)
        for i in bi:
            circ.x(q[i])
        cnx(circ, q, [q[i] for i in range(n - 2, -1, -1)], q[n - 1])
        circ.barrier(q)
        launch2(circ)

        circ_m = measure(circ, q, [i for i in range(n)])
    return circ_m


def crootnxtest(n):
    """Truth table for crootnx"""
    b = bina(n)
    for bi in b:
        print(bi)
        q = QuantumRegister(n, 'q')
        circ = QuantumCircuit(q)
        for i in bi:
            circ.x(q[i])
        for _ in range(2 ** (n)):
            crootnx(circ, q, q[0], q[n - 1], 2 ** n, False)
        circ.barrier(q)
        launch2(circ)

        circ_m = measure(circ, q, [i for i in range(n)])

    return circ_m


def oracletest(n):
    """Test All Oracles"""
    b = bina(n)
    for bi in b:
        print(bi)
        q = QuantumRegister(n, 'q')
        circ = QuantumCircuit(q)
        circ.h(q)
        # for i in bi:
        #    circ.x(q[i])
        oracle(circ, q, n, bi)
        circ.barrier(q)
        launch2(circ)
        circ_m = measure(circ, q, [i for i in range(n)])

    return circ_m


def ccx_otest(n):
    """Truth table for cnx"""
    b = [bina(n + 1)[0]]
    # b = bina(n+1)
    for bi in b:
        print(bi)
        q = QuantumRegister(n + 1, 'q')
        circ = QuantumCircuit(q)
        for i in bi:
            circ.x(q[i])
        cnx_o(circ, q, [q[i] for i in range(n - 2, -1, -1)], q[n], q[n - 1])
        # circ.mct([q[i] for i in range(n-2,-1,-1)], q[n], q[n-1])
        circ.barrier(q)
        launch2(circ)

        circ_m = measure(circ, q, [i for i in range(n + 1)])
    return circ_m


def addition(a,b):
    """mesure b = b+a, a reste a"""
    bina = [int(x) for x in bin(a)[2:]]
    binb = [int(x) for x in bin(b)[2:]]
    while len(bina)>=len(binb):
        binb = [0]+binb
    while len(bina)<len(binb)-1:
        bina = [0]+bina
    bina.reverse()
    binb.reverse()
    n = len(bina)+len(binb)
    na = len(bina)
    q = QuantumRegister(n+1, 'q')
    circ = QuantumCircuit(q)
    for i in range(na):
        if bina[i]:
            circ.x(q[i])
    for i in range(len(binb)):
        if binb[i]:
            circ.x(q[na+i])
    add(circ, q, [q[i] for i in range(len(bina))], [q[i+na] for i in range(len(binb)-1)], q[n], q[na+len(binb)-1])
    circ_m = measure(circ, q, [i for i in range(na,n)])
    return circ_m


def addition_o(a, b):
    """mesure b = b+a, a reste a"""
    bina = [int(x) for x in bin(a)[2:]]
    binb = [int(x) for x in bin(b)[2:]]
    while len(bina)>=len(binb):
        binb = [0]+binb
    while len(bina)<len(binb)-1:
        bina = [0]+bina
    bina.reverse()
    binb.reverse()
    n = len(bina)+len(binb)
    na = len(bina)
    q = QuantumRegister(n+1, 'q')
    circ = QuantumCircuit(q)
    for i in range(na):
        if bina[i]:
            circ.x(q[i])
    for i in range(len(binb)):
        if binb[i]:
            circ.x(q[na+i])
    add_o(circ, q, [q[i] for i in range(len(bina))], [q[i+na] for i in range(len(binb)-1)], q[n], q[na+len(binb)-1])
    circ_m = measure(circ, q, [i for i in range(na,n)])
    return circ_m


def soustraction(a,b):
    """a = a, b = b-a, mesure b-a"""
    bina = [int(x) for x in bin(a)[2:]]
    binb = [int(x) for x in bin(b)[2:]]
    while len(bina)>=len(binb):
        binb = [0]+binb
    while len(bina)<len(binb)-1:
        bina = [0]+bina
    bina.reverse()
    binb.reverse()
    n = len(bina)+len(binb)
    na = len(bina)
    q = QuantumRegister(n+1, 'q')
    circ = QuantumCircuit(q)
    for i in range(na):
        if bina[i]:
            circ.x(q[i])
    for i in range(len(binb)):
        if binb[i]:
            circ.x(q[na+i])
    sub(circ, q, [q[i] for i in range(len(bina))], [q[i+na] for i in range(len(binb)-1)], q[n], q[na+len(binb)-1])
    circ_m = measure(circ, q, [i for i in range(na,n)])
    return circ_m


def addition_mod(a,b,nbr):
    """mesure b = b+a, a reste a"""
    bina = [int(x) for x in bin(a)[2:]]
    binb = [int(x) for x in bin(b)[2:]]
    binn = [int(x) for x in bin(nbr)[2:]]
    print(binn)
    while len(bina)>=len(binb):
        binb = [0]+binb
    while len(bina)<len(binb)-1:
        bina = [0]+bina
    while len(binn)<len(bina):
        binn = [0]+binn
    while len(binn)>len(bina):
        bina = [0]+bina
        binb = [0]+binb
    binn.reverse()
    bina.reverse()
    binb.reverse()
    print(bina, binb, binn)
    n = len(bina)+len(binb)+len(binn)
    na = len(bina)
    nab = len(bina)+len(binb)
    q = QuantumRegister(n+2, 'q')
    circ = QuantumCircuit(q)
    for i in range(na):
        if bina[i]:
            circ.x(q[i])
    for i in range(len(binb)):
        if binb[i]:
            circ.x(q[na+i])
    for i in range(len(binn)):
        if binn[i]:
            circ.x(q[nab+i])
    addmod(circ, q,# A, B, lost, last, N, lost2, binn):
           [q[i] for i in range(len(bina))],
           [q[i+na] for i in range(len(binb)-1)],
           q[n],
           q[na+len(binb)-1],
           [q[i+nab] for i in range(len(binn))],
           q[n+1],
           binn)
    circ_m = measure(circ, q, [i for i in range(na,nab)])
    return circ_m


def mult_mod(a,b,nbr,control):
    """mesure b = b*a, a reste a, if control else b = a"""
    bina = [int(x) for x in bin(a)[2:]]
    # binb = [int(x) for x in bin(b)[2:]]
    binn = [int(x) for x in bin(nbr)[2:]]
    while len(binn)<len(bina):
        binn = [0]+binn
    # print(bina, binn)
    binn.reverse()
    bina.reverse()
    n = len(bina)+len(binn)*3+1
    na = len(bina)
    nan = len(bina)+len(binn)#debut de Y
    nany = len(bina)+2*len(binn)+1#debut de "A" (ici c'est b)
    q = QuantumRegister(n+2+1, 'q')#+lost+lost2+control
    circ = QuantumCircuit(q)
    for i in range(na):
        if bina[i]:
            circ.x(q[i])
    for i in range(len(binn)):
        if binn[i]:
            circ.x(q[na+i])
    if control:
        circ.x(q[n+2])
    cmultmod(circ, q,#control, X, a, A, Y, n, N, binn, lost, lost2
           q[n+2],
           [q[i] for i in range(len(bina))],
           b,
           [q[i+nany] for i in range(len(binn))],
           [q[i+nan] for i in range(len(binn)+1)],
           nbr,
           [q[i+na] for i in range(len(binn))],
           binn,
           q[n],
           q[n+1])
    circ_m = measure(circ, q, [i for i in range(nan,nany)])
    return circ_m


def exp_mod(a,b,nbr):
    """mesure b = b*a, a reste a, if control else b = a"""
    bina = [int(x) for x in bin(a)[2:]]
    #binb = [int(x) for x in bin(b)[2:]]
    binn = [int(x) for x in bin(nbr)[2:]]
    #while len(binn)<len(bina):
    #    binn = [0]+binn
    #print(bina, binn)
    binn.reverse()
    bina.reverse()
    n = len(bina)+len(binn)*4+1
    na = len(bina)
    nan = len(bina)+len(binn)#debut de Y
    nany = len(bina)+2*len(binn)+1#debut de "A" (ici c'est b)
    nanya = len(bina)+3*len(binn)+1#debut de "APOW" (ce qui doit etre mesuré)
    q = QuantumRegister(n+2, 'q')#+lost+lost2
    circ = QuantumCircuit(q)
    for i in range(na):
        if bina[i]:
            circ.x(q[i])
    for i in range(len(binn)):
        if binn[i]:
            circ.x(q[na+i])
    expmod(circ, q,# X, a, A, APOW, Y, n, N, binn, lost, lost2)
           [q[i] for i in range(len(bina))],
           b%nbr,
           [q[i+nany] for i in range(len(binn))],
           [q[i+nanya] for i in range(len(binn))],
           [q[i+nan] for i in range(len(binn)+1)],
           nbr,
           [q[i+na] for i in range(len(binn))],
           binn,
           q[n],
           q[n+1])
    if len(bina)%2:
        circ_m = measure(circ, q, [i for i in range(nan,nany)])
    else:
        circ_m = measure(circ, q, [i for i in range(nanya,n)])
    #circ_m = measure(circ, q, [i for i in range(n)])
    return circ_m


def test_QFTn(n):
    q = QuantumRegister(n, 'q')#+lost+lost2
    circ = QuantumCircuit(q)
    circ.x(q[0])
    RegX = [q[i] for i in range(n)]
    QFTn(circ, q, RegX)
    print(RegX)
    iQFTn(circ, q, RegX)
    launch2(circ)
    circ_m = measure_direct(circ, q, RegX)
    return circ_m


# Tests
def tests_truth():
    circ_m = ccxtest(4)
    print(circ_m)
    circ_m = crootnxtest(4)
    print(circ_m)
    circ_m = oracletest(4)
    print(circ_m)
    circ_m = ccx_otest(4)
    print(circ_m)


def test_arith():
    n_max = 15
    a = r.randrange(2**(n_max//2))
    b = r.randrange(2**(n_max//2))
    circ_m = addition(a, b)
    print(circ_m.depth(), circ_m.width())
    #print(circ_m)
    print("{}+{} = ".format(a,b))
    print(launch(1, circ_m))

    a = r.randrange(2**(n_max//2))
    b = r.randrange(2**(n_max//2))
    circ_m = addition_o(a, b)
    print(circ_m.depth(), circ_m.width())
    #print(circ_m)
    print("{}+{} = ".format(a, b))
    print(launch(1, circ_m))

    a = r.randrange(2**(n_max//2))
    b = r.randrange(2**(n_max//2))
    circ_m = soustraction(a, b)
    print(circ_m.depth(), circ_m.width())
    # print(circ_m)
    print("{}-{} = ".format(a, b))
    print(launch(1, circ_m))

    n_max -= 1
    nbr = r.randrange(1, 2**(n_max//3))
    a = r.randrange(2**(n_max//3)) % nbr
    b = r.randrange(2**(n_max//3)) % nbr
    circ_m = addition_mod(a, b, nbr)
    print(circ_m.depth(), circ_m.width())
    # print(circ_m)
    print("{}+{}%{} = ".format(a, b, nbr))
    print(launch(1, circ_m))

    n_max -= 1
    print("Nmax", n_max)
    nbr = r.randrange(1, 2**(n_max//4))
    a = r.randrange(1, 2**(n_max//3)) % nbr
    b = r.randrange(1, 2**(n_max//3)) % nbr
    circ_m = mult_mod(a, b, nbr, True)
    print(circ_m.depth(), circ_m.width())
    # print(circ_m)
    print("{}*{}%{} = ".format(a, b, nbr))
    print(launch(1, circ_m))

    nbr = r.randrange(1, 2**(n_max//4))
    a = r.randrange(1, 2**(n_max//4))
    b = r.randrange(1, 2**(n_max//4)) % nbr
    circ_m = exp_mod(a, b, nbr)
    print(circ_m.depth(), circ_m.width())
    # print(circ_m)
    print("{}**{}%{} = ".format(b, a, nbr))
    print(launch(1, circ_m))


def test_draw():
    circ_m = test_QFTn(3)
    print(launch(1024, circ_m))
    fig = circ_m.draw(output='mpl', filename='C:/Users/RaphaelLambert/Pictures/test.png')


def test_24():
    n = 5
    q = QuantumRegister(n, 'q')  # +lost+lost2
    circ = QuantumCircuit(q)
    for i in range(4):
        circ.h(q[i])
    circ.ch(q[3], q[4])
    circ_m = measure(circ, q, [i for i in range(n)])
    counts = launch(2048, circ_m)
    print(counts, len(counts))

def test_stat():
    nt = 35
    age = [18, 19, 21, 20, 23, 22, 19, 19, 19, 19, 27, 24, 23, 18, 17, 24]
    age.append(29) # niklas
    n = len(age)
    mean = np.mean(age)
    var = np.std(age)
    diff = 1.96*var*(np.sqrt((nt-n)/(nt-1))/np.sqrt(n))
    print(mean, var, n, 1.96*var*(np.sqrt((nt-n)/(nt-1))/np.sqrt(n)), mean-diff, mean+diff)


def test_compar(K):
    K_int = int(np.ceil(K))
    n_k = len(bin(K_int))-1
    complement = np.binary_repr(-K_int, width=n_k)
    qr = QuantumRegister(5, 'q')
    qc = QuantumCircuit(qr)
    for i in range(3):
        qc.h(qr[i])
    qc.ccx(qr[0], qr[1], qr[3])
    for i in [2, 3, 4]:
        qc.x(qr[i])
    qc.ccx(qr[2], qr[3], qr[4])
    for i in [2, 3]:
        qc.x(qr[i])
    qc.ccx(qr[0], qr[1], qr[3])
    circ_m = measure(qc, qr, [i for i in range(5)])
    counts = launch(4000, circ_m)
    print(counts)
    print(complement)


def classical_kernel_estimation(in_train, in_test, labels):
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    for lab in labels:
        for datum in in_train[lab]:
            X_train.append(datum)
            Y_train.append(lab)
        for datum in in_test[lab]:
            X_test.append(datum)
            Y_test.append(lab)
    clf = svm.SVC(gamma='auto')
    clf.fit(X_train, Y_train)
    print(clf.score(X_test, Y_test))


def my_impl(in_train, in_test, labels):
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    for lab in labels:
        for datum in in_train[lab]:
            X_train.append(datum)
            Y_train.append(lab)
        for datum in in_test[lab]:
            X_test.append(datum)
            Y_test.append(lab)
    kernel_estimation(X_train, Y_train, X_test, Y_test)


def custom_constr(x, qr, inverse, depth):
    qc = QuantumCircuit(qr)
    maxi, mini = max(x), min(x)
    n = x.shape[0]
    qc_wv = Wavelets(n).construct_circuit(register=qr)
    for _ in range(depth):
        qc.h(qr)
        for i in range(n):
            qc.u1(2*np.pi*(x[i]-mini)/(maxi-mini), qr[i])
        for i in range(n):
            qc.cx(qr[i], qr[(i + 1) % n])
            qc.u1((2*np.pi)**2*(x[i]-mini)*(x[(i+1) % n]-mini)/(maxi-mini)**2, qr[(i + 1) % n])
            qc.cx(qr[i], qr[(i + 1) % n])
        qc = qc + qc_wv
    if inverse:
        return qc.inverse()
    return qc


def test_from_func(pres, nbr_by_label, nbr_by_label_test, nbr_comp, plot_graph, function, quantum_instance):
    print(pres)
    _, samp_train, samp_test, labels = function(nbr_by_label, nbr_by_label_test, nbr_comp, plot_graph)
    #print(samp_train)

    print("Success of the classical kernel:")
    classical_kernel_estimation(samp_train, samp_test, labels)

    # Generate the feature map
    feature_map = FirstOrderExpansion(feature_dimension=nbr_comp, depth=2)

    # Run the Quantum Kernel Estimator and classify the test data
    if len(labels) > 2:
        qsvm = QSVM(feature_map=feature_map, training_dataset=samp_train,
                    test_dataset=samp_test, multiclass_extension=AllPairs(_QSVM_Estimator, [feature_map]))
    else:
        qsvm = QSVM(feature_map=feature_map, training_dataset=samp_train,
                    test_dataset=samp_test)

    result = qsvm.run(quantum_instance)
    print("Success of the FirstOrder feature map kernel:")
    print(result['testing_accuracy'])

    # Generate the feature map
    feature_map = SecondOrderExpansion(feature_dimension=nbr_comp, depth=2)

    # Run the Quantum Kernel Estimator and classify the test data
    if len(labels) > 2:
        qsvm = QSVM(feature_map=feature_map, training_dataset=samp_train,
                    test_dataset=samp_test, multiclass_extension=AllPairs(_QSVM_Estimator, [feature_map]))
    else:
        qsvm = QSVM(feature_map=feature_map, training_dataset=samp_train,
                    test_dataset=samp_test)
    result = qsvm.run(quantum_instance)
    print("Success of the SecondOrder feature map kernel:")
    print(result['testing_accuracy'])

    if len(labels) == 2:
        print("Success for my implementation (second order):")
        my_impl(samp_train, samp_test, labels)

    feature_map = CustomExpansion(num_qubits=nbr_comp, constructor_function=custom_constr, feature_param=[1])

    if len(labels) > 2:
        qsvm = QSVM(feature_map=feature_map, training_dataset=samp_train,
                    test_dataset=samp_test, multiclass_extension=AllPairs(_QSVM_Estimator, [feature_map]))
    else:
        qsvm = QSVM(feature_map=feature_map, training_dataset=samp_train,
                    test_dataset=samp_test)
    result = qsvm.run(quantum_instance)
    print("Success of the Custom feature map kernel:")
    print(result['testing_accuracy'])

    return 0


def Sequence(nbr_by_label, nbr_by_label_test, nbr_comp, plot_graph):
    normal, muta = genere_chains(nbr_comp, 4)
    n_tot = (nbr_by_label + nbr_by_label_test + 1)
    X_n = [mutation(normal, [0.1, 0.1]) for _ in range(n_tot)]
    X_m = [mutation(muta, [0.1, 0.1]) for _ in range(n_tot)]
    X_crash_n = []
    X_crash_m = []
    for seq in X_n:
        crash = []
        for nucleotid in seq:
            crash.append((0 * (nucleotid == 'A') + 1 * (nucleotid == 'C') + 2 * (nucleotid == 'T') + 3 * (
                        nucleotid == 'G')) * np.pi / 2)
        X_crash_n.append(crash)
    for seq in X_m:
        crash = []
        for nucleotid in seq:
            crash.append((0 * (nucleotid == 'A') + 1 * (nucleotid == 'C') + 2 * (nucleotid == 'T') + 3 * (
                        nucleotid == 'G')) * np.pi / 2)
        X_crash_m.append(crash)
    X_n = np.array(X_crash_n)
    X_m = np.array(X_crash_m)
    if plot_graph:
        plt.scatter(X_n[:, 0][:nbr_by_label], X_n[:, 0][:nbr_by_label])
        plt.scatter(X_m[:, 0][:nbr_by_label], X_m[:, 0][:nbr_by_label])

        plt.title("ADN sequences")
        plt.show()
    training_input = {"N": X_n[:nbr_by_label], "M": X_m[:nbr_by_label]}
    test_input = {"N": X_n[nbr_by_label:n_tot], "M": X_m[nbr_by_label:n_tot]}
    return [X_n, X_m], training_input, test_input, ["N", "M"]



def test_svm():
    backend = BasicAer.get_backend('statevector_simulator')
    random_seed = 10598

    quantum_instance = QuantumInstance(backend, seed=random_seed, seed_transpiler=random_seed)

    # iris
    pres = "Test pour le data set Iris (facile, classique)"
    test_from_func(pres, 15, 10, 3, True, Iris, quantum_instance)

    # breast cancer
    pres = "Test pour le data set Breast Cancer (facile, classique)"
    #test_from_func(pres, 15, 10, 3, True, Breast_cancer, quantum_instance)

    # digits
    # pres = "Test pour le data set Digits (difficile, classique)"
    # test_from_func(pres, 10, 10, 10, True, Digits, quantum_instance)

    # wine
    pres = "Test pour le data set Wine (moyen, classique)"
    test_from_func(pres, 15, 10, 5, True, Wine, quantum_instance)

    # gaussian
    pres = "Test pour des données gaussiennes (moyen, classique)"
    test_from_func(pres, 15, 10, 2, True, Gaussian, quantum_instance)

    # small adn strings
    print("Test pour des séquences ADN courtes (difficile, classique)")
    test_from_func(pres, 15, 10, 14, True, Sequence, quantum_instance)


def test_svm_quantique():
    backend = BasicAer.get_backend('statevector_simulator')
    random_seed = 10598

    quantum_instance = QuantumInstance(backend, seed=random_seed, seed_transpiler=random_seed)
    pres = "Test pour des données générées par ordinateur quantique (facile, quantique)"
    print(pres)
    _, samp_train, samp_test, labels = ad_hoc_data(15, 10, 2, 0.3, True)
    sample_m, sample_p = stock_get(20, 0.3)

    labels_me = [-1, 1]
    samp_train_me = {-1: np.array(sample_m[:15]), 1: np.array(sample_p[:15])}
    samp_test_me = {-1: np.array(sample_m[15:]), 1: np.array(sample_p[15:])}
    print(samp_train)
    print(samp_train_me)
    print(samp_test)
    print(samp_test_me)
    classical_kernel_estimation(samp_train, samp_test, labels)
    classical_kernel_estimation(samp_train_me, samp_test_me, labels_me)
    # Generate the feature map
    feature_map = FirstOrderExpansion(feature_dimension=2, depth=2)

    # Run the Quantum Kernel Estimator and classify the test data
    qsvm = QSVM(feature_map=feature_map, training_dataset=samp_train,
                test_dataset=samp_test)
    qsvm_me = QSVM(feature_map=feature_map, training_dataset=samp_train_me,
                test_dataset=samp_test_me)

    result = qsvm.run(quantum_instance)
    result_me = qsvm_me.run(quantum_instance)
    print("Success of the FirstOrder feature map kernel:")
    print(result['testing_accuracy'])
    print(result_me['testing_accuracy'])

    # Generate the feature map
    feature_map = SecondOrderExpansion(feature_dimension=2, depth=2)

    # Run the Quantum Kernel Estimator and classify the test data
    qsvm = QSVM(feature_map=feature_map, training_dataset=samp_train,
                test_dataset=samp_test)
    qsvm_me = QSVM(feature_map=feature_map, training_dataset=samp_train_me,
                test_dataset=samp_test_me)

    result = qsvm.run(quantum_instance)
    result_me = qsvm_me.run(quantum_instance)
    print("Success of the SecondOrder feature map kernel:")
    print(result['testing_accuracy'])
    print(result_me['testing_accuracy'])

    print("Success for my implementation (second order):")
    my_impl(samp_train, samp_test, labels)
    my_impl(samp_train_me, samp_test_me, labels_me)


# test_svm_quantique()
test_svm()
# test_compar(1.9)
# test_stat()
# test_24()
# test_arith()
# test_QFTn(3)
# test_draw()
