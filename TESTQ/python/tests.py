from General import *
from svm import *
from gan_charging_back import *
from custom_map import CustomExpansion
from sklearn import svm
from qiskit.aqua import QuantumInstance
from qiskit.aqua.components.feature_maps import SecondOrderExpansion, FirstOrderExpansion
from qiskit.aqua.algorithms import QSVM
from qiskit.aqua.algorithms.many_sample.qsvm._qsvm_estimator import _QSVM_Estimator
from qiskit.aqua.components.multiclass_extensions.all_pairs import *
from qsvm_datasets import *
from wavelets import Wavelets
from qiskit.aqua.algorithms.adaptive.qgan import *
from qiskit.visualization import plot_state_city
from qiskit.ignis.mitigation.measurement import (complete_meas_cal, tensored_meas_cal,
                                                 CompleteMeasFitter, TensoredMeasFitter)


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
        q = QuantumRegister(n+1, 'q')
        circ = QuantumCircuit(q)
        for i in range(n):
            circ.h(q[i])
        # for i in bi:
        #    circ.x(q[i])
        oracle_o(circ, q, n, bi)
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


def addition(a, b):
    """mesure b = b+a, a reste a"""
    bina = [int(x) for x in bin(a)[2:]]
    binb = [int(x) for x in bin(b)[2:]]
    while len(bina) >= len(binb):
        binb = [0]+binb
    while len(bina) < len(binb)-1:
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
    circ_m = measure(circ, q, [i for i in range(na, n)])
    return circ_m


def addition_o(a, b):
    """mesure b = b+a, a reste a"""
    bina = [int(x) for x in bin(a)[2:]]
    binb = [int(x) for x in bin(b)[2:]]
    while len(bina) >= len(binb):
        binb = [0]+binb
    while len(bina) < len(binb)-1:
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
    circ_m = measure(circ, q, [i for i in range(na, n)])
    return circ_m


def soustraction(a,b):
    """a = a, b = b-a, mesure b-a"""
    bina = [int(x) for x in bin(a)[2:]]
    binb = [int(x) for x in bin(b)[2:]]
    while len(bina) >= len(binb):
        binb = [0]+binb
    while len(bina) < len(binb)-1:
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
    circ_m = measure(circ, q, [i for i in range(na, n)])
    return circ_m


def addition_mod(a, b, nbr):
    """mesure b = b+a, a reste a"""
    bina = [int(x) for x in bin(a)[2:]]
    binb = [int(x) for x in bin(b)[2:]]
    binn = [int(x) for x in bin(nbr)[2:]]
    #print(binn)
    while len(bina) >= len(binb):
        binb = [0]+binb
    while len(bina) < len(binb)-1:
        bina = [0]+bina
    while len(binn) < len(bina):
        binn = [0]+binn
    while len(binn) > len(bina):
        bina = [0]+bina
        binb = [0]+binb
    binn.reverse()
    bina.reverse()
    binb.reverse()
    #print(bina, binb, binn)
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
    addmod(circ, q,  # A, B, lost, last, N, lost2, binn):
           [q[i] for i in range(len(bina))],
           [q[i+na] for i in range(len(binb)-1)],
           q[n],
           q[na+len(binb)-1],
           [q[i+nab] for i in range(len(binn))],
           q[n+1],
           binn)
    circ_m = measure(circ, q, [i for i in range(na,nab)])
    return circ_m


def mult_mod(a, b, nbr, control):
    """mesure b = b*a, a reste a, if control else b = a"""
    bina = [int(x) for x in bin(a)[2:]]
    # binb = [int(x) for x in bin(b)[2:]]
    binn = [int(x) for x in bin(nbr)[2:]]
    while len(binn) < len(bina):
        binn = [0]+binn
    # print(bina, binn)
    binn.reverse()
    bina.reverse()
    n = len(bina)+len(binn)*3+1
    na = len(bina)
    nan = len(bina)+len(binn)  # debut de Y
    nany = len(bina)+2*len(binn)+1  # debut de "A" (ici c'est b)
    q = QuantumRegister(n+2+1, 'q')  # +lost+lost2+control
    circ = QuantumCircuit(q)
    for i in range(na):
        if bina[i]:
            circ.x(q[i])
    for i in range(len(binn)):
        if binn[i]:
            circ.x(q[na+i])
    if control:
        circ.x(q[n+2])
    cmultmod(circ, q,  # control, X, a, A, Y, n, N, binn, lost, lost2
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


def exp_mod(a, b, nbr):
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
    nan = len(bina)+len(binn)  # debut de Y
    nany = len(bina)+2*len(binn)+1  # debut de "A" (ici c'est b)
    nanya = len(bina)+3*len(binn)+1  # debut de "APOW" (ce qui doit etre mesuré)
    q = QuantumRegister(n+2, 'q')  # +lost+lost2
    circ = QuantumCircuit(q)
    for i in range(na):
        if bina[i]:
            circ.x(q[i])
    for i in range(len(binn)):
        if binn[i]:
            circ.x(q[na+i])
    expmod(circ, q,  # X, a, A, APOW, Y, n, N, binn, lost, lost2)
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
    """A test for the QFT circuit"""
    q = QuantumRegister(n, 'q')  # +lost+lost2
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
    """To check if everything is correct"""
    circ_m = ccxtest(4)
    print(circ_m)
    circ_m = crootnxtest(4)
    print(circ_m)
    circ_m = oracletest(4)
    print(circ_m)
    circ_m = ccx_otest(4)
    print(circ_m)


def test_arith():
    """Arithmetic checks"""
    n_max = 17
    a = r.randrange(2**(n_max//2))
    b = r.randrange(2**(n_max//2))
    circ_m = addition(a, b)
    print(circ_m.depth(), circ_m.width())
    #print(circ_m)
    print("{}+{} = ".format(a,b))
    lect_bin(launch(1, circ_m))

    a = r.randrange(2**(n_max//2))
    b = r.randrange(2**(n_max//2))
    circ_m = addition_o(a, b)
    print(circ_m.depth(), circ_m.width())
    #print(circ_m)
    print("{}+{} = ".format(a, b))
    lect_bin(launch(1, circ_m))

    a = r.randrange(2**(n_max//2))
    b = r.randrange(2**(n_max//2))
    circ_m = soustraction(a, b)
    print(circ_m.depth(), circ_m.width())
    # print(circ_m)
    print("{}-{} = ".format(a, b))
    lect_bin(launch(1, circ_m))

    n_max -= 1
    nbr = r.randrange(1, 2**(n_max//3))
    a = r.randrange(2**(n_max//3)) % nbr
    b = r.randrange(2**(n_max//3)) % nbr
    circ_m = addition_mod(a, b, nbr)
    print(circ_m.depth(), circ_m.width())
    # print(circ_m)
    print("{}+{}%{} = ".format(a, b, nbr))
    lect_bin(launch(1, circ_m))

    n_max -= 1
    print("Nmax", n_max)
    nbr = r.randrange(1, 2**(n_max//4))
    a = r.randrange(1, 2**(n_max//3)) % nbr
    b = r.randrange(1, 2**(n_max//3)) % nbr
    circ_m = mult_mod(a, b, nbr, True)
    print(circ_m.depth(), circ_m.width())
    # print(circ_m)
    print("{}*{}%{} = ".format(a, b, nbr))
    lect_bin(launch(1, circ_m))

    nbr = r.randrange(1, 2**(n_max//4))
    a = r.randrange(1, 2**(n_max//4))
    b = r.randrange(1, 2**(n_max//4)) % nbr
    circ_m = exp_mod(a, b, nbr)
    print(circ_m.depth(), circ_m.width())
    # print(circ_m)
    print("{}**{}%{} = ".format(b, a, nbr))
    lect_bin(launch(1, circ_m))


def test_draw():
    """To check for issues in drawing circuits"""
    circ_m = test_QFTn(3)
    print(launch(1024, circ_m))
    fig = circ_m.draw(output='mpl', filename='C:/Users/RaphaelLambert/Pictures/test.png')
    return fig


def test_24():
    """Test for controlled H"""
    n = 5
    q = QuantumRegister(n, 'q')
    circ = QuantumCircuit(q)
    for i in range(4):
        circ.h(q[i])
    circ.ch(q[3], q[4])
    circ_m = measure(circ, q, [i for i in range(n)])
    counts = launch(2048, circ_m)
    print(counts, len(counts))


def test_stat():
    """A statistical test"""
    nt = 35
    age = [18, 19, 21, 20, 23, 22, 19, 19, 19, 19, 27, 24, 23, 18, 17, 24, 29]
    # age.append(29)  # niklas
    n = len(age)
    mean = np.mean(age)
    var = np.std(age)
    diff = 1.96*var*(np.sqrt((nt-n)/(nt-1))/np.sqrt(n))
    print(mean, var, n, 1.96*var*(np.sqrt((nt-n)/(nt-1))/np.sqrt(n)), mean-diff, mean+diff)


def test_compar(K):
    """No idea why I did that"""
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
    """will give the score of a classical kernel, given train and test data, and the labels"""
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
    """The same kernel test, but with quantum kernel"""
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


def my_impl_variational(in_train, in_test, labels):
    """The subroutine for variationnal learning"""
    X_train = []
    X_test = []
    for lab in labels:
        for datum in in_train[lab]:
            X_train.append([datum, lab])
        for datum in in_test[lab]:
            X_test.append([datum, lab])
    Variationer_learn(X_train, 500, 1, 0.01, X_test, labels)


def custom_constr(x, qr, inverse, depth):
    """A constructor for a circuit for the quantum kernel, using a different method than the paper"""
    qc = QuantumCircuit(qr)
    maxi, mini = max(x), min(x)
    n = x.shape[0]
    #qc_wv = Wavelets(n).construct_circuit(register=qr)
    for _ in range(depth):
        qc.h(qr)
        for i in range(n):
            qc.u2(np.pi*(x[(i+1) % n]-mini)/(maxi-mini), 2*np.pi*(x[i]-mini)/(maxi-mini), qr[i])
        for i in range(n):
            qc.cx(qr[i], qr[(i + 1) % n])
            qc.u2(np.pi*(x[(i+1) % n]-mini)/(maxi-mini),
                  ((2*np.pi)**2*(x[i]-mini)*(x[(i+1) % n]-mini)/(maxi-mini)**2) % 2*np.pi,
                  qr[(i + 1) % n])
            qc.cx(qr[i], qr[(i + 1) % n])
        #qc = qc + qc_wv
    if inverse:
        return qc.inverse()
    return qc


def concat_succ(L):
    """return the successive concatenation"""
    if len(L) < 2:
        return L
    res = []
    last = L.pop()
    othe = L.pop()
    for i in last:
        for j in othe:
            if type(i) is list:
                if type(j) is list:
                    res.append(i+j)
                else:
                    res.append(i+[j])
            elif type(j) is list:
                res.append([i] + j)
            else:
                res.append([i] + [j])
    L = [res] + L
    return concat_succ(L)


def test_from_func(pres, nbr_by_label, nbr_by_label_test, nbr_comp, plot_graph, function, quantum_instance):
    """A usefull subroutine to make tests efficients for kernel and varitationnal learning"""
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
    if nbr_comp == 0:
        total_map = qsvm.predict(np.array(concat_succ([[i/20 for i in range(-3*20, 3*20)] for _ in range(nbr_comp)])[0]), quantum_instance)
        print(total_map)
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


def test_from_func_variational(pres, nbr_by_label, nbr_by_label_test, nbr_comp, plot_graph, function):
    """The same function but with variationnal models"""
    print(pres)
    _, samp_train, samp_test, labels = function(nbr_by_label, nbr_by_label_test, nbr_comp, plot_graph)

    if len(labels) == 2:
        print("Success for my implementation (second order, variational):")
        my_impl_variational(samp_train, samp_test, labels)

    return 0


def Sequence(nbr_by_label, nbr_by_label_test, nbr_comp, plot_graph):
    """A generator for random DNA sequences and a mutated version."""
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
    """The full SVM tests"""
    backend = BasicAer.get_backend('statevector_simulator')
    random_seed = r.randint(1, 10598)

    quantum_instance = QuantumInstance(backend, seed=random_seed, seed_transpiler=random_seed)

    # iris
    pres = "Test pour le data set Iris (facile, classique)"
    test_from_func(pres, 15, 10, 3, True, Iris, quantum_instance)

    # breast cancer
    pres = "Test pour le data set Breast Cancer (facile, classique)"
    test_from_func(pres, 15, 10, 3, True, Breast_cancer, quantum_instance)

    # digits (it's long so be careful)
    #pres = "Test pour le data set Digits (difficile, classique)"
    #test_from_func(pres, 10, 10, 10, True, Digits, quantum_instance)

    # wine
    pres = "Test pour le data set Wine (moyen, classique)"
    test_from_func(pres, 15, 10, 5, True, Wine, quantum_instance)

    # gaussian
    pres = "Test pour des données gaussiennes (moyen, classique)"
    for _ in range(1):
        print("\n")
        print("New iteration")
        test_from_func(pres, 25, 10, 2, True, Gaussian, quantum_instance)
        print("\n")

    # small adn strings
    pres = "Test pour des séquences ADN courtes (difficile, classique)"
    test_from_func(pres, 10, 15, 14, True, Sequence, quantum_instance)


def test_svm_quantique():
    """To test the SVM models on quantum computer generated data"""
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

    # Last implementation using the custom circuit generator
    print("Success for my implementation (second order):")
    my_impl(samp_train, samp_test, labels)
    my_impl(samp_train_me, samp_test_me, labels_me)

    feature_map = CustomExpansion(num_qubits=2, constructor_function=custom_constr, feature_param=[1])

    qsvm = QSVM(feature_map=feature_map, training_dataset=samp_train,
                test_dataset=samp_test)
    qsvm_me = QSVM(feature_map=feature_map, training_dataset=samp_train_me,
                   test_dataset=samp_test_me)

    result = qsvm.run(quantum_instance)
    result_me = qsvm_me.run(quantum_instance)
    print("Success of the Custom feature map kernel:")
    print(result['testing_accuracy'])
    print(result_me['testing_accuracy'])


def test_variational():
    """The full tests with variationnal learning. Not ready for multi-label"""
    # iris
    #pres = "Test pour le data set Iris (facile, classique)"
    #test_from_func_variational(pres, 15, 10, 3, True, Iris)

    # breast cancer
    pres = "Test pour le data set Breast Cancer (facile, classique)"
    test_from_func_variational(pres, 15, 10, 3, True, Breast_cancer)

    # digits
    # pres = "Test pour le data set Digits (difficile, classique)"
    # test_from_func(pres, 10, 10, 10, True, Digits, quantum_instance)

    # wine
    # pres = "Test pour le data set Wine (moyen, classique)"
    # test_from_func(pres, 15, 10, 5, True, Wine, quantum_instance)

    # gaussian
    pres = "Test pour des données gaussiennes (moyen, classique)"
    for _ in range(1):
        print("\n")
        print("New iteration")
        test_from_func_variational(pres, 25, 10, 2, True, Gaussian)
        print("\n")

    # small adn strings
    pres = "Test pour des séquences ADN courtes (difficile, classique)"
    test_from_func_variational(pres, 10, 15, 14, True, Sequence)

    #Quantum data
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

    my_impl_variational(samp_train, samp_test, labels)
    print("Pour autres données quantiques")
    my_impl_variational(samp_train_me, samp_test_me, labels_me)


def general_gantest(proba, nbr_qubits):
    """All tests for gan model"""
    for m in [4096, 2048]:
        for l in [1, 2, 3]:
            print("Easy mode results for m={} and l={}:".format(m, l))
            Variationer_learn_gan(1000, l, m, proba=proba, n=nbr_qubits, distri_size=0, easy=True)
            print("\n")
            print("Distribution learning results for m={} and l={}:".format(m, l))
            for d in [256, 512]:
                print("For ", d, ": ")
                Variationer_learn_gan(1000, l, m, proba=proba, n=nbr_qubits, distri_size=d, easy=False)
            print("Singleton learning results for m={} and l={}:".format(m, l))
            Variationer_learn_gan(1000, l, m, proba=proba, n=nbr_qubits, distri_size=0, easy=False)


def test_gan_qiskit(n, Database):
    """Test for in qiskit gan"""
    mini = np.min(Database)
    maxi = np.max(Database)
    h = (maxi - mini) / (2 ** n)
    bins = [[k for d in Database if mini + h * k < d < mini + h * (k + 1)] for k in range(2 ** n)]
    interv = [mini + h * k for k in range(2 ** n)]
    backend = BasicAer.get_backend('statevector_simulator')
    random_seed = 10598

    quantum_instance = QuantumInstance(backend, seed=random_seed, seed_transpiler=random_seed)
    gan_test = QGAN(Database, num_qubits=[n], snapshot_dir=None,
                    quantum_instance=quantum_instance, batch_size=int(len(Database) / 20), num_epochs=300)
    gan_test.train()
    samp, bins_var = gan_test.generator.get_output(gan_test.quantum_instance, shots=4096)

    compar = [len(b) / len(Database) for b in bins]
    if len(interv) == len(compar):
        plt.plot(interv, compar)

    plt.plot(interv, bins_var)

    plt.show()


def test_gan():
    """Launch the subroutines for GAN test"""
    nbr_qubits = 5

    # Normal law
    # N = 5*10 ** 3
    #
    # Database = np.random.normal(0, 1, N)
    # test_gan_qiskit(nbr_qubits, Database)

    # beta
    arr_beta = beta_proba(nbr_qubits, 2, 5)

    general_gantest(arr_beta, nbr_qubits)

    # uniform not on [0, 32]
    if nbr_qubits == 5:
        arr_unif = [1 / 24] * 24 + 8 * [0]
        general_gantest(arr_unif, nbr_qubits)


def test_imag(online=False):
    """All the tests for the imag presentation. """
    from qiskit.providers.aer import noise
    from qiskit import Aer

    # Calibration
    qr = QuantumRegister(2)
    qubit_list = [0, 1]
    meas_calibs, state_labels = complete_meas_cal(qubit_list=qubit_list, qr=qr, circlabel='mcal')

    # Execute the calibration circuits without noise
    backend = Aer.get_backend('qasm_simulator')
    job = execute(meas_calibs, backend=backend, shots=1000)
    cal_results = job.result()

    # The calibration matrix without noise is the identity matrix
    meas_fitter = CompleteMeasFitter(cal_results, state_labels, circlabel='mcal')
    print(meas_fitter.cal_matrix)
    T1 = [50e3,100e3]
    T2 = [1e3, 1e3]  # arbitrarily chosen T2 times
    time_measure = 10e3  # arbitrarily chosen measurement time
    noise_thermal = noise.NoiseModel()

    for j in range(2):
        noise_thermal.add_quantum_error(noise.errors.standard_errors.thermal_relaxation_error(T1[j], T2[j], time_measure), "measure", [j])

    backend = Aer.get_backend('qasm_simulator')
    job = execute(meas_calibs, backend=backend, shots=1000, noise_model=noise_thermal)
    cal_results = job.result()
    meas_fitter = CompleteMeasFitter(cal_results, state_labels, qubit_list=qubit_list, circlabel='mcal')
    print(meas_fitter.cal_matrix)
    meas_fitter.plot_calibration()
    plt.show()

    # ensimag presentation
    presentation_imag(online)


def test_new():
    """Tests for the new oracles for Grover"""
    from qiskit import BasicAer
    from qiskit.aqua.algorithms import Grover
    from qiskit.aqua.components.oracles import LogicalExpressionOracle

    expr = "your logical expression goes here"
    algorithm = Grover(LogicalExpressionOracle(expr))
    backend = BasicAer.get_backend('qasm_simulator')
    result = algorithm.run(backend, seed=101110)
    print(result)


def test_dag():
    """Test for the DAG generation in transpilation"""
    qr = QuantumRegister(5, 'qr')
    cr = ClassicalRegister(5, 'cr')
    ghz = QuantumCircuit(qr, cr, name='ghz')

    ghz.h(qr[2])
    ghz.cx(qr[2], qr[1])
    ghz.cx(qr[1], qr[0])
    ghz.cx(qr[2], qr[3])
    ghz.cx(qr[3], qr[4])
    ghz.draw()

    # ghz_dag = circuit_to_dag(ghz)

    # print(ghz.width(), ghz_dag.width())


def ab_finder(Func, Ind):
    """Find some values in a function. Is used for the SAT study"""
    b = []
    for i in range(len(Func)):
        for j in range(i+1, len(Func)):
            b.append(np.log(Func[i]/Func[j])/np.log(Ind[j]/Ind[i]))
    print(b)
    plt.plot([i for i in range(len(b))], b)
    b = np.mean(b)
    a = np.mean([Func[i]*(Ind[i]**b) for i in range(len(Func))])
    return a, b


def norm1(f, g):
    """|f-g|_1"""
    res = 0
    for i in range(len(f)):
        res += np.abs(f[i]-g[i])
    return res


def test_fonction():
    """Results of computation for SAT study"""
    Func = [[7729.018255678793, 140.153834155207, 68.77595919655846, 31.62018118184545, 18.030431610812485, 11.480451328936848, 8.854799040173322, 5.891748736768329, 4.107058029460621, 3.525987646397012, 2.6501857762543453, 1.9939336429398156, 1.796115967192535, 1.3439730213174272, 1.0573728322694307, 0.9370165183504918, 0.6862225806758537, 0.58629480789044, 0.46467717773394074, 0.4351295050299971, 0.31030829231196316, 0.18283441858118177, 0.2508750473787763, 0.12603102215466033, 0.1403733845624147, 0.208944572364959, 0.05937056209629393, 0.06406561737973851, 0.02549828229037716, 0.044190126138167286, 0.12220850634047802, 0.07250107250107221, 0.008166145780824684, 1.0000000000000118, 0.0016025641025641038, 0.00644122383252819, 0.0016025641025641038], [0.9610949812702193, 0.4109019899274278, 0.3089377623397382, 0.23495840772645324, 0.20677330100735603, 0.19949331148184576, 0.19889217797273162, 0.1983236356606282, 0.20419249563878353, 0.22439671222315674, 0.2468001421725052, 0.26934377202851223, 0.3026764978536294, 0.318102154625913, 0.36021755759452945, 0.3948630408193794, 0.4151320064818989, 0.4874842761804363, 0.5224160769563155, 0.5581950422944579, 0.5694688385000937, 0.55031751183993, 0.6849362225850419, 0.6075803939330335, 0.7350416126522452, 0.8427100469155232, 0.6733603675051251, 0.7902933382920369, 0.6836315434546335, 0.8276834938319588, 0.9487611479434883, 0.9330768568229448, 0.8193885540523317, 0.997, 0.6670224119530418, 0.8010253123998721, 0.6670224119530418]]
    Func2 = [[351.7858724605074, 86.17819599440456, 40.60817807215555, 22.41370879569776, 13.71370737429577, 9.380805367958237, 6.500771030437166, 5.1934818237317595, 3.8672723604183825, 2.9429262287072286, 2.2293044011542276, 1.7413726101499962, 1.4110020679698105, 1.145285045596285, 0.8493079536648233, 0.7075504964039413, 0.5314726701362551, 0.3946934984396482, 0.35765493303260854, 0.26720947446336063, 0.20483675723548622, 0.16768018253752273, 0.1348248093028533, 0.09131928959311625, 0.06352807714123655, 0.10443083847008712, 0.08486764614717036, 0.07948616458565717, 0.034928848641655796, 0.01957585644371944, 0.0101010101010101, 0.0060868252930227846, 0.003034178389628626, 1.0000000000000095, 0.00040016006402561054, 0.999999999998485, 1.0000000000005178], [0.4014557102478067, 0.19662718856027814, 0.14714271380376454, 0.12376760998544561, 0.11465015678691455, 0.11701257764627322, 0.11971607647950641, 0.12500413021169918, 0.13585072294949355, 0.15396764607614966, 0.1688776087223454, 0.19466317245480091, 0.22196728054180964, 0.25356761298043673, 0.2838683431502283, 0.32214826256665563, 0.3474420903976237, 0.38101744922057085, 0.4472615852893438, 0.4838915064306636, 0.5092442536661715, 0.5393723754562065, 0.6281883708635211, 0.6075021365541242, 0.5864414731409806, 0.8093479024729197, 0.8515064760519562, 0.8942467588311623, 0.7951232441028355, 0.8030838419531001, 0.8347245409015025, 0.7345183562814673, 0.5796857049505159, 0.9992, 0.500100020004001, 0.9997999999999997, 0.9996000000000002]]
    Func3 = [[0, 587.8353916144741, 122.51606563339197, 43.412900839437754, 23.57555460166956, 15.269810446375592,
              10.302086288261853, 6.559749680166944, 4.733027276883256, 3.5899442373476638, 2.6188091438741785,
              1.8973511635960179, 1.59962405851026, 1.1900558241207209, 0.9154742469037501, 0.6663774582713599,
              0.5510310065149937, 0.3962213545720932, 0.34948402621422503, 0.25056075858741617, 0.2104583602324079,
              0.14271205720517338, 0.14355391510677562, 0.09448580768732265, 0.09563164108618655, 0.05255180659992084,
              0.05056730634542649, 0.04830593760483059, 0.017362995116657634, 0.018563316442158327,
              0.008064516129032258,
              1.000000000000367, 0.007556675062972293],
             [0, 0.3286716474211523, 0.13242848819742412, 0.08446949128973799, 0.06942310179309806, 0.06710098910557027,
              0.0672674712138261, 0.07277714425838515, 0.07860653317204705, 0.08731142910806171, 0.10727105079265914,
              0.11444337732064816, 0.1388584129446931, 0.1545716580456971, 0.1909675139048617, 0.2078287621915297,
              0.24481225152845482, 0.2809639446229486, 0.3427375708843379, 0.3563349568749672, 0.44658806716684524,
              0.4323139797237823, 0.5209648901237686, 0.5384242118067787, 0.6800151114469211, 0.6244881568587771,
              0.6892348767357834, 0.8342138924420222, 0.7306736811340057, 0.7746555651829621, 0.6684491978609626,
              0.9988000000000005, 0.7158196134574087]]
    Func4 = [[15748.752858254895, 191.911645065407, 73.47842499410935, 30.41633464294201, 19.98626405383976, 11.452154438755791, 7.133577668465118, 5.558028573739012, 3.9221979491511965, 2.883103216560325, 2.134843944573437, 1.672594713096574, 1.2840603701894708, 0.9760043686353028, 0.7383287083076139, 0.5201345636008389, 0.3922546743466271, 0.3075443182363077, 0.2190805647347894, 0.20447301461609166, 0.13718384831088948, 0.10116884193975774, 0.06605172314636748, 0.05724132569770158, 0.05521472392638043, 0.028443817583450853, 0.030927835051546386, 0.020842194810718865, 0.014610389610389598, 0.008522010992889928, 0.999999999998485, 0.004924128228318766, 0.9999999999999153], [0.8339118479799609, 0.09465261480352483, 0.05645445767221664, 0.04042946161369488, 0.039816125266879804, 0.038808660214436394, 0.04108030531513686, 0.04722150007344185, 0.05756881734809522, 0.06174501095337855, 0.07754371601798399, 0.08835218320247393, 0.11277873469258441, 0.14265723594157298, 0.1591593134391955, 0.18294405416467058, 0.20022650693883753, 0.25880068050710453, 0.28508276563948803, 0.38530886060602676, 0.3827846845148159, 0.3989797846053703, 0.4321293982608086, 0.549123155471954, 0.6479481641468685, 0.6177554137838079, 0.7204610951008645, 0.7485830392471395, 0.8589097572148419, 0.7662293265433628, 0.9997999999999997, 0.7786256145580744, 0.9994]]
    Func5 = [[6025.007118984573, 201.67078346958766, 61.164046929888165, 28.570275345356297, 16.42844389234335, 10.118865208041463, 6.5569441993908315, 5.115524806237155, 3.670230880870961, 2.578942628606752, 1.9336145754313054, 1.3128869676600932, 1.0509490349325865, 0.8260606265098674, 0.6192337836269376, 0.39548178329078904, 0.3125342768364706, 0.2535171070520563, 0.17950996366785027, 0.1589593169850413, 0.08776166654828464, 0.07186902085373958, 0.05087852147014008, 0.04031209362808843, 0.015025041736227053, 0.017452299442196745, 0.017452299442196745], [0.4900449307377233, 0.052013394197577424, 0.02763077179222995, 0.022073148401405642, 0.021551468349526978, 0.022874943012867084, 0.025129502524592704, 0.0290677734501479, 0.03586807797479691, 0.04200099407612618, 0.05395970879783885, 0.06222885447396381, 0.08011656818055503, 0.10740529383718123, 0.13349486346877767, 0.14548961984802525, 0.16651317027639345, 0.2157377384182918, 0.2536490552393858, 0.3281846996863911, 0.3205769005384313, 0.37653580728960656, 0.42583822467919963, 0.5652911249293386, 0.5004170141784822, 0.6177302303447393, 0.6177302303447393]]
    Func6 = [[np.mean([4447.600676225763, 4028.6041733343945]), 1122.3591601299543, np.mean([566.3324626056967, 560.7330884421851]), 330.62666503822817, 210.43180030971985, 149.0424349106643, 82.3987852348437, 44.547601156494196, 26.426565843449865, 17.824560051495858, 13.783607579356286, 10.40131640361244, 7.455414078577158, 5.499251804904008, 4.302343361171181, 3.539763892078505, 2.784540963982787, 2.2236920038741386, 1.7559944701404429, 1.4584556538378022, 1.200557790786275, 1.0699270271332681, 0.8634699222550425, 0.7434193327535079, 0.5714709844638692, 0.4396553281521431, 0.3818399186003246, 0.32953426555502685, 0.24102244704033102, 0.22355346426540507, 0.15212505517750333, 0.16001860110466576, 0.10928264810439764], [np.mean([0.2617607830185082, 0.24308742821120072]), 0.0927423079289474, 0.05568821342607846, 0.03793131631871315, 0.0278257161673426, 0.022656568305311583, 0.016435055935377363, 0.013349794613250844, 0.011828502037667653, 0.011907722737912936, 0.011906786143346124, 0.01350556077146206, 0.014414705674904129, 0.015921318603495407, 0.018673975709110842, 0.02270468991422641, 0.02309589946853888, 0.027514993109446845, 0.032185252317963355, 0.039533712217873974, 0.04836171488037051, 0.05571282912235797, 0.0662400500529835, 0.0842953982688298, 0.09463525780280219, 0.11078755407852524, 0.12333175657275465, 0.1523270728384864, 0.1633804430013571, 0.2132018494926845, 0.20858345530936606, 0.26749188764633536, 0.28116189267776115]]

    index = np.linspace(0.5, 11.3, 37)
    index3 = np.linspace(0.1, 9.7, 33)
    index4 = np.linspace(0.2, 9.8, 33)
    index5 = np.linspace(0.2, 7.8, 27)
    index6 = [0.15, 0.25, 0.3, 0.4, 0.45]+[0.5+i*0.2 for i in range(28)]

    n = 100
    b = np.log(Func2[0][4]/Func2[0][5])/np.log(2/1.7)
    a = Func2[0][4]*(1.7**b)
    print(a, b)
    a_t, b_t = ab_finder(Func[0][3:20], index[3:20])
    a_t2, b_t2 = ab_finder(Func2[0][3:20], index[3:20])
    a_t3, b_t3 = ab_finder(Func3[0][3:20], index3[3:20])
    a_t4, b_t4 = ab_finder(Func4[0][3:20], index4[3:20])
    a_t5, b_t5 = ab_finder(Func5[0][3:20], index5[3:20])
    a_t6, b_t6 = ab_finder(Func6[0][3:20], index6[3:20])
    plt.show()
    print(a_t, b_t, a_t2, b_t2, a_t3, b_t3, a_t4, b_t4, a_t5, b_t5, a_t6, b_t6)

    plt.plot(index, Func[0], color='purple')
    #plt.plot(index, Func2[0], color='red')
    plt.plot(index3, Func3[0], color='black')
    #plt.plot(index4, Func4[0], color='blue')
    plt.plot(index5, Func5[0], color='yellow')
    plt.plot(index6, Func6[0], color='orange')
    index_comp = np.linspace(0.1+1/n, 11, n)
    #plt.plot(index_comp, a/((index_comp)**b), color='blue')
    plt.plot(index_comp, a_t6/(index_comp**b_t6), color='green')
    plt.show()


def test_fonction_p():
    """Other computation for SAT study"""
    Func = [[7729.018255678793, 140.153834155207, 68.77595919655846, 31.62018118184545, 18.030431610812485,
             11.480451328936848, 8.854799040173322, 5.891748736768329, 4.107058029460621, 3.525987646397012,
             2.6501857762543453, 1.9939336429398156, 1.796115967192535, 1.3439730213174272, 1.0573728322694307,
             0.9370165183504918, 0.6862225806758537, 0.58629480789044, 0.46467717773394074, 0.4351295050299971,
             0.31030829231196316, 0.18283441858118177, 0.2508750473787763, 0.12603102215466033, 0.1403733845624147,
             0.208944572364959, 0.05937056209629393, 0.06406561737973851, 0.02549828229037716, 0.044190126138167286,
             0.12220850634047802, 0.07250107250107221, 0.008166145780824684, 1.0000000000000118, 0.0016025641025641038,
             0.00644122383252819, 0.0016025641025641038],
            [0.9610949812702193, 0.4109019899274278, 0.3089377623397382, 0.23495840772645324, 0.20677330100735603,
             0.19949331148184576, 0.19889217797273162, 0.1983236356606282, 0.20419249563878353, 0.22439671222315674,
             0.2468001421725052, 0.26934377202851223, 0.3026764978536294, 0.318102154625913, 0.36021755759452945,
             0.3948630408193794, 0.4151320064818989, 0.4874842761804363, 0.5224160769563155, 0.5581950422944579,
             0.5694688385000937, 0.55031751183993, 0.6849362225850419, 0.6075803939330335, 0.7350416126522452,
             0.8427100469155232, 0.6733603675051251, 0.7902933382920369, 0.6836315434546335, 0.8276834938319588,
             0.9487611479434883, 0.9330768568229448, 0.8193885540523317, 0.997, 0.6670224119530418, 0.8010253123998721,
             0.6670224119530418]]
    Func2 = [[351.7858724605074, 86.17819599440456, 40.60817807215555, 22.41370879569776, 13.71370737429577, 9.380805367958237, 6.500771030437166, 5.1934818237317595, 3.8672723604183825, 2.9429262287072286, 2.2293044011542276, 1.7413726101499962, 1.4110020679698105, 1.145285045596285, 0.8493079536648233, 0.7075504964039413, 0.5314726701362551, 0.3946934984396482, 0.35765493303260854, 0.26720947446336063, 0.20483675723548622, 0.16768018253752273, 0.1348248093028533, 0.09131928959311625, 0.06352807714123655, 0.10443083847008712, 0.08486764614717036, 0.07948616458565717, 0.034928848641655796, 0.01957585644371944, 0.0101010101010101, 0.0060868252930227846, 0.003034178389628626, 1.0000000000000095, 0.00040016006402561054, 0.999999999998485, 1.0000000000005178], [0.4014557102478067, 0.19662718856027814, 0.14714271380376454, 0.12376760998544561, 0.11465015678691455, 0.11701257764627322, 0.11971607647950641, 0.12500413021169918, 0.13585072294949355, 0.15396764607614966, 0.1688776087223454, 0.19466317245480091, 0.22196728054180964, 0.25356761298043673, 0.2838683431502283, 0.32214826256665563, 0.3474420903976237, 0.38101744922057085, 0.4472615852893438, 0.4838915064306636, 0.5092442536661715, 0.5393723754562065, 0.6281883708635211, 0.6075021365541242, 0.5864414731409806, 0.8093479024729197, 0.8515064760519562, 0.8942467588311623, 0.7951232441028355, 0.8030838419531001, 0.8347245409015025, 0.7345183562814673, 0.5796857049505159, 0.9992, 0.500100020004001, 0.9997999999999997, 0.9996000000000002]]
    Func3 = [[0, 587.8353916144741, 122.51606563339197, 43.412900839437754, 23.57555460166956, 15.269810446375592,
              10.302086288261853, 6.559749680166944, 4.733027276883256, 3.5899442373476638, 2.6188091438741785,
              1.8973511635960179, 1.59962405851026, 1.1900558241207209, 0.9154742469037501, 0.6663774582713599,
              0.5510310065149937, 0.3962213545720932, 0.34948402621422503, 0.25056075858741617, 0.2104583602324079,
              0.14271205720517338, 0.14355391510677562, 0.09448580768732265, 0.09563164108618655, 0.05255180659992084,
              0.05056730634542649, 0.04830593760483059, 0.017362995116657634, 0.018563316442158327, 0.008064516129032258,
              1.000000000000367, 0.007556675062972293],
            [0, 0.3286716474211523, 0.13242848819742412, 0.08446949128973799, 0.06942310179309806, 0.06710098910557027,
             0.0672674712138261, 0.07277714425838515, 0.07860653317204705, 0.08731142910806171, 0.10727105079265914,
             0.11444337732064816, 0.1388584129446931, 0.1545716580456971, 0.1909675139048617, 0.2078287621915297,
             0.24481225152845482, 0.2809639446229486, 0.3427375708843379, 0.3563349568749672, 0.44658806716684524,
             0.4323139797237823, 0.5209648901237686, 0.5384242118067787, 0.6800151114469211, 0.6244881568587771,
             0.6892348767357834, 0.8342138924420222, 0.7306736811340057, 0.7746555651829621, 0.6684491978609626,
             0.9988000000000005, 0.7158196134574087]]
    Func4 = [[15748.752858254895, 191.911645065407, 73.47842499410935, 30.41633464294201, 19.98626405383976, 11.452154438755791, 7.133577668465118, 5.558028573739012, 3.9221979491511965, 2.883103216560325, 2.134843944573437, 1.672594713096574, 1.2840603701894708, 0.9760043686353028, 0.7383287083076139, 0.5201345636008389, 0.3922546743466271, 0.3075443182363077, 0.2190805647347894, 0.20447301461609166, 0.13718384831088948, 0.10116884193975774, 0.06605172314636748, 0.05724132569770158, 0.05521472392638043, 0.028443817583450853, 0.030927835051546386, 0.020842194810718865, 0.014610389610389598, 0.008522010992889928, 0.999999999998485, 0.004924128228318766, 0.9999999999999153], [0.8339118479799609, 0.09465261480352483, 0.05645445767221664, 0.04042946161369488, 0.039816125266879804, 0.038808660214436394, 0.04108030531513686, 0.04722150007344185, 0.05756881734809522, 0.06174501095337855, 0.07754371601798399, 0.08835218320247393, 0.11277873469258441, 0.14265723594157298, 0.1591593134391955, 0.18294405416467058, 0.20022650693883753, 0.25880068050710453, 0.28508276563948803, 0.38530886060602676, 0.3827846845148159, 0.3989797846053703, 0.4321293982608086, 0.549123155471954, 0.6479481641468685, 0.6177554137838079, 0.7204610951008645, 0.7485830392471395, 0.8589097572148419, 0.7662293265433628, 0.9997999999999997, 0.7786256145580744, 0.9994]]
    Func5 = [[6025.007118984573, 201.67078346958766, 61.164046929888165, 28.570275345356297, 16.42844389234335, 10.118865208041463, 6.5569441993908315, 5.115524806237155, 3.670230880870961, 2.578942628606752, 1.9336145754313054, 1.3128869676600932, 1.0509490349325865, 0.8260606265098674, 0.6192337836269376, 0.39548178329078904, 0.3125342768364706, 0.2535171070520563, 0.17950996366785027, 0.1589593169850413, 0.08776166654828464, 0.07186902085373958, 0.05087852147014008, 0.04031209362808843, 0.015025041736227053, 0.017452299442196745, 0.017452299442196745], [0.4900449307377233, 0.052013394197577424, 0.02763077179222995, 0.022073148401405642, 0.021551468349526978, 0.022874943012867084, 0.025129502524592704, 0.0290677734501479, 0.03586807797479691, 0.04200099407612618, 0.05395970879783885, 0.06222885447396381, 0.08011656818055503, 0.10740529383718123, 0.13349486346877767, 0.14548961984802525, 0.16651317027639345, 0.2157377384182918, 0.2536490552393858, 0.3281846996863911, 0.3205769005384313, 0.37653580728960656, 0.42583822467919963, 0.5652911249293386, 0.5004170141784822, 0.6177302303447393, 0.6177302303447393]]
    Func6 = [[np.mean([4447.600676225763, 4028.6041733343945]), 1122.3591601299543, np.mean([566.3324626056967, 560.7330884421851]), 330.62666503822817, 210.43180030971985, 149.0424349106643, 82.3987852348437, 44.547601156494196, 26.426565843449865, 17.824560051495858, 13.783607579356286, 10.40131640361244, 7.455414078577158, 5.499251804904008, 4.302343361171181, 3.539763892078505, 2.784540963982787, 2.2236920038741386, 1.7559944701404429, 1.4584556538378022, 1.200557790786275, 1.0699270271332681, 0.8634699222550425, 0.7434193327535079, 0.5714709844638692, 0.4396553281521431, 0.3818399186003246, 0.32953426555502685, 0.24102244704033102, 0.22355346426540507, 0.15212505517750333, 0.16001860110466576, 0.10928264810439764], [np.mean([0.2617607830185082, 0.24308742821120072]), 0.0927423079289474, 0.05568821342607846, 0.03793131631871315, 0.0278257161673426, 0.022656568305311583, 0.016435055935377363, 0.013349794613250844, 0.011828502037667653, 0.011907722737912936, 0.011906786143346124, 0.01350556077146206, 0.014414705674904129, 0.015921318603495407, 0.018673975709110842, 0.02270468991422641, 0.02309589946853888, 0.027514993109446845, 0.032185252317963355, 0.039533712217873974, 0.04836171488037051, 0.05571282912235797, 0.0662400500529835, 0.0842953982688298, 0.09463525780280219, 0.11078755407852524, 0.12333175657275465, 0.1523270728384864, 0.1633804430013571, 0.2132018494926845, 0.20858345530936606, 0.26749188764633536, 0.28116189267776115]]
    index = np.linspace(0.5, 11.3, 37)
    print(index)
    index3 = np.linspace(0.1, 9.7, 33)
    index4 = np.linspace(0.2, 9.8, 33)
    index5 = np.linspace(0.2, 7.8, 27)
    index6 = [0.15, 0.25, 0.3, 0.4, 0.45]+[0.5+i*0.2 for i in range(28)]

    print(index6)
    plt.plot(index, Func[1], color='purple')
    plt.plot(index, Func2[1], color='red')
    plt.plot(index3, Func3[1], color='black')
    plt.plot(index4, Func4[1], color='blue')
    plt.plot(index5, Func5[1], color='yellow')
    plt.plot(index6, Func6[1], color='orange')

    plt.show()


def test_clause():
    """Will give a 3SAT problem"""
    V = [i for i in range(4)]
    E = [(0, i) for i in range(1, 4)]
    k = 3
    SAT = graph_to_clause(V, E, k)
    print(SAT)
    sat_to_3sat(SAT, k*len(V))


def test_calendrier(day, month):
    months = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    initial = day
    mo = 1
    while mo < month:
        initial += months[mo-1]
        mo += 1
    while initial < 809:
        print(initial)
        initial += 366

# All tests can be launched here

# test_svm_quantique()
# test_svm()
# test_compar(1.9)
# test_stat()
# test_24()
# test_arith()
# test_QFTn(3)
# test_draw()
# test_gan()
# test_imag(True)
# oracletest(2)
#test_new()
#test_fonction_p()
#test_fonction()
#test_clause()
# test_variational()
