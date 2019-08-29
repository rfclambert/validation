from General import *
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

from qiskit.aqua.components.optimizers import SPSA
from qiskit.aqua.components.uncertainty_models import NormalDistribution
from scipy.stats import beta, nbinom


def sum_l(L):
    """sum of what inside a list, even if not integers"""
    res = []
    for l in L:
        res += l
    return res


def norm1(f, g):
    """norm 1 of two lists of points of th esame size"""
    res = 0
    for i in range(len(f)):
        res += np.abs(f[i]-g[i])
    return res


def cross_entropy(q, p):
    """technically the kl divergence"""
    assert(len(q) == len(p))
    res = 0
    for i in range(len(q)):
        if q[i] <= 10**-5:
            q[i] = 10**-5
        if p[i] <= 10**-5:
            p[i] = 10**-5
        res += p[i]*np.log(p[i]/q[i])
    return res


def Variationer_learn_gan(shots, l, m, proba=None, n=4, distri_size=0, easy=False, prior=False):
    """Learn the theta to match the distribution.
    Shots is the amount optimization steps.
    l is the depth of the variationner.
    m is the size of each batch
    proba is the distribution given. must sum to 1
    n is the number of qubits, and defines the numerical precision. 2**n must be the size of proba
    distri_size is the size of each distribution if used in distribution comparison mode
    easy is a boolean to choose if you want to compare distribution using L1 norm, instead of a
    neural network.
    Example:
    Variationer_learn_gan(1000, 1, 4096, proba=[1/24]*24+8*[0], n=5, distri_size=0, easy=True)"""

    if easy:
        print("Distribution learning in easy mode. It won't use a Neural Network to learn the distribution,"
              "the error function is the L1 distance to the provided distribution.")
        if distri_size > 0:
            print("Easy learning overwrites distribution learning.")
    elif distri_size > 0:
        print("Distribution learning in distribution batch mode."
              " Each optimization step will draw {}".format(int(m/distri_size)) + "batch of size distri_size"
              "and directly compare, using a neural network, to batches from real distribution.")
    else:
        print("Distribution learning in singleton batch mode."
              "Each optimization step will draw m sample from the circuit and give it to the neural network,"
              "with m samples from the provided distribution.")

    N = 10 ** 5  # Taille de la database

    if proba is None:
        # By default, we learn the normal 0 1 distribution.
        Database = np.random.normal(0, 1, N)
        mini = np.min(Database)
        maxi = np.max(Database)
        h = (maxi - mini) / (2 ** n)
        bins = [[k for d in Database if mini + h * k < d < mini + h * (k + 1)] for k in range(2 ** n)]
    else:
        assert (2 ** n == len(proba))
        assert (0.98 < sum(proba) < 1.01)
        Database = sum_l([[k] * int(proba[k] * N) for k in range(len(proba))])
        mini = 0
        maxi = (2 ** n) - 1
        h = (maxi - mini) / (2 ** n)
        bins = [[0] * int(p * N) for p in proba]

    interv = [mini + h * k for k in range(2 ** n)]

    # The adversary neural network to be trained
    clf = MLPClassifier(hidden_layer_sizes=(4, 8, 16, 8, 4, 2))

    # Initial random theta
    Theta = np.random.normal(0, 10**-2, 2*n*(l+1))
    print("Number of parameters to be fitted: {}".format(2*n*(l+1)))

    # Initial circuit
    q = QuantumRegister(n, 'q')
    RegX = [q[i] for i in range(n)]

    # To keep track of what's done
    curve = []
    # parameters
    low = 0
    high = 10

    # initialize distribution
    mu = 5
    sigma = 1
    normal = NormalDistribution(n, mu, sigma, low, high)

    # create circuit for distribution
    q_normal = QuantumRegister(n)
    qc_normal = QuantumCircuit(q)
    normal.build(qc_normal, q_normal)

    def Remp(theta):
        """The error function"""
        # The circuit that is trained
        circ = QuantumCircuit(q)
        W(circ, q, RegX, theta)
        circ_var = circ
        if prior:
            circ_var = qc_normal+circ_var
        circ_m = measure_direct(circ_var, q, RegX)
        counts = launch(m, circ_m)

        # We transform the results
        input_var = []
        for KEY in counts:
            value = int('0b' + KEY, 2)
            for _ in range(counts[KEY]):
                input_var.append(value)
        r.shuffle(Database)
        r.shuffle(input_var)

        # Singleton learning (the original)
        if distri_size == 0:
            input_db = Database[:m]
            input_tot = input_var + input_db
            target = [0]*m + [1]*m
            index = [i for i in range(2*m)]
            r.shuffle(index)
            input_tot = [[input_tot[i]] for i in index]
            target = [target[i] for i in index]
            clf.partial_fit(input_tot, target, [0, 1])
            prob = np.sum(-np.log(clf.predict_proba([[inp] for inp in input_var])), axis=0)
            err_theta = prob[1] / m

        # Distribution learning
        else:
            assert(distri_size <= m)
            n_input = int(m/distri_size)
            input_db = sorted([Database[i*distri_size:(i+1)*distri_size] for i in range(n_input)])
            input_var = sorted([input_var[i*distri_size:(i+1)*distri_size] for i in range(n_input)])
            input_tot = input_var+input_db
            target = [0] * n_input + [1] * n_input
            index = [i for i in range(2 * n_input)]
            r.shuffle(index)
            input_tot = [input_tot[i] for i in index]
            target = [target[i] for i in index]
            clf.partial_fit(input_tot, target, [0, 1])
            prob = np.sum(-np.log(clf.predict_proba(input_var)), axis=0)
            err_theta = prob[1]/n_input

        curve.append(err_theta)
        if len(curve) % 100 == 0:
            print(err_theta, len(curve))
        return err_theta

    def Remp_easy(theta):
        """The error function, when using only L1 norm"""
        # The circuit that is trained
        circ = QuantumCircuit(q)
        W(circ, q, RegX, theta)
        circ_var = circ
        if prior:
            circ_var = qc_normal+circ_var
        circ_m = measure_direct(circ_var, q, RegX)
        counts = launch(m, circ_m)

        # We transform the results
        bins_var = [0 for _ in range(2 ** n)]
        for KEY in counts:
            value = int('0b' + KEY, 2)
            bins_var[value] = counts[KEY] / m

        # L1 distance/cross-entropy calculation
        compar = [len(b) / N for b in bins]
        err_theta = cross_entropy(bins_var, compar)

        curve.append(err_theta)
        if len(curve) % 100 == 0:
            print(err_theta, len(curve))
        return err_theta

    # The optimizer
    optimizer = SPSA(max_trials=int(shots/2), c0=4.0, c1=0.1, c2=0.602,c3=0.101,c4=0.0, skip_calibration=True)
    optimizer.set_options(save_steps=1)

    if easy:
        theta_star = optimizer.optimize(2*n*(l+1), Remp_easy, initial_point=Theta)
    else:
        theta_star = optimizer.optimize(2 * n * (l + 1), Remp, initial_point=Theta)

    plt.plot([i for i in range(len(curve))], curve)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Loss function")
    plt.show()

    print("Learning done! \nDebut du test...")

    # We test the trained model
    circ_test = QuantumCircuit(q)
    W(circ_test, q, RegX, theta_star[0])
    circ_m_test = measure_direct(circ_test, q, RegX)
    prec_here = 4096
    counts_test = launch(prec_here, circ_m_test)

    # We transform the results
    bins_var_test = [0 for _ in range(2**n)]
    for KEY_test in counts_test:
        value_test = int('0b' + KEY_test, 2)
        bins_var_test[value_test] = counts_test[KEY_test]/prec_here
    compar_test = [len(b)/N for b in bins]

    if len(interv) == len(compar_test):
        plt.plot(interv, compar_test)
    plt.plot(interv, bins_var_test)
    plt.xlabel("x")
    plt.ylabel("density")
    plt.title("PDF loaded in the Quantum Channel (orange)")
    plt.show()
    err_theta_test = norm1(compar_test, bins_var_test)
    print("Error L1: ", err_theta_test)

    return theta_star[0], norm1(compar_test, bins_var_test)


def Variational_prepared(theta, n):
    """Variational_prepared(theta_inpu, 4)"""
    N = 10**5
    q = QuantumRegister(n, 'q')
    circ_test = QuantumCircuit(q)
    RegX = [q[i] for i in range(n)]
    # We use the given theta. It's a way to test models after they have been stored
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
    """To test a lot of parameters for model selection"""
    res = []
    for shots in [500, 1000]:
        for l in range(1, 6):
            for m in [16, 32, 64, 128, 256, 512, 1024]:
                res.append(Variationer_learn_gan(shots, l, m))
                print(res[-1])
    print(res)


def constants():
    """The theta of some good models. To test them, use Variational_prepared(theta, n)
    for the theta of your choice."""
    t_1_32 = [3.31864808,  2.70163485,  0.17376359, -2.09200851, -5.64278358,
              -0.02805571, -1.69355561, -1.80010967, -0.90994795,  1.04365892,
              -1.66310046,  4.20790657, -1.59076603,  1.0952253, -1.65380373,
              -0.04839213]

    t_1_256 = [1.4836451,  0.61164992, -0.26072754, -0.00992019, -3.00359695,
               0.3096171, -1.32958612, -1.55170692,  1.53032967,  1.30949908,
               2.90979934,  0.42641291,  1.38031126, -1.6912363, -2.22642753,
               -0.63554297]

    t_2_16 = [-0.54727354,  0.57433066, -5.08689108,  1.6563377, -2.05346556,
              -1.74382868,  1.95610819, -0.13959262,  1.22253357, -6.83772177,
              -2.02789211, -1.89190738,  1.40594423,  2.89055446,  1.49767865,
              -1.90666786, -0.52193196,  1.74087753,  0.98799308, -7.2175616,
              -2.2153014, -0.89120928, -1.0228835, -0.32941139]

    t_2_64 = [-0.85748291,  1.11221879,  3.33988323,  2.89990863, -0.07923698,
              1.07160315, -3.52751917, -0.62689393, -4.23035176, -1.87440595,
              -0.25602441, -3.23864092,  3.20829116, -3.15028101,  1.3024739,
              2.4723658,  1.95011207, -2.26265338,  2.8332384, -1.83818937,
              1.58268919,  4.18310784, -0.28816408,  1.29977144]

    t_2_128 = [-1.43015547,  1.02366801, -0.09334577,  0.2010502,  0.08024445,
               0.9377122,  1.74697802,  1.81112195, -0.47388993,  1.80337122,
               1.9189899, -1.5312677, -1.57508169,  3.08854922,  0.39991324,
               -1.79535668, -1.11659818, -2.86820592,  0.33618413, -0.72414528,
               -1.65401045,  0.95549606, -0.19515165,  3.14973983]

    t_2_1024 = [-2.51086614, -2.62059864,  3.65781682, -0.98812452,  2.69424146,
                1.15062414,  2.00499229, -3.27208132,  2.32961203, -2.49879457,
                -0.58160747, -3.32934437, -3.51547155, -1.81037178,  2.4925058,
                -1.8028326, -3.63278992,  3.32062055, -2.48923665, -4.77655673,
                -1.85874307,  0.78029357, -3.85954062,  1.54315157]

    t_3_128 = [-0.16353129,  0.30213317, -0.57802369,  0.32674041, -2.07815837,
               -0.48253043,  0.72770648, -0.10635877,  0.15449772,  1.23119864,
               -0.83217582,  0.3067664,  0.20340706, -0.12048108,  1.20854069,
               -3.36809841, -1.43859462,  0.37496259, -0.95597734,  0.1222873,
               1.30951163, -1.39630901,  0.19073165, -1.41750384, -0.826935,
               -0.22753217, -0.02185718, -0.3672339, -1.78464805, -0.88456567,
               -0.03227605, -1.55262202]

    t_5_64 = [2.11592348, -0.4183771, -0.88343005, -0.04419628,  0.89521532,
              0.1025278, -1.64100914,  1.689986, -1.12291366, -0.07800934,
              -0.09101488,  0.27321852,  0.47481255,  1.12146641,  1.1434541,
              -0.07711449,  2.43015488,  3.45196626,  1.28896883,  0.99762344,
              0.16134645, -0.37324293, -0.13385824,  1.77017139,  0.39123187,
              1.30140444, -0.03931637,  0.3242943,  0.20750545, -1.91669459,
              1.45649439,  0.12820491,  2.46844238,  0.25753582, -1.91430595,
              -0.71813909, -0.25858911, -1.01574223,  1.22409724,  1.80389738,
              0.3796137, -1.43587756,  0.86979539,  0.55311133,  1.46602903,
              -0.6184541,  0.16036584, -0.11123353]

    t_5_128 = [1.08989702e+00, -1.49051697e+00,  8.14486974e-01,  7.92240415e-02,
               1.72197113e-04,  3.48801508e+00,  1.14651122e-01,  7.70220780e-02,
               -1.68742668e+00,  3.00499651e+00, -4.69534163e-01,  1.72281306e+00,
               1.45666856e+00, -4.15322493e-01, -9.53684888e-03, -1.68472224e+00,
               2.06987894e+00,  3.02494019e-02, -2.61707012e+00, -1.19113974e+00,
               -8.89763224e-01,  2.50143514e-01,  4.24896092e-01, -2.83741961e+00,
               8.79821590e-01, -1.67107892e-01,  6.64131817e-02, -2.04659989e+00,
               7.47541286e-02, -1.02310543e+00,  3.13924387e+00, -3.51509956e+00,
               2.50874300e-01,  4.87987278e-01,  1.27683214e+00,  3.09043521e-01,
               -2.76703038e-02,  2.79941447e-01, -1.38439304e+00,  3.84249639e-01,
               -7.85157759e-01, -8.11820162e-01, -1.69661489e-01,  1.64321699e+00,
               -2.56474508e+00, -1.40091359e+00, -2.55790355e+00, -1.66886329e+00]

    t_5_512 = [-1.77064036,  0.07006809, -0.2256065,  0.3727588,  0.20034255,
               -1.58783425,  1.97756253,  0.0463704, -2.51186054,  3.03336371,
               -2.13948468, -2.22479575, -0.33007197, -4.12867618, -1.85697507,
               0.48595115, -1.5498664, -4.21647648,  2.04479253, -0.69110098,
               3.44946553, -0.37834516, -0.89584749,  0.77448499,  0.78626296,
               2.67985402,  2.96641449,  0.43725711, -0.05472007, -3.39525164,
               3.85456382,  0.19171059, -1.8037696, -1.94612507, -1.68660112,
               6.46625259,  0.61241911,  2.50767962, -0.35229409,  2.32265905,
               -1.37065472, -0.61143652,  2.17962736, -2.68667727,  1.30171312,
               -4.10315302,  3.05412034,  1.33134782]

    t_1_1024_alt = [0.78943029,  3.72717697, -0.03977744,  3.03962715,  0.11358853,
                    1.82752726, -1.85536213, -1.71534693, -0.90914775,  1.55746411,
                    1.39099534,  2.08814849, -1.73172696, -0.85733386, -3.05651896,
                    4.97588336]

    t_2_64_alt = [0.176965, -1.43126866, -1.11105571, -1.16346468,  0.14418196,
                  -1.48919534,  0.35999701,  1.37074422,  1.34800631, -1.93305451,
                  0.21550213, -1.95229256,  0.04903009, -0.96279237,  1.72939789,
                  0.55809168, -0.86191505,  1.03934622,  1.16567844, -1.01335362,
                  1.56165517, -0.46908402,  0.35119359, -1.58643566]

    t_3_128_alt = [-0.59679254, -0.04733473, -1.09783163, -0.44303939,  2.04188429,
                   -0.1708647, -0.41036841,  1.26836572, -1.66145149, -0.72814127,
                   -0.2481015,  0.14673307, -0.38109231, -0.71773962,  0.79885609,
                   -1.1115943, -2.24668751,  0.93335477, -1.52470429,  0.04192152,
                   -0.62900467,  0.35050511, -1.38900529,  2.42017759, -1.19878385,
                   -2.24033744, -0.6647399, -0.82208469, -1.10141209,  1.6091096,
                   0.48592227,  1.06083672]

    t_3_1024_alt = [-5.7910756,   4.84194105,   5.75501927,  26.29183391,
                    16.97341335,   2.40644695,  15.71535522,  -9.21635434,
                    8.80833214,  11.07212514,   9.74792251,   8.19639719,
                    0.63020537,  -9.17585195,  -1.61840255,  11.21261345,
                    -20.33110023,   5.18022469, -10.76741451,   3.31542823,
                    -1.54483389,  11.23199795,   3.2421057,  14.15190138,
                    -7.8001903,  -1.99138789,  -6.76047747,  -5.35990669,
                    5.27875273,   7.17360254,   3.55301609,   1.68752982]

    t_4_1024_alt = [0.62052425,  1.48430751, -0.3065588,  0.99694099,  0.25938114,
                    -1.51068502,  1.00519182, -0.34252084,  0.61231532,  0.75828028,
                    -0.57085469, -0.68402363, -1.95724342,  0.24454005, -1.78344647,
                    -0.51723547,  1.38912609, -2.01828835, -1.74786356, -0.28671573,
                    -0.9417902, -0.319721, -0.32485401,  1.41327651, -0.02982839,
                    -0.531383, -0.29914802, -0.50821502,  0.43570611, -0.51779266,
                    1.36993631,  0.87753105,  0.28127834,  1.64252461,  0.05784274,
                    -1.06500583,  1.26083344, -0.47440593, -0.26470805, -0.20345767]

    t_5_1024_alt = [-5.47090021,  14.1117991,   3.84301668,  11.0155575,
                    6.04007004,  -1.73874385,   9.62980885,   2.84995437,
                    5.96455344,  -3.8221146, -12.23978494,  15.39351006,
                    -2.97482218, -14.37463493, -17.15366823,   4.93120997,
                    30.09493954,  35.7585838,  -4.59535679,   3.55670253,
                    -16.02314829,  10.89593126, -10.50525443,  -7.80923094,
                    2.54359644, -13.37889343,   3.41189793,  -2.6354193,
                    8.3279459, -4.2641389,  -1.92107563,  12.17582338,
                    30.2212493,  -9.07009046, -11.47063616,   7.20135522,
                    15.37658419, -12.66725036, -15.5572806,  11.17455258,
                    -2.75836741,  15.03734549, -13.77320335,   2.35618223,
                    19.65803084,  -0.48621641,   7.68344555,   0.39935987]

    normal = [t_1_32, t_1_256, t_1_1024_alt, t_2_16, t_2_64, t_2_64_alt, t_2_128, t_2_1024, t_3_128, t_3_128_alt,
              t_3_1024_alt, t_4_1024_alt, t_5_64, t_5_128, t_5_512, t_5_1024_alt]

    #cards

    t_1_4096_cartes = np.array([13.86993424,  5.04909209, -1.79695972, -0.84285724,  1.08545796,
                                2.97134661,  6.46717987,  3.13638555,  5.088643, -5.46823185,
                               -1.68518226,  4.67255675, -1.73082473,  0.69172197, -2.14461026,
                               -1.4988397,  1.46973947, -4.73567944, -1.89753097,  1.24460023])

    return normal, t_1_4096_cartes


def beta_proba(nbr_qubits, a, b):
    """returns 2**nbr_qubits values of beta(a, b)"""
    n = 2**nbr_qubits
    arr = np.linspace(0, 1, n)
    res = beta.pdf(arr, a, b)
    plt.plot(arr, res)
    plt.show()
    res *= 1/sum(res)
    return res


def nbinom_proba(nbr_qubits, a):
    """returns 2**nbr_qubits values of nbinom(a)"""
    n = 2**nbr_qubits
    arr = np.linspace(0, n, n+1)
    print(arr)
    for b in np.linspace(0.1, 0.9, 6):
        print(b)
        res = nbinom.pmf(arr, a, b)
        print(res)
        plt.plot(arr, res)
    plt.show()
    b = 0.5
    for a in range(1, 20, 5):
        for b in np.linspace(0.1, 0.9, 4):
            print(a, b)
            res = nbinom.pmf(arr, a, b)
            print(np.sum(res*arr), a, b, a*(1-b)/b, a*b/(1-b))
            print(res)
            plt.plot(arr, res)
    plt.show()
    res *= 1/sum(res)
    return res
