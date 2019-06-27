from math import asin, sqrt, sin
from random import randint, sample
from itertools import product
from matplotlib import pyplot as plt
from scipy.stats import nbinom, norm
import numpy as np
from scipy.special import binom


def cata(n):
    return binom(2*n, n)/(n+1)


def u_i(a, n, i):
    if i == 1:
        return 1 - 2*a/n

    v = u_i(a, n, i-1)
    print(v)
    return 2*v**2 - 1


# u_i(5, 8, 10)


def p_failure_conditionned(a, n):
    total = 2**n
    theta_a = asin(sqrt(a/total))

    v = 1
    for i in range(1, n + 1):
        print(v)
        v *= sin((2**(n-i+1) + 1)*theta_a)

    return 1 - v


# print(p_failure_conditionned(8, 5))


def create_3sat(nb_clauses, nb_variables):
    instance = []
    for _ in range(nb_clauses):
        instance.append(sample(range(1, nb_variables + 1), 3))
        instance[-1].sort()
        for i in range(3):
            instance[-1][i] *= 2*randint(0, 1) - 1

    return instance


def find_number_of_sols(instance, nb_variables, all_sols, acc):
    nv_nb_var = nb_variables
    coeff = 1
    if acc:
        instance, nv_nb_var, coeff = remove_unused_var(instance, nb_variables)
    t = ["".join(seq)[::-1] for seq in product("01", repeat=nv_nb_var)]

    nb_sols = 0
    for val in t:
        nb_sols += evaluate_instance(instance, val)

    all_sols[nb_sols * coeff] += 1

    return nb_sols * coeff


def remove_unused_var(instance, nb_variables):

    # print("Ratio : ", len(instance)/nb_variables)

    unused_variables = [i + 1 for i in range(nb_variables)]
    for clause in instance:
        for var in clause:
            if abs(var) in unused_variables:
                unused_variables.remove(abs(var))

    for index, i in enumerate(unused_variables):
        for clause in instance:
            for ind, var in enumerate(clause):
                if abs(var) > i - index:
                    clause[ind] -= int(var / abs(var))

    nb_variables_utiles = nb_variables - len(unused_variables)
    return instance, nb_variables_utiles, 2**(nb_variables - nb_variables_utiles)


def evaluate_instance(instance, val):

    for clause in instance:
        total = 0

        for var in clause:
            if var < 0:
                if val[-var - 1] == '0':
                    total += 1
                    continue
            else:
                if val[var - 1] == '1':
                    total += 1
                    continue

        if total == 0:
            return False
    return True


def proba(nb_variables=15):
    ratio = 2.5
    iterations = 25
    nb_instances = 400

    ratios = []
    means = []

    for i in range(iterations):
        print(i)
        ratio += 0.15
        nb_clauses = int(ratio * nb_variables)
        satisfiable = 0
        for _ in range(nb_instances):
            instance = create_3sat(nb_clauses, nb_variables)
            if find_number_of_sols(instance, nb_variables) > 0:
                satisfiable += 1
        mean = satisfiable/nb_instances
        ratios.append(ratio)
        means.append(mean)

    return ratios, means


def curve():
    for i in range(5, 10):
        ratios, means = proba(i)
        plt.plot(ratios, means, label=i)
    plt.legend()
    plt.show()


def variance(result, index):
    prec = 10000000
    result_sam = result * prec
    tab = []
    for i,res in enumerate(result_sam):
        for _ in range(int(res)):
            tab.append(index[i])
    #print(np.mean(tab))
    return np.var(tab)

def moyenne(result, index):
    prec = 10000000
    result_sam = result * prec
    tab = []
    for i, res in enumerate(result_sam):
        for _ in range(int(res)):
            tab.append(index[i])
    #print(np.mean(tab))
    return np.mean(tab)


def distribution(nb_variables, nb_iterations=100, ratio=5):
    print("Nb var:", nb_variables, "Ratio:", ratio)
    nb_clauses = int(nb_variables * ratio)
    nb_sols = [0 for _ in range(2**nb_variables)]
    instances = [create_3sat(nb_clauses, nb_variables) for _ in range(nb_iterations)]
    for instance in instances:
        find_number_of_sols(instance, nb_variables, nb_sols, True)

    mini = 0
    maxi = 2**nb_variables - 1

    results = [nb_sols[v]/nb_iterations for v in range(mini, maxi + 1)]
    for i, res_t_p in enumerate(results):
        if res_t_p != 0:
            print(i)
    index = [i for i in range(mini, maxi + 1)]
    mean = moyenne(np.array(results), np.array(index))
    var = variance(np.array(results), np.array(index))
    print("mean : ", mean, "var : ", var)
    if mean == 0 or var == 0:
        return index, [0 for _ in range(len(index))], 0, 0
    p_est = (var/mean)
    n_est = abs((mean**2)/(var-mean))
    p_est_alt = mean/var
    n_est_alt = var*p_est_alt**2/(1-p_est_alt)
    print("estimation of p : ", p_est, "estimation of n : ", n_est, p_est**n_est,results[0])
    print("estimation of p : ", p_est_alt, "estimation of n : ", n_est_alt, p_est_alt**n_est_alt,results[0])

    #print(sum(results))
    index = np.array(index)
    if p_est < 1:
        lab = str(nb_variables) + ", " + str(ratio)
        nbi_est = nbinom.pmf(index, n_est, p_est)# * norm.pdf(index, mean, var)
        #print("Sum : ", np.sum(nbi_est))
        mean = moyenne(np.array(nbi_est), index)
        var = variance(np.array(nbi_est), index)
        print("mean_est : ", mean, "var_est : ", var, n_est*(1-p_est)/p_est, n_est*(1-p_est)/(p_est**2), nbinom.stats(n_est, p_est))
        plt.plot(index, nbi_est, ls="dashed", label=lab)
    else:
        # lab = str(nb_variables) + ", " + str(ratio)
        nbi_est = nbinom.pmf(index, n_est_alt, p_est_alt)  # * norm.pdf(index, mean, var)
        # print("Sum : ", np.sum(nbi_est))
        mean = moyenne(np.array(nbi_est), index)
        var = variance(np.array(nbi_est), index)
        print("mean_est : ", mean, "var_est : ", var, n_est_alt * (1 - p_est_alt) / p_est_alt, n_est_alt * (1 - p_est_alt) / (p_est_alt ** 2),
              nbinom.stats(n_est_alt, p_est_alt))
        plt.plot(index, nbi_est, ls="dashed")
        n_est = n_est_alt
        p_est = p_est_alt
    print("\n")
    return index, results, n_est, p_est


def distrib_plot(nb_variables_min, nb_variables_max, ratio_min, ratio_max, step, nb_iterations=1500):

    N = []
    P = []
    for nb_variables in range(nb_variables_min, nb_variables_max + 1):
        print("nb_variables ", nb_variables)
        ratio = ratio_min
        N.append([])
        P.append([])
        while ratio < ratio_max:

            index, results, n_e, p_e = distribution(nb_variables=nb_variables, nb_iterations=nb_iterations, ratio=ratio)
            #lab = str(nb_variables) + ", " + str(ratio)
            plt.plot(index, results)
            ratio += step
            N[-1].append(n_e)
            P[-1].append(p_e)

    #plt.legend()
    plt.show()
    X = [i for i in range(len(N[0]))]
    for i in range(len(N)):
        plt.plot(X, N[i])
        plt.plot(X, P[i], ls="dashed")
    print(N, P)
    plt.show()


def negat_binom():
    n, p = 16, 0.5
    fig, ax = plt.subplots(1, 1)
    x = np.arange(0, 16)
    ax.plot(x, nbinom.pmf(x, n, p), ms=8, label='nbinom pmf')
    plt.show()


# inst = create_3sat(12, 4)
# print(inst)
# print(find_number_of_sols(instance=inst, nb_variables=4))
# proba()
# curve()
# distribution(6, nb_iterations=5000)
distrib_plot(16, 16, 0.1, 0.5, 0.05, nb_iterations=5000)
# negat_binom()
# [[64.43300175640161, 39.87710572302016, 21.479148327836953, 13.224678074918591, 10.114095025032283, 7.305964354826329, 4.779006314185672, 3.9003414963141245, 2.9728511206282526, 2.255847723258879, 1.8402515956550494, 1.4626559953319762, 1.0939152525346068, 0.8845963390333511, 0.8127131644061196, 0.5757293483606997, 0.4368193241146636, 0.4409229424690633, 0.33088294692624637, 0.2663705853343308, 0.18539326302172154]] [[0.2945050883023091, 0.2531559775888849, 0.214240139860882, 0.19986190619045985, 0.1993510610365876, 0.21197491330634824, 0.209940650362044, 0.21606216135314324, 0.24363935367170633, 0.26472587716540835, 0.277219320349285, 0.3133330013167413, 0.33676388142469943, 0.38021049212210195, 0.42681998497112733, 0.4441912754223663, 0.4712084341195817, 0.5453437612080309, 0.5637971739666657, 0.6018714170375933, 0.5896874979386468]]
# [[7729.018255678793, 140.153834155207, 68.77595919655846, 31.62018118184545, 18.030431610812485, 11.480451328936848, 8.854799040173322, 5.891748736768329, 4.107058029460621, 3.525987646397012, 2.6501857762543453, 1.9939336429398156, 1.796115967192535, 1.3439730213174272, 1.0573728322694307, 0.9370165183504918, 0.6862225806758537, 0.58629480789044, 0.46467717773394074, 0.4351295050299971, 0.31030829231196316, 0.18283441858118177, 0.2508750473787763, 0.12603102215466033, 0.1403733845624147, 0.208944572364959, 0.05937056209629393, 0.06406561737973851, 0.02549828229037716, 0.044190126138167286, 0.12220850634047802, 0.07250107250107221, 0.008166145780824684, 1.0000000000000118, 0.0016025641025641038, 0.00644122383252819, 0.0016025641025641038]] [[0.9610949812702193, 0.4109019899274278, 0.3089377623397382, 0.23495840772645324, 0.20677330100735603, 0.19949331148184576, 0.19889217797273162, 0.1983236356606282, 0.20419249563878353, 0.22439671222315674, 0.2468001421725052, 0.26934377202851223, 0.3026764978536294, 0.318102154625913, 0.36021755759452945, 0.3948630408193794, 0.4151320064818989, 0.4874842761804363, 0.5224160769563155, 0.5581950422944579, 0.5694688385000937, 0.55031751183993, 0.6849362225850419, 0.6075803939330335, 0.7350416126522452, 0.8427100469155232, 0.6733603675051251, 0.7902933382920369, 0.6836315434546335, 0.8276834938319588, 0.9487611479434883, 0.9330768568229448, 0.8193885540523317, 0.997, 0.6670224119530418, 0.8010253123998721, 0.6670224119530418]]