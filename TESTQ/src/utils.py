import numpy as np
import pandas as pd


def get_data(sel, kind = "mat_100"):
    """
    ouvre et lit des données (type a présiser, "raw" : données brutes; "mat_100" bag of word sous forme de matrice)
    renvoie X, Y, T (données entrainement, labels enntrainement, données test)
    """
    x = []
    y = []
    t = []
    if kind == "mat_100":
        t.append(pd.read_csv("data/Xte{}_mat100.csv".format(sel), sep=" ", header=None).values)
        x.append(pd.read_csv("data/Xtr{}_mat100.csv".format(sel), sep=" ", header=None).values)
        y.append(pd.read_csv("data/Ytr{}.csv".format(sel)).values)
    if kind == "raw":
        t.append(pd.read_csv("data/Xte{}.csv".format(sel)).values[:,1])
        x.append(pd.read_csv("data/Xtr{}.csv".format(sel)).values[:,1])
        y.append(pd.read_csv("data/Ytr{}.csv".format(sel)).values)
    return np.concatenate(x), np.concatenate(y), np.concatenate(t)


def split(X, Y, prop_test=0.2, prop_valid=0.1):
    """
    découpte les données X et Y en trois jeux, : train, test,  et valid selon les proportion.
    Renvoie X_train, Y_train, X_test, Y_test, X_valid, Y_valid.
    """
    np.random.seed(seed=0)
    n = X.shape[0]
    X_train, X_test, X_valid, Y_train, Y_test, Y_valid = [], [], [], [], [], []
    for i in range(n):
        u = np.random.random()
        if u < prop_valid:
            X_valid.append(X[i])
            Y_valid.append(Y[i])
        elif u < prop_valid+prop_test:
            X_test.append(X[i])
            Y_test.append(Y[i])
        else:
            X_train.append(X[i])
            Y_train.append(Y[i])
    return np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test), np.array(X_valid), np.array(Y_valid)

