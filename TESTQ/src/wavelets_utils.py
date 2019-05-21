import numpy as np


def wave_coefs(n):
    """returns the (n,n) matrix to be applied to a vector of size n to give a
    Daubechies D4 Wavelets Transform"""
    if n % 2:
        print("Size mut be even, please add a value to you signal.")
        return wave_coefs(n+1)
    weight = (4*np.sqrt(2))
    s3 = np.sqrt(3)
    c0 = 1+s3
    c1 = 3+s3
    c2 = 3-s3
    c3 = 1-s3
    k = [c1, c2, c3, c0]
    coefs = np.zeros((n, n))
    for i in range(0, n, 2):
        coefs[i][i] = k[0]
        coefs[i][i + 1] = k[1]
        coefs[i][(i + 2) % n] = k[2]
        coefs[i][(i + 3) % n] = k[3]
        coefs[i + 1][i] = k[3]
        coefs[i + 1][i + 1] = -k[2]
        coefs[i + 1][(i + 2) % n] = k[1]
        coefs[i + 1][(i + 3) % n] = -k[0]

    return coefs/weight


def CZeroP(circuit, reg, target):
    """Adds a C0' gate at qubit target"""
    weight = (4 * np.sqrt(2))
    s3 = np.sqrt(3)
    k2 = (1 - s3)/weight
    #k3 = (1 + s3) / weight
    theta = 2*np.arcsin(2*k2)
    circuit.u3(theta, 0, 0, target)
    circuit.x(target)


def COne(circuit, reg, target):
    """Adds a C1 gate at qubit target"""
    weight = (4 * np.sqrt(2))
    s3 = np.sqrt(3)
    k3 = (1 + s3)/weight
    k0 = (3 + s3)/weight
    #k1 = (3 - s3)/weight
    #k2 = (1 - s3)/weight
    theta = 2*np.arccos(0.5*k0/k3)
    circuit.u3(theta, 0, np.pi, target)


def Qn(circuit, reg, target):
    """Add qn gate on all target (list)"""
    n = len(target)
    lamb = [2 * np.pi / (2 ** m) for m in range(2, n + 1)]
    target.reverse()
    for i in range(n):
        circuit.h(target[i])
        for j in range(len(lamb) - i):
            circuit.cu1(lamb[j], target[1 + j + i], target[i])
    lamb2 = [np.pi]+lamb
    for i in range(n):
        circuit.u1(lamb2[-i-1], target[i])
    for i in range(n-1, -1, -1):
        for j in range(len(lamb) - i-1, -1, -1):
            circuit.cu1(-lamb[j], target[1 + j + i], target[i])
        circuit.h(target[i])
