import numpy as np

#To study the generalization of the algorithm, we can use matrix
#We have to be cautious though of not losing track of the real gates used

#Basic gates
H = np.array([[1,1],[1,-1]])*1/(np.sqrt(2))
X = np.array([[0,1],[1,0]])
S = np.diag([1,1j])
I1 = np.eye(2)
Cx = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
Cxinv = np.array([[1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0]])

def grover(n):
    """Returns the matrix of the Grover Operator (no oracle, with each of the intermediate matrix,
    Hn is the n-bits Hadamar, Xn is the n-bits NOT, H_nm1 is the Hadamar on the last qbit and
    Tn is the n-bit Control NOT."""
    H = np.array([[1,1],[1,-1]])*1/(np.sqrt(2))
    X = np.array([[0,1],[1,0]])
    I1 = np.eye(2)
    Tn = np.eye(2**n)
    temp = Tn[-2,].copy()
    Tn[-2,]= Tn[-1,].copy()
    Tn[-1,]= temp
    Hn = H
    Xn = X
    H_nm1 = H
    for _ in range(1,n):
        Hn = np.kron(Hn,H)
        Xn = np.kron(Xn,X)
        H_nm1 = np.kron(I1,H_nm1)
    return [Hn@Xn@H_nm1@Tn@H_nm1@Xn@Hn, Hn, Xn, H_nm1,Tn]


res = grover(4)

#To be able to build the n-bits controle not,
#the only gate not present in qskit for grover,
#we need to study the SQRT of X (NOT)

a = 0.5*(1+1j)
b = 0.5*(1-1j)

#The root of X
RX = np.array([[a,b],[b,a]])

#Its transposed conjugate
RXd = np.matrix(RX).getH()

#Every Unitary matrix can be decomposed into a product
#of three matrix (a rotation around z, a rotation around y and
#a roation around z), with eventually a phase shift

def rz(alpha):
    return np.diag([np.exp(1j*alpha/2),np.exp(-1j*alpha/2)])

def ry(theta):
    return np.array([[np.cos(theta/2),np.sin(theta/2)],[-np.sin(theta/2),np.cos(theta/2)]])

def ph(delta):
    return np.diag([np.exp(1j*delta),np.exp(1j*delta)])

#Here we have the decomposition of RX
delta = np.pi/4

alpha = -np.pi/2

beta = np.pi/2

theta = np.pi/2

#RX = RX1@RX2@RX3@RX4
RX1 = np.diag([np.exp(1j*delta),np.exp(1j*delta)])
RX2 = np.diag([np.exp(1j*alpha/2),np.exp(-1j*alpha/2)])
RX3 = np.array([[np.cos(theta/2),np.sin(theta/2)],[-np.sin(theta/2),np.cos(theta/2)]])
RX4 = np.diag([np.exp(1j*beta/2),np.exp(-1j*beta/2)])

#The final goal is to control RX, which is possible with this, because
#RX = AXBXC, ans ABC=I, so with two CX we can control RX
A = rz(alpha)@ry(theta/2)
B = ry(-theta/2)@rz(-(alpha+beta)/2)
C = rz((beta-alpha)/2)

#And this is the general matrix for the n-th root of X
def rootX(n):
    return np.array([[np.cos(np.pi/(2*n)),-1j*np.sin(np.pi/(2*n))],[-1j*np.sin(np.pi/(2*n)),np.cos(np.pi/(2*n))]])

