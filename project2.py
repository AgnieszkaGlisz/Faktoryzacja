import numpy as np
import math as m
import time as tm
import matplotlib.pyplot as plt


def createMatrix(N,a1,a2,a3):
    A = np.zeros((N,N))
    np.fill_diagonal(A,a1)

    for i in range(N - 1):
        A[i + 1][i] = A[i][i + 1] = a2
        if i < N - 2:
            A[i + 2][i] = A[i][i + 2] = a3
    #print(A)
    return A

def vectorWhetting(N,f):
   b = np.zeros((N,1))
   for n in range(N):
       b[n, 0] = m.sin(n * (f + 1))
   #print(b)
   return b

   print("Time: ", end - start)

def jackobi(A,b,N,res):
    start = tm.time()
    normRes = 1
    iterations = 0
    x = np.zeros(N)
    
    D = -(A - np.tril(A) - np.triu(A))
    Dinv = np.linalg.inv(D)
    LU = A - D;
    
    while normRes > res:
        iterations+=1
        x= -np.dot(np.dot(Dinv,LU),x) + np.dot(Dinv,b)

        normRes = np.linalg.norm(np.subtract(np.dot(A,x),b))
   
    end = tm.time()
    print("\tJackobi. Size of A matix: ", N)
    print("\tNumber of iterations: ", iterations)
    print("\tTime: ", end - start)
    return end-start

def gaussSeidl(A,b,N,res):
    start = tm.time()
    normRes = 1
    iterations = 0
    x=np.ones(N)
   
    D_Linv = np.linalg.inv(np.tril(A, k=0))
    U = np.triu(A,k=1)

    while normRes > res:
        iterations+=1
        x= -np.dot(D_Linv,np.dot(U,x)) + np.dot(D_Linv,b)

        normRes = np.linalg.norm(np.subtract(np.dot(A,x),b))

    end = tm.time()
    print("\tGauss. Size of A matix: ", N)
    print("\tNumber of iterations: ", iterations)
    print("\tTime: ", end - start)
    return end-start

def factorisation(A,b,N):
    start = tm.time()
    L = np.zeros((N, N))
    U = A.copy()

    for i in range(0, N):
        L[i, i] = 1
 
    for i in range(0, N-1):
        for j in range(i+1, N):
            L[j, i] = U[j, i]/U[i, i]
            U[j, i:N] = U[j, i:N] - L[j, i] * U[i, i:N]

    Y = np.zeros(N)
    for i in range(0, N):
        Y[i] = b[i]
        for k in range(0, i):
          Y[i] -= L[i, k]*Y[k]
 
    X = np.ones((N, 1)) 
    for i in range(N-1, -1, -1):
        X[i, 0] = Y[i]
        for k in range(i+1, N):
            X[i,0] -= U[i, k]*X[k, 0]
        X[i, 0] /= U[i, i]

    end = tm.time()
    normRes = np.linalg.norm(np.subtract(np.dot(A,X),b))
    print("\tLU factorisation. Size of A matix: ", N)
    print("\tResiduum: ", normRes)
    print("\tTime: ", end - start)
    return end-start


N = 964
a1 = 5 + 5

#Zadanie A
A = createMatrix(N,a1,-1,-1)
b = vectorWhetting(N,1)

#Zadanie B
jackobi(A,b,N, m.pow(10,-9))
#print()
#gaussSeidl(A,b,N,m.pow(10,-9))

#Zadanie C
#a1 = 10
#A = createMatrix(N, a1, -1, -1)
#print("\tTask C: \n")
#jackobi(A,b,N, m.pow(10,-9))
#print()
#gaussSeidl(A,b,N,m.pow(10,-9))

#Zadanie D
#factorisation(A,b,N)

#Zadanie E
#Ne = [100, 500, 1000, 2000, 3000]
#times = np.zeros((3,5))

#for i in range(5):
#    b = vectorWhetting(Ne[i],1)
#    A = createMatrix(Ne[i],a1,-1,-1)

#    times[0,i]= jackobi(A,b,Ne[i], m.pow(10,-9))
#    times[1,i]= gaussSeidl(A,b,Ne[i],m.pow(10,-9))
#    times[2,i]= factorisation(A,b,Ne[i])

#plt.plot(Ne, times[1], 'm', Ne, times[0], 'y', Ne, times[2], 'c')
#plt.plot(Ne, times[0], 'y')
#plt.xlabel("Matrix sizes")
#plt.ylabel("Time [seconds]")
#plt.show()
#2+2