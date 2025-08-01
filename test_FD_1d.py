import torch
import numpy as np


def u_exact(x):
    return torch.sin(np.pi * x)

def f(x):
    return np.pi ** 2 * torch.sin(np.pi * x)

def matrix_PE(N,f):
    A = torch.zeros(N+1,N+1)
    F = torch.zeros(N+1)
    h = 1.0/N
    x = torch.linspace(0,1,N+1)
    for i in range(N+1):
        if i > 0 and  i < N:
            A[i,i-1] = -1
            A[i,i] = 2
            A[i,i+1] = -1
            F[i] = h ** 2 * f(x[i])
    A[0, 0] = 1        
    A[N, N] = 1
    F[0] = 0
    F[N] = 0
    return A, F

def solve_PE(N,f):
    A, F = matrix_PE(N,f)
    u_sol = torch.linalg.solve(A,F)
    return u_sol


N = 3
x = torch.linspace(0,1,N+1)
A,F = matrix_PE(N,f)
print(A)
print(F)
u_pre = solve_PE(N,f)
u_true = u_exact(x)
error = torch.max(torch.abs(u_true-u_pre))

print(error)


import matplotlib.pyplot as plt
x = x.numpy()
u_pre = u_pre.numpy()
u_true = u_true.numpy()

plt.figure()
plt.plot(x,u_pre,'r',label = 'pre solution')
plt.plot(x,u_true,'b',label = 'true solution')
plt.title('solution of PE for FD of 1d')
plt.legend()
plt.show()
