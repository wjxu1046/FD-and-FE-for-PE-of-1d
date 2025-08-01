import torch
import numpy as np


def u_exact(x):
    return torch.sin(np.pi * x)

def f(x):
    return np.pi ** 2 * torch.sin(np.pi * x)


    
def two_gauss_int(f,a,b,n=2):
    int = torch.zeros(2)
    guass_points = {
        1:[0.0],
        2:[-1.0/np.sqrt(3), 1.0/np.sqrt(3)]
    }
    guass_weight = {
        1:[2.0],
        2:[1.0, 1.0]
    }
    
    points = guass_points[n]
    weights  =guass_weight[n]
    for i in range(n):
        x = 0.5 * (b - a) * points[i] + 0.5 * (a + b)
        phi0 = (b - x) / (b-a)
        phi1 = (x - a) / (b-a)
        int[0] += weights[i] * phi0 * f(x)
        int[1] += weights[i] * phi1 * f(x)
        
        
    int *= 0.5 * (b-a)
    
    return int

def matrix_PE(N,f):
    K = torch.zeros(N+1,N+1)
    F = torch.zeros(N+1)
    h = 1.0/N
    x = torch.linspace(0,1,N+1)
    for i in range(N):
        Ke = 1.0 / h * torch.tensor([[1., -1.],
                                         [-1., 1.]])
        Fe = two_gauss_int(f,x[i],x[i+1])
        K[i:i+2,i:i+2] += Ke
        F[i:i+2] += Fe
        
        K[0,:] = 0
        K[0,0] = 1
        F[0] = 0
        
        K[-1,:] = 0
        K[-1,-1] = 1
        F[-1] = 0  
        
    return K, F

def solve_PE(N,f):
    K, F = matrix_PE(N,f)
    u_sol = torch.linalg.solve(K,F)
    return u_sol


N = 20
x = torch.linspace(0,1,N+1)
# K,F = matrix_PE(N,f)
# print(K)
# print(F)
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
plt.title('solution of PE for FE of 1d')
plt.legend()
plt.show()
