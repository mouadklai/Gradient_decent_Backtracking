import numpy as np
import numpy.linalg as l

def gradient_method_backtracking(f,g,x_0,beta,alpha,epsilon,k_max):
    x=x_0
    k=0
    while(l.norm(g(x))>epsilon):
        k+=1 #nombre d'iteration
        t=1  
        while(f(x)-f(x-t*g(x))-alpha*(l.norm(g(x))**2)<0): #Armijo
            t*=beta
        x+=-t*g(x)
        if k==k_max:
            break

    return x,f(x),g(x),k


def gradient_method_fixe(f,g,x_0,beta,alpha,epsilon,k_max,t):
    x_k=x_0
    k=0
    c='converged'
    while(l.norm(g(x_k))>epsilon):
        k+=1 #nombre d'iteration
        x_k+=-t*g(x_k)
        if k==k_max:
            c="Maximum number of iterations="+str(k)+"reached"
            break
    return x_k,f(x_k),g(x_k),c,k

print("gradient_method_fixe alpha = 1",gradient_method_fixe(lambda x: x**2,lambda x:2*x,5,0.5,0.01,0.001,1000,1))
print("gradient_method_fixe alpha = 0.1",gradient_method_fixe(lambda x: x**2,lambda x:2*x,5,0.5,0.01,0.001,1000,0.1))

print("gradient_method_backtraking",gradient_method_backtracking(lambda x: x**2,lambda x:2*x,5,0.5,0.01,1e-3,3000))