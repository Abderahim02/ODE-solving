import numpy as np
import matplotlib.pyplot as plt
from math import sqrt


##Implimentation de la méthode d'Euler
def step_euler(y, t, h, f):
    return y+h*f(t, y) #y{n+1}

##Implimentation de la méthode du point milieu
def step_mid_point(y, t, h, f):
    y_int=y+(h/2)*f(t,y) # un point intérmediere y_int = y_{n+(1/2)}
    return y+h*f(t+(h/2), y_int) #le résultat est y_{n+1}

##Implimentation de la méthode de Heun
def step_heun(y, t, h, f):
    p1=f(t,y)
    y2=y+h*p1
    p2=f(t+h, y2)
    return y+(h/2)*(p1+p2) #y{n+1}

##Implimentation de la méthode de Runge-Kutta d'ordre 4
def step_runge_kutta_4(y, t, h, f):
    p1=f(t,y)
    y2=y+(1/2)*h*p1
    p2=f(t+(h/2), y2)
    y3=y+(h/2)*p2
    p3=f(t+(2/h), y3)
    y4=y+h*p3
    p4=f(t+h, y4)
    return y+(h/6)*(p1+2*p2+2*p3+p4)

def norme_infinie(y):
    s=0
    for i in range(len(y)):
        s+=y[i]**2
    return sqrt(s)


##Cette fonction calcule le nombre N de pas de taille constante h
def meth_n_step(y0, t0, N, h, f, meth):
    t=[t0 for i in range(N)]
    y=[y0 for i in range(N)]
    for i in range(1, N):
        t[i]=t0+i*h
        y[i]=meth(y[i-1], t[i-1], h, f)
    return t,y


##Cette fonction calcule une solution approché avec un paramètre d'erreur epsilon
def meth_epsilon(y0, t0, tf, eps, f, meth):
    N=100
    h=(tf-t0)/N
    y=y0
    t=t0
    while t<tf:
        h=min(h, tf-t)/N
        y_N=meth_n_step(y, t, 1, h, f, meth)
        y_2N=meth_n_step(y, t, 2, h/2, f, meth)
        erreur=abs(norme_infinie(y-N)-norme_infinie(y_2N))
        if erreur<= eps:
            y=y_N
            t+=h
        h=h*(eps/erreur)
    return y

##Cette fonction permet de dessiner le champ des tangentes de l'équation différentielle (on utilise quiver)

def plot_tang_equ_diff(f, t, y, res, meth):
    X, Y = np.meshgrid(np.linspace(t[0], t[1], res), np.linspace(y[0], y[1], res))
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    for i in range(res):
        for j in range(res):
            y = np.array([X[i,j], Y[i,j]])
            u, v = meth(y, 0, 0.1, f)
            U[i,j] = u
            V[i,j] = v
    plt.quiver(X, Y, U, V)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

##les fonction de test de dimension 1 et 2##

def test_func_dim1():
    return 0

def test_func_dim2():
    return 0


def f(y, t):
    return -y

sol=meth_n_step(1, 0, 10, 0.1, f, step_euler)
print(sol)

def f(y, t):
    return np.array([-y[0], -y[1]])

t = [-5, 5]
y = [-5, 5]
res = 20
