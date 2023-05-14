import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
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
    return np.array(t), np.array(y)

def norm(y, z):
    L=[0 for i in range(len(y))]
    for i in range(len(y)):
        for j in range(len(z)):
            if(i==2*j):
                L[i]=L[i]-L[j]
    return np.linalg.norm(L)

##Cette fonction calcule une solution approché avec un paramètre d'erreur epsilon
def meth_epsilon(y0, t0, tf, eps, f, meth):
    N=100
    h=(tf-t0)/N
    y_N=meth_n_step(y0, t0, N, h, f, meth)[1]
    y_2N=meth_n_step(y0, t0, 2*N, h/2, f, meth)[1]
    error = np.abs(norm(y_N, y_2N))
    while(error > eps):
            print(N)
            N=N*2
            h=(tf-t0)/N
            y_N=y_2N
            y_2N=meth_n_step(y0, t0, 2*N, h/2, f, meth)
            error = np.abs(norm(y_N, y_2N))
    return y_N



##Cette fonction permet de dessiner le champ des tangentes de l'équation différentielle (on utilise quiver)


def plot_champ_tang_equ_diff(f, t0, tf, y0, yf, N):
    X, Y, U, V = [1 for i in range(0, N**2)], [1 for i in range(0, N**2)], [1 for i in range(0, N**2)], [1 for i in range(0, N**2)]
    hx, hy = (tf - t0)/N, (yf - y0)/N
    for i in range(0, N):
        for j in range(0, N):
            X[N*i + j] = t0 + i*hx
            Y[N*i + j] = y0 + j*hy
            V[N*i + j] = f(t0 + i*hx, y0 + j*hy)
    colors = np.arctan2(U, V)
    norm = Normalize()
    norm.autoscale(colors)
    colormap = cm.viridis
    plt.quiver(X, Y, U, V, color=colormap(norm(colors)))
    plt.show()

# plot_champ_tang_equ_diff(fonction5, -5, 5, -5, 5, 20)

##les fonction de test de dimension 1 et 2##
t0 = -5
tf = 5
y0 = 1
K = y0/(np.exp(np.arctan(t0)))

def f(t, y):
    return y/(1+t**2)
import matplotlib.cm as cm
def fonction5(x, y):
    return np.sin(x)*np.sin(y)

def y(t):
    return K*np.exp(np.arctan(t))

def test_func_dim1(y0, t0, tf, f, y, eps, meth):
    y_N=meth_epsilon(y0, t0, tf, eps, f, meth)
    print(len(y_N))
    X=np.linspace(t0, tf, len(y_N))
    sol_exact=[y(x) for x in X]
    plt.plot(X, y_N, label='méthode')

def test_all_meth(y0, t0, tf, f, y, eps):
    test_func_dim1(y0, t0, tf, f, y, eps,step_euler)
    test_func_dim1(y0, t0, tf, f, y, eps,step_heun)
    test_func_dim1(y0, t0, tf, f, y, eps,step_mid_point)
    test_func_dim1(y0, t0, tf, f, y, eps,step_runge_kutta_4)
    y_N=meth_epsilon(y0, t0, tf, eps, f, step_euler)
    X=np.linspace(t0, tf, len(y_N))
    sol_exact=[y(x) for x in X]
    plt.plot(X, sol_exact, label="sol exact")
    plt.legend()
    plt.show()

eps=0.0001
#test_all_meth(y0, t0, tf, f, y, eps)

def test_func_dim2():
    return 0

# Création du graphe de convergence en fonction des différentes méthodes
# t0 = -5
# tf = 5
# y0 = 1
# K = y0/(np.exp(np.arctan(t0)))
# lim = 8


# def f(t, y):
#     return y/(1+t**2)

# def y(t):
#     return K*np.exp(np.arctan(t))

# n = [10**i for i in range(1,lim)]

# def res_meth(y0, t0, tf,  f, meth, lim):
#     array = np.zeros([1, lim - 1])[0]
#     for i in range(1, lim) :
#         t, y_p = meth_n_step(y0, t0, 10**i, (tf-t0)/10**i, f, meth)
#         val = np.array([y(h) for h in t])
#         array[i - 1] = max(val-y_p)
#     return array
    


# euler = res_meth(y0, t0, tf, f, step_euler, lim)
# mid_point = res_meth(y0, t0, tf, f, step_mid_point, lim)
# heun = res_meth(y0, t0, tf,f,  step_heun, lim)
# runge_k = res_meth(y0, t0, tf, f, step_runge_kutta_4, lim)

# plt.grid()


# plt.plot(n, euler, label='Euler') 
# plt.plot(n, mid_point, label='Middle Point')
# plt.plot(n, heun, label='Heun')
# plt.plot(n, runge_k, label='Runge Kutta')

# plt.yscale("log")
# plt.xscale("log")

# plt.xlabel("Nombre de subdivisions")
# plt.ylabel("Précision du résultat")
# plt.title("Précision du calcul de l'équation différentielle y'= y/(1+t**2) en fonction du nombre de subdivisions, pour chaque méthode de résolution")

# plt.legend()

# plt.show()
