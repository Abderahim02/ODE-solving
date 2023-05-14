from res_equa_diff import *

t0 = -5
tf = 5
y0 = 1
eps=0.0001

def fonction_1(t, y):
    return y/(1+t**2)

def sol_exact_1(y0, t0, t):
    K=y0*np.exp(np.arctan(t0))
    return K*np.exp(np.arctan(t))


def plot_test_sol(y0, t0, tf, eps, f, sol):
    y_N_euler=meth_epsilon(y0, t0, tf, eps, f, step_euler)
    X_1=np.linspace(t0, tf, len(y_N_euler))
    y_N_heun=meth_epsilon(y0, t0, tf, eps, f, step_heun)
    X_2=np.linspace(t0, tf, len(y_N_heun))
    y_N_midp=meth_epsilon(y0, t0, tf, eps, f, step_mid_point)
    X_3=np.linspace(t0, tf, len(y_N_midp))
    y_N_rk4=meth_epsilon(y0, t0, tf, eps, f, step_runge_kutta_4)
    X_4=np.linspace(t0, tf, len(y_N_rk4))
    sol_exact=np.array([sol(y0, t0, x) for x in X_4])

    plt.plot(X_1, y_N_euler[:, 0], label="méthode d'euler")
    plt.plot(X_2, y_N_heun[:, 0], label="méthode d'heun")
    plt.plot(X_3, y_N_midp[:, 0], label="méthode de mid point")
    plt.plot(X_4, y_N_rk4[:, 0], label="méthode de rk4")
    plt.plot(X_4, sol_exact[:, 0, 0], label="la solution exacte")
    plt.title("La solution par différentes méthodes")
    plt.legend()
    plt.show()



y0 = np.array([[1], [0]])
def fonction_2(t, y):
    return np.array([-y[1],y[0]])

def sol_exact_2(y0, t0, t):
    K=1#np.array([np.cos(t0)], [np.sin(t0)]])
    return K*np.array([[np.cos(t)], [np.sin(t)]])

plot_test_sol(y0, t0, tf, eps, fonction_2, sol_exact_2)