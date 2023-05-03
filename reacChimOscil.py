from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

# Constantes
a, c, d = 10, 200, 50

# Définition du système d'équations différentielles
def system(y, t, b):
    dydt = np.zeros((3,))
    dydt[0] = a + b*y[2] - y[0] - y[0]*y[1]**2
    dydt[1] = c*(y[0] + y[0]*y[1]**2 - y[1])
    dydt[2] = d*(y[1] - y[2])
    return dydt

# Conditions initiales
y0 = [1, 0, 0]

# Temps
t = np.linspace(0, .2, 101)

# Variable
B = [.125, .15, .175]

fig, axs = plt.subplots(3, len(B))

# Boucle sur les valeurs de B
for i in range(len(B)):
    b = B[i]

    # Résolution du système d'équations différentielles
    sol = odeint(system, y0, t, args=(b,))

    # Graphique
    axs[i][0].plot(t, sol[:, 0], 'b')
    axs[i][1].plot(t, sol[:, 1], 'g')
    axs[i][2].plot(t, sol[:, 2], 'r')
    
axs[2][0].set(xlabel='x(t)')
axs[2][1].set(xlabel='y(t)')
axs[2][2].set(xlabel='z(t)')
axs[0][0].set(ylabel='b = ' + '.125')
axs[1][0].set(ylabel='b = ' + '.15')
axs[2][0].set(ylabel='b = ' + '.175')


plt.show()

        