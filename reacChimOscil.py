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

# Nb de points
n = 1001

# Max de t
tmax = 10

# Temps
t = np.linspace(0, tmax, n)

# Nb de valeurs de B
bmax = 5

# Variable
B = np.linspace(.1, .2, bmax)

# fig, axs = plt.subplots(len(B), 3)

# # Boucle sur les valeurs de B
# for i in range(len(B)):
#     b = B[i]

#     # Résolution du système d'équations différentielles
#     sol = odeint(system, y0, t, args=(b,))

#     # Graphique
#     axs[i][0].plot(t, sol[:, 0], 'b')
#     axs[i][1].plot(t, sol[:, 1], 'g')
#     axs[i][2].plot(t, sol[:, 2], 'r')

# axs[2][0].set(xlabel='x(t)')
# axs[2][1].set(xlabel='y(t)')
# axs[2][2].set(xlabel='z(t)')
# axs[0][0].set(ylabel='b = ' + '.125')
# axs[1][0].set(ylabel='b = ' + '.15')
# axs[2][0].set(ylabel='b = ' + '.175')

# plt.show()

# b = .125
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# sol = odeint(system, y0, t, args=(b,))
# X = np.array([np.copy(t), sol[:, 0]])
# Y = np.array([np.copy(t), sol[:, 1]])
# Z = np.array([np.copy(t), sol[:, 2]])
# # X, Y, Z = axes3d.get_test_data(0.05)
# ax.plot_surface(X, Y, Z, edgecolor='royalblue', lw=0.5, rstride=8, cstride=8, alpha=0.3)  


# ax.contour(X, Y, Z, zdir='x', cmap='coolwarm')
# ax.contour(X, Y, Z, zdir='y', cmap='coolwarm')
# ax.contour(X, Y, Z, zdir='z', cmap='coolwarm')

# ax.set(xlim=(0, 1), ylim=(0, 200), zlim=(0, 40), xlabel='X', ylabel='Y', zlabel='Z')

# plt.show()

fig, axs = plt.subplots(len(B))

# Boucle sur les valeurs de B
for i in range(len(B)):
    b = B[i]

    # Résolution du système d'équations différentielles
    sol = odeint(system, y0, t, args=(b,))

    # Graphique
    axs[i].plot(t, sol[:, 1], 'g')

# axs[2][1].set(xlabel='y(t)')
# axs[0][0].set(ylabel='b = ' + '.125')
# axs[1][0].set(ylabel='b = ' + '.15')
# axs[2][0].set(ylabel='b = ' + '.175')

plt.show()

