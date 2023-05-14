from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np
def F(Y ,t, mA, G, xA, yA):
    # Y est le vecteur d'état [xB, yB, dxB/dt, dyB/dt]
    xB, yB, dxB, dyB = Y
    r = ((xB - xA)**2 + (yB - yA)**2)**0.5
    ddxB = G * mA * (xA - xB) / r**3
    ddyB = G * mA * (yA - yB) / r**3
    return [dxB, dyB, ddxB, ddyB]

def solution_1():
    mA = 100.0  
    G = 1.0  
    xA = 0.0  
    yA = 0.0  
    xB0 = 15.0 
    yB0 = 5.0 
    dxB0 = 0.0 
    dyB0 = 2.0 
    Y0 = np.array([xB0, yB0, dxB0, dyB0])
    t = np.linspace(0, 25, 1000) # time array
    sol = odeint(F, Y0, t, args=(mA, G, xA, yA))
    # Extract the xB and yB values from the solution array
    xB = sol[:, 0]
    yB = sol[:, 1]
    fig, ax = plt.subplots()
    ax.plot(xB, yB)
    ax.plot(xA, yA, 'ro', label='A')
    ax.legend()
    ax.set_xlabel('xB')
    ax.set_ylabel('yB')
    ax.set_title('Trajectoire du corps B')
    plt.show()



def F_3corps(y, t, mA, mB, G):
    xC, yC, vxC, vyC = y
    # la distance entre A et C
    rC_A = np.sqrt(xC**2 + yC**2)

    # la distance entre A et C
    rC_B = np.sqrt((xC - 1)**2 + yC**2)
    rC_A3 = rC_A**3
    rC_B3 = rC_B**3

    # remplacer les valeurs des derivee dans le systeme
    dxC_dt = vxC
    dyC_dt = vyC
    dvxC_dt = -G * mA * xC / rC_A3 - G * mB * (xC - 1) / rC_B3
    dvyC_dt = -G * mA * yC / rC_A3 - G * mB * yC / rC_B3
    
    dy_dt = [dxC_dt, dyC_dt, dvxC_dt, dvyC_dt]
    return dy_dt

def solution_2():
    mA = 100.0  
    G = 1
    mB = 0.5
    xA = 1.0  
    yA = 1.0  
    xC0 = 5.0 
    yC0 = 2.0 
    dxC0 = 0.0 
    dyC0 = 2.0 
    Y0 = np.array([xC0, yC0, dxC0, dyC0])
    t = np.linspace(0, 25, 1000) # time array
    sol = odeint(F_3corps, Y0, t, args=(mA, mB, G))
    # Extract the xB and yB values from the solution array
    rB = 1.0
    xB = xA + np.cos(2*np.pi*t/(2*np.pi))
    yB = yA + np.sin(2*np.pi*t/(2*np.pi))
    xC = sol[:, 0]
    yC = sol[:, 1]
    fig, ax = plt.subplots()
    ax.plot(xC, yC, label = 'C')
    ax.plot(xB, yB, 'g', label='B')
    ax.plot(xA, yA, 'ro', label='A')
    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Trajectoire du système à trois corps')
    plt.show()

def nouveau_ref():
    mA = 100.0  
    G = 1
    mB = 0.5
    xA = 1.0  
    yA = 1.0  
    xC0 = 5.0 
    yC0 = 2.0 
    dxC0 = 0.0 
    dyC0 = 2.0 
    Y0 = np.array([xC0, yC0, dxC0, dyC0])
    t = np.linspace(0, 25, 1000) 
    # Calcul de la trajectoire de C et de B (comme précédemment)
    sol_C = odeint(F_3corps, Y0, t, args=(mA, mB, G))
    xB = np.cos(2*np.pi*t/(2*np.pi))
    yB = np.sin(2*np.pi*t/(2*np.pi))
    t0=20
    # Transformation de la trajectoire de C
    theta = -2 * np.pi * t0 / (2 * np.pi)  # Angle de rotation
    R = np.array([[np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]])  # Matrice de rotation
    C_rotated = R @ sol_C[:, :2].T  # Transformation des coordonnées de C
    xC_rotated, yC_rotated = C_rotated[0], C_rotated[1]  # Récupération des coordonnées

    # Tracé de la trajectoire de C dans le nouveau référentiel
    fig, ax = plt.subplots()
    ax.plot(xC_rotated, yC_rotated)
    ax.plot(xB, yB, 'g', label='B')
    ax.plot(xA, yA, 'ro', label='A')
    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Trajectoire de C dans le référentiel fixe de A et B')
    plt.show()
if __name__ == "__main__":
    solution_1()
    solution_2()
    nouveau_ref()
