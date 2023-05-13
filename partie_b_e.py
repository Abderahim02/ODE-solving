def F(t, Y, mA, G, xA, yA):
    # Y est le vecteur d'état [xB, yB, dxB/dt, dyB/dt]
    xB, yB, dxB, dyB = Y
    r = ((xB - xA)**2 + (yB - yA)**2)**0.5
    ddxB = G * mA * (xA - xB) / r**3
    ddyB = G * mA * (yA - yB) / r**3
    return [dxB, dyB, ddxB, ddyB]

def solution():