#Modules---------------------------------------------------------------------------------------------------------------

from matplotlib import pyplot as plt
import numpy as np
import math
from celluloid import Camera

#Functions-------------------------------------------------------------------------------------------------------------

#Trapezoid Integration Function
def integrate(u,dx,x_vals):
    area_under_curve = 0
    for i in range(0,len(x_vals)-1):
        area = dx * (u[i+1]+u[i]) / 2
        area_under_curve = area_under_curve + area
    return area_under_curve

#Gaussian Function
def gaussian(x, mu, sigma):
    denom = sigma * ((2 * math.pi)**0.5)
    numerator = math.exp(((x - mu)**2) / ((sigma) **2) / -2)
    res = numerator / denom
    return res

# Habitat Preference
def preference(x):
    pi = math.pi
    res = 0.1 + math.sin(2 * pi * x / 50) ** 2  # Example 1
    #res = math.exp(-0.01*abs(x-50)**2) # Example 2
    res = np.float64(res)
    return res

# Derivative of Habitat Preference Function
def preference_slope(x):
    pi = math.pi
    res = (2 * pi / 25) * math.cos(pi * x / 25) * math.sin(pi * x / 25)  # Example 1
    #res = -0.02*math.exp(-0.01*(x-50)**2)*(x-50) # Example 2
    res = np.float64(res)
    return res

#Model------------------------------------------------------------------------------------------------------------------

# Bounds
start = 0 # start bound
stop = 50 # stop bound


# Model parameters
dt = 0.01 # delta t
dx = 0.05  # delta x
T = 100 # Total time
Nt = int(T / dt)  # Number of time steps
Nx = int((abs(stop-start))/dx)  # Number of x steps
mean_sl = 0.04 #mean step length

# Initializing x values where u(x,t) will be calculated
Xs = np.arange(start, stop + dx, dx)

# Initializing array with Nt, Nx columns, and [x,u(x,t)] per cell
u = np.zeros((Nt, 2, len(Xs)))

# Habitat Preference Function
w = np.zeros((1,2,len(Xs)))
w[0][0] = Xs
for i in range(0,len(Xs)):
    w[0][1][i] = preference(w[0][0][i])

# Spatial Derivative of Habitat Preference Function
wx = np.zeros((1,2,len(Xs)))
wx[0][0] = Xs
for i in range(0,len(Xs)):
    wx[0][1][i] = preference_slope(w[0][0][i])

# Populating x values in array u
for j in range(0, Nt):
    u[j][0]=Xs

# Setting Initial Condition
IC=[]
for i in range(0, len(Xs)):
    x = Xs[i]
    res = gaussian(x = x,
                   mu = 23,
                   sigma = 0.5) #Gaussin IC
    res = np.float64(res)
    IC.append(res)
u[0][1] = IC #Populating initial condition in array u

# Finite Difference Scheme
for j in range(0,Nt-1):
    if ((((j + 1) / Nt) * 100 % 1 == 0)):
        print(str((j + 1) / Nt * 100), " %")
    for i in range(0,Nx+1):
        c = (mean_sl**2) * wx[0][1][i] / w[0][1][i] / dt  # Advection with variable speed
        #c = 0.1 # advection with constant speed
        p = c * dt / dx /2 # Courant number
        #First-order upwind scheme
        #if c > 0:
            #if i == 0:
                #u[j+1][1][i] = u[j][1][i] - p * (u[j][1][i] - u[j][1][-1])
            #else:
                #u[j + 1][1][i] = u[j][1][i] - p * (u[j][1][i] - u[j][1][i - 1])
        #Second-order upwind scheme
        if c > 0:
            if i == 0:
                u[j + 1][1][i] = -p * (3*u[j][1][i] - 4 * u[j][1][Nx] + u[j][1][Nx-1]) + u[j][1][i]
            elif i == 1:
                u[j + 1][1][i] = -p * (3 * u[j][1][i] - 4 * u[j][1][i - 1] + u[j][1][Nx]) + u[j][1][i]
            else:
                u[j + 1][1][i] = -p * (3*u[j][1][i] - 4 * u[j][1][i-1] + u[j][1][i-2]) + u[j][1][i]
        else:
            if i == Nx:
                u[j + 1][1][i] = u[j][1][i] - p * (-u[j][1][1] + 4 * u[j][1][0] - 3 * u[j][1][Nx])
            elif i == (Nx-1):
                u[j + 1][1][i] = u[j][1][i] - p * (-u[j][1][0] + 4 * u[j][1][Nx] - 3 * u[j][1][Nx-1])
            else:
                u[j + 1][1][i] = u[j][1][i] - p * (-u[j][1][i+2] + 4 * u[j][1][i + 1] - 3 * u[j][1][i])
    area = integrate(u = u[j + 1][1],
                     dx = dx,
                     x_vals = Xs)
    u[j+1][1] = u[j+1][1]/area

# Checking for conservation of area under the curve
for j in range(0,Nt):
    res = integrate(u = u[j][1],
              dx = dx,
              x_vals = Xs)
    print(res)

# Plot
t0 = u[0][1]
t1 = u[int(0.2*Nt)][1]
t2 = u[int(0.4*Nt)][1]
t3 = u[int(0.6*Nt)][1]
t4 = u[int(0.8*Nt)][1]
t5 = u[int(Nt-1)][1]

plt.figure
plt.plot(Xs,w[0][1], color = "#5a5a5a", label = fr"$w(x)$", linestyle = "dashed")
plt.plot(Xs,t0, label = fr'$u(x,0)$', color = "#003f5c" )
plt.plot(Xs,t1, label = fr'$u(x,{int(0.2*T)})$', color = "#444e86")
plt.plot(Xs,t2, label = fr'$u(x,{int(0.4*T)})$', color = "#955196")
plt.plot(Xs,t3, label = fr'$u(x,{int(0.6*T)})$', color = "#dd5182")
plt.plot(Xs,t4, label = fr'$u(x,{int(0.8*T)})$', color = "#ff6e54")
plt.plot(Xs,t5, label =fr'$u(x,{int(T)})$', color = "#ffa600")
plt.legend(loc="upper right")
plt.title("Figure 4. Advection with Variable Speed")
plt.show()

#Animation
fig1, ax1 = plt.subplots()
camera = Camera(fig1)
for i in range(Nt):
    Ys= u[i][1]
    if i%500 == 0:
        if i == 0:
            ax1.plot(w[0][0], w[0][1], color = "#5a5a5a", label = "w(x)", linestyle = "dashed")
            x = ax1.plot(Xs, Ys, color="#959e19", label="u(x,t)")
        else:
            ax1.plot(w[0][0], w[0][1], color="#5a5a5a", linestyle="dashed")
            x = ax1.plot(Xs, Ys, color="#959e19")
        ax1.legend(loc="upper right")
        camera.snap()
animation = camera.animate()
animation.save('Advection.mp4', writer = 'ffmpeg')
