#Modules---------------------------------------------------------------------------------------------------------------

from matplotlib import pyplot as plt
import numpy as np
import math
from celluloid import Camera
import statistics

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

#Model------------------------------------------------------------------------------------------------------------------
# Bounds
start = 0  # start bound
stop = 20 # stop bound

# Model parameters
dt = 0.1  # delta t
dx = 0.05  # delta x
T = 500  # Total time
Nt = int(T / dt)  # Number of time steps
Nx = int((abs(stop-start))/dx)  # Number of x steps
mean_sl = 0.04 # mean step length
k = (mean_sl**2)/2/dt # diffusion coefficient
r = k * dt / dx / dx  # Fourier number

# Initializing array with Nt, Nx columns, and [x,u(x,t)] per cell
u = np.zeros((Nt, 2, Nx+1))

# Initializing x values where u(x,t) will be calculated
Xs = np.arange(start, stop+dx, dx)

# Populating x values in array u
for j in range(0, Nt):
    u[j][0]=Xs

#Setting Initial Condition
IC=[]
for i in range(0, len(Xs)):
    x = Xs[i]
    res = gaussian(x = x,
                   mu = 10,
                   sigma = 1) #Gaussin IC
    IC.append(res)
u[0][1] = IC #Populating initial condition in array u

#Finite Difference Scheme
for j in range(0,Nt-1):
    if ((((j+1)/Nt)*100 % 1 == 0)):
        print(str((j+1)/Nt * 100)," %")
    for i in range(0,Nx+1):
        if i == 0:
            u[j + 1][1][i] = u[j][1][i] + r * (u[j][1][i + 1] - 2 * u[j][1][i] + u[j][1][Nx])
        elif i == Nx:
            u[j + 1][1][i] = u[j][1][i] + r * (u[j][1][0] - 2 * u[j][1][i] + u[j][1][i - 1])
        else:
            u[j + 1][1][i] = u[j][1][i] + r * (u[j][1][i + 1] - 2 * u[j][1][i] + u[j][1][i - 1])

# Checking for conservation of area under the curve
for j in range(0,Nt):
    res = integrate(u = u[j][1],
              dx = dx,
              x_vals = Xs)
    print(res)

t0 = u[0][1]
t1 = u[int(0.01*Nt)][1]
t2 = u[int(0.1*Nt)][1]
t3 = u[int(0.25*Nt)][1]
t4 = u[int(0.5*Nt)][1]
t5 = u[int(Nt-1)][1]

# Animation
fig = plt.figure()
camera = Camera(fig)
for i in range(Nt):
    if i % 500 == 0:
        Ys= u[i][1]
        plt.plot(Xs,Ys, color = "blue")
        camera.snap()
animation = camera.animate()
animation.save('Diffusion.mp4', writer = 'ffmpeg')

# Plot
plt.figure()
plt.plot(Xs,t0, label = r'$u(x,0)$', color = "#003f5c" )
plt.plot(Xs,t1, label = fr'$u(x,{int(0.01*T)})$', color = "#444e86")
plt.plot(Xs,t2, label = fr'$u(x,{int(0.1*T)})$', color = "#955196")
plt.plot(Xs,t3, label = fr'$u(x,{int(0.25*T)})$', color = "#dd5182")
plt.plot(Xs,t4, label = fr'$u(x,{int(0.5*T)})$', color = "#ff6e54")
plt.plot(Xs,t5, label =fr'$u(x,{int(T)})$', color = "#ffa600")
plt.legend(loc="upper right")
plt.text(0,0.75,"Gaussian IC ( " + r'$\mu = 10$' + ", " + r'$\sigma = 0.5)$' + f"\ndx = {dx}, dt = {dt}")
plt.title("Figure 1. Central Difference Approximation for Diffusion")
plt.show()
