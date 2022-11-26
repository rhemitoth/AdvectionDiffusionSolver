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
    print (area_under_curve)

#Gaussian Function
def gaussian(x, mu, sigma):
    denom = sigma * ((2 * math.pi)**0.5)
    numerator = math.exp(((x - mu)**2) / ((sigma) **2) / -2)
    res = numerator / denom
    return res

#Model------------------------------------------------------------------------------------------------------------------
# Bounds
start = 0  # start bound
stop = 40 # stop bound
center = statistics.mean([start,stop])

# Model parameters
dt = 0.1  # delta t
dx = 0.05  # delta x
T = 1000  # Total time
Nt = int(T / dt)  # Number of time steps
Nx = int((abs(stop-start))/dx)  # Number of x steps
mean_sl = 0.05 # mean step length
k = (mean_sl**2)/2/dt # diffusion coefficient equal to mean step length squared
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
                   mu = 20,
                   sigma = 1) #Gaussin IC
    IC.append(res)
u[0][1] = IC #Populating initial condition in array u

#Explicit Finite Difference Scheme
# Periodic Boundary Conditions/Zero Flux
# Forward difference approximation for time derivative (order dt)
# Central difference approximation for space derivative (order dx^2)


for j in range(0,Nt-1):
    for i in range(0,Nx+1):
        if i == 0:
            u[j + 1][1][i] = u[j][1][i] + r * (u[j][1][i + 1] - 2 * u[j][1][i] + u[j][1][Nx])
        elif i == Nx:
            u[j + 1][1][i] = u[j][1][i] + r * (u[j][1][0] - 2 * u[j][1][i] + u[j][1][i - 1])
        else:
            u[j + 1][1][i] = u[j][1][i] + r * (u[j][1][i + 1] - 2 * u[j][1][i] + u[j][1][i - 1])

for j in range(0,Nt):
    integrate(u = u[j][1],
              dx = dx,
              x_vals = Xs)
#Checking for conservation
#for j in range(0,Nt):
    #integrate(u=u,dx=dx)

#Ploting u(x,t) at T = 1, 2, 3, 4, and 5
t0 = u[0][1]
t1 = u[int(0.2*Nt)][1]
t2 = u[int(0.4*Nt)][1]
t3 = u[int(0.6*Nt)][1]
t4 = u[int(0.8*Nt)][1]
t5 = u[int(Nt-1)][1]

plt.plot(Xs,t0, label = 't0')
plt.plot(Xs,t1, label = 't1')
plt.plot(Xs,t2, label = 't2')
plt.plot(Xs,t3, label = 't3')
plt.plot(Xs,t4, label = 't4')
plt.plot(Xs,t5, label ='t5')
plt.show()

fig = plt.figure()
camera = Camera(fig)
for i in range(Nt):
    Ys= u[i][1]
    plt.plot(Xs,Ys, color = "blue")
    camera.snap()
animation = camera.animate()
animation.save('Diffusion.mp4', writer = 'ffmpeg')