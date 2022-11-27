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
    #res = 0.1 + math.sin(2 * pi * x / 50) ** 2  # w1
    #res = 1 + 0.5*math.cos(x*pi/25) # w2
    res = math.exp(-0.01*abs(x-50)**2) # w3
    #res = 0.1 / (1 + math.exp(-(x-40))) # w4
    #res = 0.5 * math.exp(-6 * (abs(x-10)**2)) #w5
    res = np.float64(res)
    return res

# Derivative of Habitat Preference Function
def preference_slope(x):
    pi = math.pi
    #res = (2 * pi / 25) * math.cos(pi * x / 25) * math.sin(pi * x / 25)  # w1
    #res = -0.5*(pi/25)*math.sin(x*pi/25) #w2
    res = -0.02*math.exp(-0.01*(x-50)**2)*(x-50) #w3
    #res = (math.exp(-x+40) / ((1 + math.exp(40-x))**2))*0.1 #w4
    #res = -6 * math.exp(-6*(x-10)**2)*(x-10) #w5
    res = np.float64(res)
    return res

#Model------------------------------------------------------------------------------------------------------------------
# Bounds
start = 0 # start bound
stop = 100 # stop bound


# Model parameters
dt = 0.01 # delta t
dx = 0.05  # delta x
T = 1000 # Total time
Nt = int(T / dt)  # Number of time steps
Nx = int((abs(stop-start))/dx)  # Number of x steps
mean_sl = 0.04 #mean step length
k = (mean_sl**2)/2/dt # diffusion coefficient equal to mean step length squared
r = k * dt / dx / dx  # Fourier number

# Initializing x values where u(x,t) will be calculated
Xs = np.arange(start, stop + dx, dx)

# Initializing array with Nt row, Nx columns, and [x,u(x,t)] per cell
u = np.zeros((Nt, 2, len(Xs)))

#Habitat Preference Function
w = np.zeros((1,2,len(Xs)))
w[0][0] = Xs
for i in range(0,len(Xs)):
    w[0][1][i] = preference(w[0][0][i])

#Spatial Derivative of Habitat Preference Function
wx = np.zeros((1,2,len(Xs)))
wx[0][0] = Xs
for i in range(0,len(Xs)):
    wx[0][1][i] = preference_slope(w[0][0][i])

# Populating x values in array u
for j in range(0, Nt):
    u[j][0]=Xs

#Setting Initial Condition
IC=[]
for i in range(0, len(Xs)):
    x = Xs[i]
    res = gaussian(x = x,
                   mu = 25,
                   sigma = 0.5) #Gaussin IC
    res = np.float64(res)
    IC.append(res)
u[0][1] = IC #Populating initial condition in array u

#Fourier Number Flag
if r > 0.5:
    print("Fourier Number " + str(r) +  " > 0.5. Adjust mean step length, dx, or dt")

#Explicit Finite Difference Scheme
# Periodic Boundary Conditions/Zero Flux
# Forward difference approximation for time derivative (order dt)
# Central difference approximation for space derivative (order dx^2)

for j in range(0,Nt-1):
    if ((((j+1)/Nt) % 0.01 == 0)):
        print(str((j+1)/Nt * 100)," %")
    for i in range(0,Nx+1):
        c = (mean_sl**2) * wx[0][1][i] / w[0][1][i] / dt  # Advection coefficient
        #Courant Number flag
        p = c * dt / dx /2 # Courant number
        if p > 1:
            print("Courant number > 1. Adjust mean step length, preference function, dx, or dt")
        if c > 0:
            if i == 0:
                u[j + 1][1][i] = r * (u[j][1][i + 1] - 2 * u[j][1][i] + u[j][1][Nx]) -p * (3*u[j][1][i] - 4 * u[j][1][Nx] + u[j][1][Nx-1]) + u[j][1][i]
            elif i == 1:
                u[j + 1][1][i] = r * (u[j][1][i + 1] - 2 * u[j][1][i] + u[j][1][i - 1]) -p * (3 * u[j][1][i] - 4 * u[j][1][i - 1] + u[j][1][Nx]) + u[j][1][i]
            elif i == Nx:
                u[j + 1][1][i] = r * (u[j][1][0] - 2 * u[j][1][i] + u[j][1][i - 1]) - p * (3 * u[j][1][i] - 4 * u[j][1][i - 1] + u[j][1][i - 2]) + u[j][1][i]
            else:
                u[j + 1][1][i] = r * (u[j][1][i + 1] - 2 * u[j][1][i] + u[j][1][i - 1]) -p * (3*u[j][1][i] - 4 * u[j][1][i-1] + u[j][1][i-2]) + u[j][1][i]
        else:
            if i == Nx:
                u[j + 1][1][i] = u[j][1][i] - p * (-u[j][1][1] + 4 * u[j][1][0] - 3 * u[j][1][Nx]) + r * (u[j][1][0] - 2 * u[j][1][i] + u[j][1][i - 1])
            elif i == (Nx-1):
                u[j + 1][1][i] = u[j][1][i] - p * (-u[j][1][0] + 4 * u[j][1][Nx] - 3 * u[j][1][Nx-1]) + r * (u[j][1][i + 1] - 2 * u[j][1][i] + u[j][1][i - 1])
            elif i == 0:
                u[j + 1][1][i] = u[j][1][i] - p * (-u[j][1][i + 2] + 4 * u[j][1][i + 1] - 3 * u[j][1][i]) + r * (u[j][1][i + 1] - 2 * u[j][1][i] + u[j][1][Nx])
            else:
                u[j + 1][1][i] = u[j][1][i] - p * (-u[j][1][i+2] + 4 * u[j][1][i + 1] - 3 * u[j][1][i]) + r * (u[j][1][i + 1] - 2 * u[j][1][i] + u[j][1][i - 1])
    area = integrate(u = u[j + 1][1],
                     dx = dx,
                     x_vals = Xs)
    u[j+1][1] = u[j+1][1]/area


for j in range(0,Nt):
    res = integrate(u = u[j][1],
              dx = dx,
              x_vals = Xs)
    print(res)

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

fig_title = f"w(x) = exp(-0.1*|x|^2) \n dx = {dx}, dt = {dt}, mean sl = {mean_sl}"
fig1 = plt.figure()
fig1.suptitle(fig_title, fontsize = 8)
camera = Camera(fig1)
for i in range(Nt):
    Ys= u[i][1]
    if i%500 == 0:
        plt.plot(w[0][0], w[0][1], color = "black")
        plt.plot(Xs,Ys, color = "blue")
        camera.snap()
animation = camera.animate()
animation.save('AdvectionDiffusion.mp4', writer = 'ffmpeg')

fig2 = plt.figure()
fig2.suptitle(fig_title, fontsize = 8)
t0 = u[0][1]
t1 = u[int(0.2*Nt)][1]
t2 = u[int(0.4*Nt)][1]
t3 = u[int(0.6*Nt)][1]
t4 = u[int(0.8*Nt)][1]
t5 = u[int(Nt-1)][1]
plt.plot(w[0][0],w[0][1], color = "black")
plt.plot(Xs,t0, label = 't0')
plt.plot(Xs,t1, label = 't1')
plt.plot(Xs,t2, label = 't2')
plt.plot(Xs,t3, label = 't3')
plt.plot(Xs,t4, label = 't4')
plt.plot(Xs,t5, label ='t5')
plt.legend(loc="upper right")
plt.savefig("AdvectionDiffusion.png", dpi = 350)
plt.show()


