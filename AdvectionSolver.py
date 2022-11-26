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

# Habitat Preference
def preference(x):
    pi = math.pi
    res = 0.1 + math.sin(2*pi*x/50)**2 #sin preference
    #res = 0.2 + 0.1*math.cos(x*pi/25) #cos preference
    #res = math.exp(-abs(x-50)**2)
    #res = 0.1 / (1 + math.exp(-(x-40))) #logistic preference
    #res = 0.5 * math.exp(-6 * (abs(x-10)**2))
    res = np.float64(res)
    return res

# Derivative of Habitat Preference Function
def preference_slope(x):
    pi = math.pi
    res = (2*pi/25)*math.cos(pi*x/25)*math.sin(pi*x/25) #sin preference
    #res = -0.1*(pi/25)*math.sin(x*pi/25) #cos IC
    #res = math.exp(-(x**2)+(100*x)-2500)*((-2*x)+100)
    #res = (math.exp(-x+40) / ((1 + math.exp(40-x))**2))*0.1 #logistic preference
    #res = -6 * math.exp(-6*(x-10)**2)*(x-10)
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
mean_sl = 0.05 #mean step length

# Initializing x values where u(x,t) will be calculated
Xs = np.arange(start, stop + dx, dx)

# Initializing array with Nt, Nx columns, and [x,u(x,t)] per cell
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
                   mu = 26,
                   sigma = 0.5) #Gaussin IC
    res = np.float64(res)
    IC.append(res)
u[0][1] = IC #Populating initial condition in array u

#Explicit Finite Difference Scheme
# Periodic Boundary Conditions/Zero Flux
# Forward difference approximation for time derivative (order dt)
# Central difference approximation for space derivative (order dx^2)

for j in range(0,Nt-1):
    if (((j/Nt) % 0.1 == 0)):
        print(str(j/Nt * 100)," %")
    for i in range(0,Nx+1):
        c = (mean_sl**2) * wx[0][1][i] / w[0][1][i] / dt  # Advection coefficient
        #c=0.1
        p = c * dt / dx /2 # Courant number
        #if c > 0:
          #  if i == 0:
               # u[j+1][1][i] = u[j][1][i] - p * (u[j][1][i] - u[j][1][-1])
           # else:
            #    u[j + 1][1][i] = u[j][1][i] - p * (u[j][1][i] - u[j][1][i - 1])
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

fig = plt.figure()

camera = Camera(fig)
for i in range(Nt):
    Ys= u[i][1]
    if i%500 == 0:
        plt.plot(w[0][0], w[0][1])
        plt.plot(Xs,Ys, color = "blue")
        camera.snap()
animation = camera.animate()
animation.save('Advection.mp4', writer = 'ffmpeg')

#plt.plot(Xs,t0, label = 't0')
#plt.plot(Xs,t1, label = 't1')
#plt.plot(Xs,t2, label = 't2')
#plt.plot(Xs,t3, label = 't3')
#plt.plot(Xs,t4, label = 't4')
#plt.plot(Xs,t5, label ='t5')
plt.plot(w[0][0],w[0][1])
#plt.text(5,0.7,"dt = 0.01 \n dx = 0.05 \n c = 0.1")
plt.show()

