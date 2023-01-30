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
T = 10000 # Total time
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

# Setting Initial Condition
IC=[]
for i in range(0, len(Xs)):
    x = Xs[i]
    res = gaussian(x = x,
                   mu = 28,
                   sigma = 0.5) # Gaussin IC
    res = np.float64(res)
    IC.append(res)
u[0][1] = IC # Populating initial condition in array u

# Fourier Number Flag
if r > 0.5:
    print("Fourier Number " + str(r) +  " > 0.5. Adjust mean step length, dx, or dt")

# Finite difference scheme
velocity_dif_signs = 0
for j in range(0,Nt-1):
    if ((((j+1)/Nt)*100 % 1 == 0)):
        print(str((j+1)/Nt * 100)," %")
    for i in range(0,Nx-1):
        c_x = (mean_sl**2) * wx[0][1][i] / w[0][1][i] / dt  # Advection
        p = dt / dx / 2

        if i == 0:
            c_xminus1 = (mean_sl**2) * wx[0][1][-1] / w[0][1][-1] / dt
            c_xminus2 = (mean_sl**2) * wx[0][1][-2] / w[0][1][-2] / dt
        elif i == 1:
            c_xminus1 = (mean_sl ** 2) * wx[0][1][0] / w[0][1][0] / dt
            c_xminus2 = (mean_sl ** 2) * wx[0][1][-1] / w[0][1][-1] / dt
        else:
            c_xminus1 = (mean_sl ** 2) * wx[0][1][i-1] / w[0][1][i-1] / dt
            c_xminus2 = (mean_sl ** 2) * wx[0][1][i-2] / w[0][1][i-2] / dt

        if i == Nt:
            c_xplus1 = (mean_sl**2) * wx[0][1][0] / w[0][1][0] / dt
            c_xplus2 = (mean_sl**2) * wx[0][1][1] / w[0][1][1] / dt
        elif i == (Nt - 1):
            c_xplus1 = (mean_sl ** 2) * wx[0][1][-1] / w[0][1][-1] / dt
            c_xplus2 = (mean_sl ** 2) * wx[0][1][0] / w[0][1][0] / dt
        else:
            c_xplus1 = (mean_sl ** 2) * wx[0][1][i+1] / w[0][1][i+1] / dt
            c_xplus2 = (mean_sl ** 2) * wx[0][1][i+2] / w[0][1][i+2] / dt

        if c_x > 0 and c_xminus1 > 0 and c_xminus2 > 0:
            #print("all velocities positive")
            if i == 0:
                u[j + 1][1][i] = r * (u[j][1][i + 1] - 2 * u[j][1][i] + u[j][1][Nx]) -p * (3* c_x * u[j][1][i] -
                                                                                           4 * c_xminus1 * u[j][1][Nx] +
                                                                                           c_xminus2 * u[j][1][Nx-1]) + u[j][1][i]
            elif i == 1:
                u[j + 1][1][i] = r * (u[j][1][i + 1] - 2 * u[j][1][i] + u[j][1][i - 1]) -p * (3 * c_x * u[j][1][i] -
                                                                                              4 * c_xminus1 * u[j][1][i - 1] +
                                                                                              c_xminus2 * u[j][1][Nx]) + u[j][1][i]
            elif i == Nx:
                u[j + 1][1][i] = r * (u[j][1][0] - 2 * u[j][1][i] + u[j][1][i - 1]) - p * (3 * c_x * u[j][1][i] -
                                                                                           4 * c_xminus1 * u[j][1][i - 1] +
                                                                                           c_xminus2 * u[j][1][i - 2]) + u[j][1][i]
            else:
                u[j + 1][1][i] = r * (u[j][1][i + 1] - 2 * u[j][1][i] + u[j][1][i - 1]) -p * (3 * c_x * u[j][1][i] -
                                                                                              4 * c_xminus1 * u[j][1][i - 1] +
                                                                                              c_xminus2 * u[j][1][i-2]) + u[j][1][i]
        elif c_x < 0 and c_xplus1 < 0 and c_xplus2 < 0:
            #print("all velocities negative")
            if i == Nx:
                u[j + 1][1][i] = u[j][1][i] - p * (-c_xplus2 * u[j][1][1] +
                                                   4 * c_xplus1 * u[j][1][0] -
                                                   3 * c_x * u[j][1][Nx]) + r * (u[j][1][0] - 2 * u[j][1][i] + u[j][1][i - 1])
            elif i == (Nx-1):
                u[j + 1][1][i] = u[j][1][i] - p * (-c_xplus2 * u[j][1][0] +
                                                   4 * c_xplus1 * u[j][1][Nx] -
                                                   3 * c_x * [j][1][Nx-1]) + r * (u[j][1][i + 1] - 2 * u[j][1][i] + u[j][1][i - 1])
            elif i == 0:
                u[j + 1][1][i] = u[j][1][i] - p * (-c_xplus2 * u[j][1][i + 2] +
                                                   4 * c_xplus1 * u[j][1][i + 1] -
                                                   3 * c_x * u[j][1][i]) + r * (u[j][1][i + 1] - 2 * u[j][1][i] + u[j][1][Nx])
            else:
                u[j + 1][1][i] = u[j][1][i] - p * (-c_xplus2 * u[j][1][i+2] +
                                                   4 * c_xplus1 * u[j][1][i + 1] -
                                                   3 * c_x * u[j][1][i]) + r * (u[j][1][i + 1] - 2 * u[j][1][i] + u[j][1][i - 1])
        else:
            c = c_x
            if c > 0:
                if i == 0:
                    u[j + 1][1][i] = r * (u[j][1][i + 1] - 2 * u[j][1][i] + u[j][1][Nx]) - p * (
                                3 * u[j][1][i] - 4 * u[j][1][Nx] + u[j][1][Nx - 1]) + u[j][1][i]
                elif i == 1:
                    u[j + 1][1][i] = r * (u[j][1][i + 1] - 2 * u[j][1][i] + u[j][1][i - 1]) - p * (
                                3 * u[j][1][i] - 4 * u[j][1][i - 1] + u[j][1][Nx]) + u[j][1][i]
                elif i == Nx:
                    u[j + 1][1][i] = r * (u[j][1][0] - 2 * u[j][1][i] + u[j][1][i - 1]) - p * (
                                3 * u[j][1][i] - 4 * u[j][1][i - 1] + u[j][1][i - 2]) + u[j][1][i]
                else:
                    u[j + 1][1][i] = r * (u[j][1][i + 1] - 2 * u[j][1][i] + u[j][1][i - 1]) - p * (
                                3 * u[j][1][i] - 4 * u[j][1][i - 1] + u[j][1][i - 2]) + u[j][1][i]
            elif c < 0:
                if i == Nx:
                    u[j + 1][1][i] = u[j][1][i] - p * (-u[j][1][1] + 4 * u[j][1][0] - 3 * u[j][1][Nx]) + r * (
                                u[j][1][0] - 2 * u[j][1][i] + u[j][1][i - 1])
                elif i == (Nx - 1):
                    u[j + 1][1][i] = u[j][1][i] - p * (-u[j][1][0] + 4 * u[j][1][Nx] - 3 * u[j][1][Nx - 1]) + r * (
                                u[j][1][i + 1] - 2 * u[j][1][i] + u[j][1][i - 1])
                elif i == 0:
                    u[j + 1][1][i] = u[j][1][i] - p * (-u[j][1][i + 2] + 4 * u[j][1][i + 1] - 3 * u[j][1][i]) + r * (
                                u[j][1][i + 1] - 2 * u[j][1][i] + u[j][1][Nx])
                else:
                    u[j + 1][1][i] = u[j][1][i] - p * (-u[j][1][i + 2] + 4 * u[j][1][i + 1] - 3 * u[j][1][i]) + r * (
                                u[j][1][i + 1] - 2 * u[j][1][i] + u[j][1][i - 1])
            else:
                if i == 0:
                    u[j + 1][1][i] = u[j][1][i] + r * (u[j][1][i + 1] - 2 * u[j][1][i] + u[j][1][Nx])
                elif i == Nx:
                    u[j + 1][1][i] = u[j][1][i] + r * (u[j][1][0] - 2 * u[j][1][i] + u[j][1][i - 1])
                else:
                    u[j + 1][1][i] = u[j][1][i] + r * (u[j][1][i + 1] - 2 * u[j][1][i] + u[j][1][i - 1])

    area = integrate(u = u[j + 1][1],
                     dx = dx,
                     x_vals = Xs)
    u[j+1][1] = u[j+1][1]/area


#Checking for conservation
for j in range(0,Nt):
    res = integrate(u = u[j][1],
              dx = dx,
              x_vals = Xs)
    print(res)

print("Number of iterations where velocities had different signs: ", str(velocity_dif_signs))
#Approximating the steady state space use

w0 = integrate(u = (w[0][1]**2),
               dx = dx,
               x_vals = Xs)
steady_state_u = np.zeros((1,2,len(Xs)))
steady_state_u[0][0] = Xs
steady_state_u[0][1] = (w[0][1]**2)/w0

# Animation
fig1, ax1 = plt.subplots()
camera = Camera(fig1)
for i in range(Nt):
    Ys= u[i][1]
    if i%500 == 0:
        if i == 0:
            ax1.plot(w[0][0], w[0][1], color = "#5a5a5a", label = "w(x)", linestyle = "dashed")
            x = ax1.plot(Xs, Ys, color="#959e19", label="u(x,t)")
            y = ax1.plot(Xs,steady_state_u[0][1], color = 'black', linestyle = 'dotted', label = "steady state u(x,t)")
        else:
            ax1.plot(w[0][0], w[0][1], color="#5a5a5a", linestyle="dashed")
            x = ax1.plot(Xs, Ys, color="#959e19")
            y = ax1.plot(Xs, steady_state_u[0][1], color='black', linestyle='dotted')
        ax1.legend(loc="upper right")

        camera.snap()
animation = camera.animate()
animation.save('AdvectionDiffusion.mp4', writer = 'ffmpeg')

# Plot
fig2 = plt.figure()
t0 = u[0][1]
t1 = u[int(0.2*Nt)][1]
t2 = u[int(0.4*Nt)][1]
t3 = u[int(0.6*Nt)][1]
t4 = u[int(0.8*Nt)][1]
t5 = u[int(Nt-1)][1]
plt.plot(Xs,w[0][1], color = "#5a5a5a", label = r"$w(x)$", linestyle = "dashed")
plt.plot(Xs,t0, label = r'$u(x,0)$', color = "#003f5c" )
plt.plot(Xs,t1, label = fr'$u(x,{int(0.2*Nt)})$', color = "#444e86")
plt.plot(Xs,t2, label = fr'$u(x,{int(0.4*Nt)})$', color = "#955196")
plt.plot(Xs,t3, label = fr'$u(x,{int(0.6*Nt)})$', color = "#dd5182")
plt.plot(Xs,t4, label = fr'$u(x,{int(0.8*Nt)})$', color = "#ff6e54")
plt.plot(Xs,t5, label =fr'$u(x,{int(Nt-1)})$', color = "#ffa600")
plt.plot(Xs,steady_state_u[0][1], color = 'black', linestyle = 'dotted', label = r"$u^{*}(x)$")
plt.legend(loc="upper right")
plt.savefig("AdvectionDiffusion.png", dpi = 350)
plt.show()



