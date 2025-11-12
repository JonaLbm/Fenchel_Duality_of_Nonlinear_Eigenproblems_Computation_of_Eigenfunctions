import os
import numpy as np
from tqdm import tqdm
import pandas
from helpers import phi, disc_p_lap, flow_step_2d, calc_dnp
np.seterr(all="ignore")

def scheme(u0, p, max_it):
    uk_list = [u0]
    plpl = disc_p_lap(p, r, h, Z, u0, Dnp, cord)
    plpls = [plpl]
    nu = np.linalg.norm(u0.ravel(), ord=p)  # norm of the current iterate uk
    nplpl = np.linalg.norm(plpl.ravel(), ord=q)  # norm of the p-Laplace of uk
    cosim = np.inner(plpl.ravel(), u0.ravel())/(nu*nplpl)
    cosims = [cosim]
    lambda_vals = [cosim * nplpl / (nu ** (p - 1))]
    error = np.linalg.norm(phi(u0, p).ravel() / (nu ** (p - 1)) - plpl.ravel() / nplpl)
    error_list = [error]
    tau = 1 - cosim
    u0 = u0 / np.linalg.norm(u0.ravel(), ord=p)
    for _ in tqdm(range(max_it)):
        u0, plpl, cosim, lambda_val, error, tau = flow_step_2d(u0, p, tau, r, h, Z, Dnp, cord, tol, Nt)
        u0 = u0/np.linalg.norm(u0.ravel(), ord=p)
        uk_list.append(u0)
        plpls.append(plpl)
        cosims.append(cosim)
        lambda_vals.append(lambda_val)
        error_list.append(error)
    return u0, uk_list, plpls, cosims, lambda_vals, error_list

max_it = 10  # number of iteration for the inverse power method
Nt = 500  # number of maximal iterations in Newton's method
tol = 10**(-12)  # tolerance of Newton's method

p=3  # p value of p-Laplacian
q=p/(p-1)  # Hölder conjugate value

# Numerical Parameters
Dnp = calc_dnp(p)  # Constant of the discretization
R = 1  # Radius of the ball
h = 0.02  # Parameter of the spatial discretization (distance between grid points)
r = 0.02**0.5  # radius of the mean value approximation

# Calculate coordinates
I_max = int(np.ceil(r/h))
Q = (r/h)**2
cord = []
for i in range(-I_max, I_max+1):
    for j in range(-I_max, I_max+1):
        if i**2 + j**2 < Q:  # L2 radius
        #if np.abs(i) + np.abs(j) < Q:  # L1 radius
            cord.append((i, j))
cord = np.array(cord)

# Create domain
xx = np.arange(-R-r, R+r, h)
yy = np.arange(-R-r, R+r, h)
X, Y = np.meshgrid(xx, yy)

# Define the L-shaped domain.
#Z = np.zeros_like(X, dtype=bool)
#Z[(Y <= 0) & (X <= 1.0)] = True  # Bottom part of L
#Z[(Y <= 1.0) & (X <= 0)] = True  # Left part of L
#Z[(Y < -1)] = False
#Z[(X < -1)] = False

# # square
Z = np.zeros_like(X, dtype=bool)
Z[(np.abs(Y) <= 1.0) & (np.abs(X) <= 1.0)] = True

# Initial solution guess
#U0 = -(1 - 2*np.abs(X+0.5)) * (1- 2*np.abs(Y+0.5)) * (1-np.abs(X)) * (1-np.abs(Y))  # ex 1 on L-shaped domain
U0 = 100 * ((X - 1) * (X + 1) * (Y - 1) * (Y + 1)) * (0.0625 - X ** 2 - Y ** 2)  # ex 2 on square
U0 = U0 * (Z == 1)
U = U0.copy()
U = U / np.linalg.norm(U.flatten(), ord=p)

# calculate the iterative scheme
uk, uk_list, plpls, cosims, lambda_vals, e0 = scheme(U, p, max_it)

# safe the study as dataframe
results_frame = pandas.DataFrame()
temp_df = pandas.DataFrame([{'p': p, 'u0': U0, 'iterates': uk_list, 'r': r, 'x': X,
                                 'plpl': plpls, 'COSIM': cosims, 'lambda_values': lambda_vals, 'error': e0}])
results_frame = pandas.concat([results_frame, temp_df], ignore_index=True)

# save results as pickle file
os.makedirs('study_results/', exist_ok=True)
results_frame.to_pickle('study_results/cosim_flow_ex2.pk')

