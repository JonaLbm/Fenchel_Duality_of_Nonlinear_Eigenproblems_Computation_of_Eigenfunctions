import os
import numpy as np
from tqdm import tqdm
import pandas
from helpers import solve_p_laplace_w_newton, disc_p_lap, phi, calc_dnp


def scheme(u0, Z, tol, p, max_it, Nt, Dnp, cord, h, r):
    '''
    scheme of the inverse power method. computes an eigenvector of the p-laplacian
    Parameters
    ----------
    u0      -   initial guess for the inverse power method
    Z       -   binary matrix, which defines the domain
    tol     -   tolerance for the newton solver
    p       -   p-value for the p-Laplacian
    max_it  -   maximum number of iterations of the inverse power method
    Nt      -   number of iterations for the newton solver
    Dnp     -   scaling parameter for the mean value approximation
    cord    -   neighbourig coordinates within the mean value radius
    h       -   spatial discretization
    r       -   radius of the mean value approximation

    Returns
    -------
    u0      -   last iterate of the inverse power method
    uk_list -   list of iterates of the inverse power method
    plpls   -   list of p-Laplace values of the iterates
    cosims  -   list of cosine similarities of the iterates
    error_list  list of L^2 error of the iterates
    '''
    # keep track of the important values / metrics
    uk_list = []
    plpls = []
    cosims = []
    error_list = []
    # add for the zero-th step
    plpl = disc_p_lap(p, r, h, Z, u0, Dnp, cord)
    plpls.append(plpl)
    cosims.append(np.inner(u0.ravel(), plpl.ravel()) / np.linalg.norm(plpl.ravel(), ord=q))
    u0 = u0 / np.linalg.norm(u0.ravel(), ord=p)
    uk_list.append(u0)
    error_list.append(np.linalg.norm(
        phi(u0, p).ravel() / np.linalg.norm(phi(u0, p).ravel()) - plpl.ravel() / np.linalg.norm(plpl.ravel()), ord=p))
    # start the iteration
    for _ in tqdm(range(max_it)):
        # solve the inner problem with initial guess u0 and right hand side phi(u0, p)
        u0, newton_error = solve_p_laplace_w_newton(u0, phi(u0, p), Z, tol, p, Nt, Dnp, cord, h, r)
        # normalize the iterate
        u0 = u0/np.linalg.norm(u0.ravel(), ord=p)
        # calculate and add the important values and metrics
        uk_list.append(u0)
        plpl = disc_p_lap(p, r, h, Z, u0, Dnp, cord)
        plpls.append(plpl)
        cosims.append(np.inner(u0.ravel(), plpl.ravel())/np.linalg.norm(plpl.ravel(), ord=q))
        error_list.append(np.linalg.norm(phi(u0, p).ravel()/np.linalg.norm(phi(u0, p).ravel()) - plpl.ravel()/np.linalg.norm(plpl.ravel()), ord=p))
    return u0, uk_list, plpls, cosims, error_list


max_it = 30  # number of iteration for the inverse power method
Nt = 500  # number of maximal iterations in Newton's method
tol = 10**(-12)  # tolerance of Newton's method

p=1.77  # 1.77, 2, 5  # p value of p-Laplacian
q=p/(p-1)  # Hölder conjugate value

# Numerical Parameters
Dnp = calc_dnp(p)  # Constant of the discretization
R = 1  # Side length of the square domain
h = 0.025  # Parameter of the spatial discretization (distance between grid points)
r = 0.2  # radius of the mean value approximation

# Calculate coordinates neihbouring within the mean value radius
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
xx = np.arange(-R - r, R + r, h)  # added padding due to the mean value radius at the boundary
yy = np.arange(-R - r, R + r, h)
X, Y = np.meshgrid(xx, yy)

# Define the L-shaped domain.
Z = np.zeros_like(X, dtype=bool)
Z[(Y <= 0) & (X <= 1.0)] = True  # Bottom part of L
Z[(Y <= 1.0) & (X <= 0)] = True  # Left part of L
Z[(Y < -1)] = False
Z[(X < -1)] = False

# # square
#Z = np.zeros_like(X, dtype=bool)
#Z[(np.abs(Y) <= 1.0) & (np.abs(X) <= 1.0)] = True

# Initial solution guess
U0 = -(1 - 2*np.abs(X+0.5)) * (1- 2*np.abs(Y+0.5)) * (1-np.abs(X)) * (1-np.abs(Y))  # ex 1 on L-shaped domain
#U0 = 100 * ((X - 1) * (X + 1) * (Y - 1) * (Y + 1)) * (0.0625 - X ** 2 - Y ** 2)  # ex 2 on square
U0 = U0 * (Z == 1)
U = U0.copy()
U = U / np.linalg.norm(U.flatten(), ord=p)

# calculate the solution with the inverse power method
uk, uk_list, plpls, cosims, e0 = scheme(U, Z, tol, p, max_it, Nt, Dnp, cord, h, r)

# save the study as dataframe
results_frame = pandas.DataFrame()
temp_df = pandas.DataFrame([{'p': p, 'u0': U0, 'iterates': uk_list, 'r': r, 'xy': (X, Y),
                             'plpl': plpls, 'COSIM': cosims, 'error': e0}])
results_frame = pandas.concat([results_frame, temp_df], ignore_index=True)

# save results as pickle file
os.makedirs('study_results/', exist_ok=True)
results_frame.to_pickle('study_results/ipm_ex1_p_1,77.pk')

