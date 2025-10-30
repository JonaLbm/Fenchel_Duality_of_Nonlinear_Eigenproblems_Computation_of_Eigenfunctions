import numpy as np
import pandas
from scipy import optimize as opt
from helpers import u_plus, u_minus, solve_p_laplace_w_newton, disc_p_lap, calc_dnp, Rplus, Rminus

np.random.seed(482)

def scheme(U, Z, tol, p, Nt, max_it):
    U_list = [U]
    V_list = []
    alpha_list = []
    alpha_star = 0.5  # 2**(-p)
    alpha = alpha_star
    for it in range(max_it):
        def phi_alpha(a):
            b = (1 - a)  # (1 - a**p)**(1/p)
            u_p = u_plus(U)
            u_p = u_p / np.linalg.norm(u_p.flatten(), ord=p)
            u_m = u_minus(U)
            u_m = u_m / np.linalg.norm(u_m.flatten(), ord=p)
            return solve_p_laplace_w_newton(U, a * u_p ** (p - 1) - b * u_m ** (p - 1),
                                            Z, tol, p, Nt, Dnp, cord, h, r)  # solve the p-laplacian

        def alpha_u(a):
            b = (1 - a)  # (1 - a**p)**(1/p)
            u_p = u_plus(U)
            u_p = u_p / np.linalg.norm(u_p.flatten(), ord=p)
            u_m = u_minus(U)
            u_m = u_m / np.linalg.norm(u_m.flatten(), ord=p)
            return a * u_p ** (p - 1) - b * u_m ** (p - 1)  # return alpha_U

        def balance_alpha(alpha):
            u, error = phi_alpha(alpha)
            if error is None:
                return None
            plu = Rplus(u, p)
            minu = Rminus(u, p)
            return plu - minu

        try:
            bracket = (0, 1)
            optres = opt.root_scalar(balance_alpha, method="ridder", bracket=bracket, x0=alpha_star,
                                     maxiter=150)  # solving the equation for alpha
            alpha = optres.root
            print('alpha: ', alpha)
            U, error = phi_alpha(alpha)
        except:
            if np.abs(alpha_list[-1] - 0.5) < 0.05:
                alpha = 0.5  # optres.root
                U, error = phi_alpha(alpha)
            elif it > 3:
                alpha = np.mean(alpha_list[2:])
                print('guessed alpha: ', alpha)
                U, error = phi_alpha(alpha)
            else:
                break
        U = U / np.linalg.norm(U.flatten(), ord=p)
        # keeping track of the progress
        alpha_list.append(alpha)
        V_list.append(alpha_u(alpha))  # zeta_alpha
        U_list.append(U)
    return U_list, V_list, alpha_list

max_it = 30  # number of iteration for the inverse power method
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
        if i**2 + j**2 < Q:
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

U_list, V_list, alpha_list = scheme(U0, Z, tol, p, Nt, max_it)
# collecting all the metrics
J = lambda r: np.abs(r)**(p-1) * np.sign(r)
J_u_list = [J(u) for u in U_list]  # dual mapping of the iterates
plpl_list = [-disc_p_lap(p, r, h, Z, u, Dnp, cord) for u in U_list]  # p laplacian of the iterates
diff_list = [np.nan if np.isnan(ju + plpl).any() else ju/np.linalg.norm(ju.ravel(), ord=q) - plpl/np.linalg.norm(plpl.ravel(), ord=q) for ju, plpl in zip(J_u_list, plpl_list)]  # difference of normed plpl and normed dual mapping
max_error_list = [np.max(np.abs(diff)) for diff in diff_list]  # maximal pointwise error of the difference
l2_error_list = [np.nan if np.isnan(diff).any() else np.linalg.norm(diff.ravel()) for diff in diff_list]  # l2 error
cosim_list = [np.nan if np.isnan(u + plpl).any() else np.inner(u.ravel(), plpl.ravel())/(np.linalg.norm(u.ravel(), ord=p)*np.linalg.norm(plpl.ravel(), ord=q)) for u, plpl in zip(U_list, plpl_list)]  # cosinus similarity of plpl and dual norm
dRQ_list = [np.nan if np.isnan(u + v).any() else np.inner(u.ravel(), v.ravel())/np.linalg.norm(v.ravel(), ord=q)**q for u, v in zip(U_list[:-1], V_list)]  # dual rayleigh quotient of the dual iterates calculated via the euler identity

# Save the results
important_quantities = [X, Y, Z, p, R, r, h, U0]
data_dict = {'U': U_list, 'V': V_list, 'dual_mapping': J_u_list, 'p_laplace': plpl_list, 'difference': diff_list,
            'max_error': max_error_list, 'l2':l2_error_list, 'cosim': cosim_list, 'dual_Rayleigh_Quotient': dRQ_list, 'etc': important_quantities}
data_frame = pandas.DataFrame.from_dict(data_dict, orient='index')
data_frame = data_frame.transpose()
#data_frame.to_pickle('inv_pm_p_3_higher_ex3_delteso_100_it_50.pk')
data_frame.to_pickle('results_2d_hipm_ex3.pk')

