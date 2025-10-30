from fenics import *
from dolfin import fem, io, mesh, plot
import sys
from scipy import optimize as opt
import numpy as np
import pandas
import logging
#import subprocess
#subprocess.run("export PKG_CONFIG_PATH=/Users/ih09odap/miniconda3/envs/fenics-env-clean/lib/pkgconfig:$PKG_CONFIG_PATH", shell = True)
logging.getLogger('FFC').setLevel(logging.WARNING)
import matplotlib
import matplotlib.pyplot as plt

# info(NonlinearVariationalSolver.default_parameters(), 1)

plt.rcParams["font.size"] = 10

M = sys.float_info.max
# mesh generation - changed the 'stamp' now closer to del teso lindgren
mesh = RectangleMesh(Point(-1.0, -1.0), Point(1.0, 1.0), 100, 100, 'crossed')  # more connections with crossed
# number of iterations
it = 200

#mesh = UnitSquareMesh(100, 100)
# finite-element-space
V = FunctionSpace(mesh, 'P', 1)
N = V.dim()
dx = Measure('dx', domain=mesh)
# boundary condition
u_D = Expression('0', degree=0)


def boundary(x, on_boundary):
    return on_boundary


bc = DirichletBC(V, u_D, boundary)
eps = 1e-12
set_log_level(LogLevel.ERROR)

# right_side

Omega = "Square"

uprev = Expression(
    '100 * (x[0]+1) * (x[1]+1) * (x[0] - 1) * (x[1] - 1) * (0.0625 - (x[0])*(x[0]) - (x[1])*(x[1]))',
    degree=6)
marker = "radial NLVS"

uprev = project(uprev, V)
# get_func = open("Square, p = 2, radial NLVS, it = 100, resulting u.txt", "r")
# array = np.loadtxt(get_func)
# for i in range(len(array)):
#	uprev.vector()[i] = array[i]
# start = 100
start = 0

u0 = uprev.copy(deepcopy=True)


# defining positive and negative parts of a function
def uplus(u):
    return conditional(u >= 0, u, 0)


def uminus(u):
    return conditional(u <= 0, -u, 0)


for p in [3]:#[1.7, 2, 3]:
    alpha_star = 2 ** (-p)
    increase = False  # whether |u_{k+1} - u_k| was increasing at the previous iteration
    uprev = u0.copy(deepcopy=True)
    s = "{3}, p = {0}, {2}, it = {1}".format(p, it, marker, Omega)
    s1 = "{3}, p = {0}, {2}, it = ".format(p, it, marker, Omega)
    #File = open(s + ".txt", 'w')
    #File.write("NEW p:\t" + str(p) + str("\n"))


    # norms, functionals etc.
    def Lp(u):
        return assemble((abs(u) ** p) * dx)

    def Lq(zeta):
        q = p/(p-1)
        return assemble((abs(zeta) ** q) * dx)

    def Wp(u):
        return assemble((grad(u)[0] ** 2 + grad(u)[1] ** 2) ** (p / 2) * dx)

    def R(u):
        return Wp(u) / Lp(u)

    def R_star(zeta, u):
        return Wp(u)/Lq(zeta)

    def Rplus(u):
        q = Lp(uplus(u))
        try:
            return Wp(uplus(u)) / q
        except:
            return M

    def Rminus(u):
        q = Lp(uminus(u))
        try:
            return Wp(uminus(u)) / q
        except:
            return M

    def cosim(u, zeta):
        return assemble(dot(zeta, u)*dx)/(Lq(zeta)**((p-1)/p) * Lp(u)**(1/p))


    uprev.vector()[:] /= (Lp(uprev)) ** (1 / p)

    #File.write("\tR[u]:\t" + str(R(uprev)) + "\n")
    Wps = []  # W-norms of |u_{k+1} - u_k|
    Lps = []  # L-norms of |u_{k+1} - u_k|
    Rs = []  # Rayleigh quotients
    cs = []  # Cosine similarity
    R_stars = []  # dual Rayleigh quotient
    U_s = [uprev.vector().get_local()]  # shape n*2*(n+1)+1
    U_raw = [uprev]
    zeta_s = []
    Rs.append(R(uprev))
    print('p = ', p)
    for i in range(start, it):
        print("\t iteration: " + str(i))
        #File.write("\nIteration:\t" + str(i) + '\n')

        def phi_alpha(a):  # sovling the boundary value problem for alpha
            b = (1-a)#(1 - a ** p) ** (1 / p)
            u_ = uprev.copy(deepcopy=True)
            u = TrialFunction(V)
            v = TestFunction(V)
            F = (grad(u)[0] ** 2 + grad(u)[1] ** 2) ** ((p - 2) / 2) * dot(grad(u), grad(v)) * dx - a * uplus(
                uprev) ** (p - 1) * v * dx + b * uminus(uprev) ** (
                            p - 1) * v * dx  # / (grad(u)[0]**2 + grad(u)[1]**2)**0.5
            F = action(F, u_)
            du = TrialFunction(V)
            J = (p - 2) * (grad(u_)[0] ** 2 + grad(u_)[1] ** 2) ** ((p - 4) / 2) * dot(grad(u_), grad(v)) * dot(
                grad(u_), grad(du)) * dx  # / (grad(u_)[0]**2 + grad(u_)[1]**2)**1.5
            J += (grad(u_)[0] ** 2 + grad(u_)[1] ** 2) ** ((p - 2) / 2) * dot(grad(v), grad(
                du)) * dx  # / (grad(u_)[0]**2 + grad(u_)[1]**2)**0.5
            problem = NonlinearVariationalProblem(F, u_, bc, J)
            # Solver Konfigurationen und Toleranzen einstellen
            solver_parameters = {"nonlinear_solver": "newton", "newton_solver": {}}
            solver_parameters["newton_solver"]["absolute_tolerance"] = 1E-12
            #solver_parameters["newton_solver"]["relative_tolerance"] = 1E-10
            solver_parameters["newton_solver"]["maximum_iterations"] = 100
            solver_parameters["newton_solver"]["convergence_criterion"] = "residual"

            solver = NonlinearVariationalSolver(problem)
            solver.parameters.update(solver_parameters)

            solver.solve()
            u = u_.copy(deepcopy=True)
            #u.vector()[:] /= (Lp(u)) ** (1 / p)
            return u

        def rho(a):  # defining an equation for alpha to be solved
            u = phi_alpha(a)
            plu = Rplus(u)
            minu = Rminus(u)
            return plu - minu

        optres = opt.root_scalar(rho, method="ridder", bracket=(0, 1), x0=alpha_star)  # solving the equation for alpha
        alpha = optres.root
        print('alpha: ', alpha)
        beta = (1-alpha)#(1 - alpha ** p) ** (1 / p)
        uprev_copy = uprev.copy(deepcopy=True)
        zeta = project(alpha * uplus(uprev_copy) ** (p-1) - beta * uminus(uprev_copy) ** (p-1), V)
        #File.write("\tOpt res:\t" + str(alpha) + ", " + str(rho(alpha)) + "\n")
        unew = phi_alpha(alpha)
        # Testfunktion
        v = TestFunction(V)

        # Berechne den p-Laplace-Operator
        grad_u = nabla_grad(unew)
        p_laplace = -div((inner(grad_u, grad_u) ** ((p - 2) / 2.0)) * grad_u)

        # Projektiere den Ausdruck, um das Ergebnis zu erhalten
        p_laplace_proj = project(p_laplace, V)
        error = assemble(abs(zeta - p_laplace_proj)**2*dx)
        #print("error = ", error)
        cs.append(cosim(unew.copy(deepcopy=True), zeta))
        U_s.append(unew.vector().get_local())
        U_raw.append(unew.copy(deepcopy=True))
        zeta_s.append(zeta.vector().get_local())
        R_stars.append(R_star(zeta, unew.copy(deepcopy=True)))
        unew.vector()[:] /= (Lp(unew)) ** (1 / p)

        #File.write("\t" + str(Rplus(unew)) + ", " + str(Rminus(unew)) + "\n")
        d = uprev - unew
        Wps.append(Wp(d) ** (1 / p))
        #File.write("\tDev W_0^{1,p}-norm:\t" + str(Wps[-1]) + "\n")
        Lps.append(Lp(d) ** (1 / p))
        #File.write("\tDev L^p-norm:\t" + str(Lps[-1]) + "\n")
        Rs.append(R(unew))
        #File.write("\tR[u]:\t" + str(Rs[-1]) + "\n")

        if (i - start > 1 and Lps[-1] > Lps[-2] and not (increase)) or i == it - 1:
            increase = True
            print("\tLocal minimum: " + str(i - 1))
            dg_curr = Rs[-1]** (-1 / p) - R_stars[-1] ** ((p - 1) / p)
            print('duality gap:', dg_curr, 'cosim:', cs[-1])
            gr = plot(uprev, mode='contour', linewidths=0.5, colors='k')  # colors = 'black' cmap='brg'
            gr = plot(uprev, cmap='jet')  # colors = 'black' cmap='brg'
            plt.colorbar(gr)
            plt.tight_layout()
            plt.savefig(s1 + str(i) + ", func.png")
            plt.clf()
            plt.close()
            #save_func = open(s1 + str(i + 1) + ", resulting u.txt", "w")
            #uprev.vector()[:].tofile(save_func, sep=" ")
            #save_func.close()
        uprev = unew.copy(deepcopy=True)
        if increase == True and Lps[-1] <= Lps[-2]:
            increase = False
    dg = [r ** (-1 / p) - r_star ** ((p - 1) / p) for r, r_star in zip(Rs[1:], R_stars)]
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True)
    ax1.plot(Rs[1:], label=r'$R[u_k]$', color='magenta')
    ax1.legend()
    ax2.plot(R_stars, label=r'$R_*(\zeta_{k})$', color='green')
    ax2.legend()
    ax3.plot(cs, label=r'cosim($u_{k+1},\zeta_k$)', color='magenta')
    ax3.set_yscale("logit")
    ax3.legend()
    ax4.plot(dg, label=r'duality gap', color='magenta')
    ax4.set_yscale("log")
    ax4.set_ylim(1e-12, 1)
    ax4.set(xlabel=r'$k$')
    ax4.legend()
    plt.savefig(s + ", Rs.png")
    plt.close()
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.semilogy(Wps, label=r'$||\nabla(\tilde u_{k+1} - \tilde u_{k})||_{p}$', color='red')
    ax1.legend()
    ax2.semilogy(Lps, label=r'$||\tilde u_{k+1} - \tilde u_{k}||_{p}$', color='green')
    ax2.legend()
    plt.savefig(s + ", norms.png")
    plt.close()


data_dict = {'U': U_s, 'zeta': zeta_s, 'cosim': cs, 'RQ': Rs, 'dual_Rayleigh_Quotient': R_stars}
data_frame = pandas.DataFrame.from_dict(data_dict, orient='index')
data_frame = data_frame.transpose()
data_frame.to_pickle('inv_pm_p_3_higher_ex3_fenics_100_it_200_crossed.pk')