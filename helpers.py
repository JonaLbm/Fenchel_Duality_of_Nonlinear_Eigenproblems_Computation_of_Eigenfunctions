import numpy as np
from scipy.sparse.linalg import spsolve, lsqr
from scipy.sparse import lil_matrix, diags
from scipy.special import gamma


def solve_p_laplace_w_newton(U, F, Z, tol, p, max_iter, Dnp, cord, h, r):
    '''
    Newton solver of the p-Laplace problem with Dirichlet boundary conditions.
    - p-Laplacian(u) = F

    Parameters
    ----------
    U           -   initial guess for the Newton method
    F           -   right hand side of the p-Dirichlet problem
    Z           -   binary matrix, which defines the domain
    tol         -   tolerance for the newton solver
    p           -   p-value for the p-Laplacian
    max_it      -   maximum number of iterations for the newton solver
    Dnp         -   scaling parameter for the mean value approximation
    cord        -   neighbouring coordinates within the mean value radius
    h           -   spatial discretization
    r           -   radius of the mean value approximation

    Returns
    -------
    best_U      -   the solution of the p-Dirichlet problem (best approximation by the Newton method)
    best_error  -   the approximation error of the solution
    '''
    # Convert to 1D for easier matrix operations
    N, M = Z.shape
    #U = U/np.linalg.norm(U.ravel(), ord=p)
    PL = -disc_p_lap(p, r, h, Z, U, Dnp, cord)  # with - sign: positive laplacian
    PL1D = PL.ravel()
    F1D = F.ravel()
    # calculate and track error
    best_error = np.linalg.norm(F1D + PL1D)  # error of the p-Dirichlet problem
    best_U = np.copy(U)
    # Perform Newton Update scheme until convergence or maximum iterations
    for iteration in range(max_iter):  # tqdm(range(max_iter)):  #
        U1D = U.ravel()
        # Compute the Jacobian matrix
        J_matrix = -jac_disc_p_lap(p, r, h, Z, U, Dnp, cord)  # with - sign: jacobian of negative laplacian
        # right hand side of newton LSE
        rhs = -PL1D - F1D  #
        # Solve newton LSE for the update
        try:
            dU = spsolve(J_matrix, rhs)
        except:
            dU = lsqr(J_matrix, rhs)[0]
            #print('lsqr newton step')
        # Update the solution
        U_new = U1D - dU  # make newton step
        U = U_new.reshape((N, M))
        #U = U * (Z == 1)
        #U = U/np.linalg.norm(U.ravel(), ord=p)  # normalizing the iterates for stability and faster convergence
        # due to the homogenetiy of the p-Laplace
        # we do not need the correct 'length', because we will normalize afterwards in the IPM anyway
        # we therefore look at the 'normalized error' for stopping criterion

        # Compute the current error as stopping criteria
        PL = -disc_p_lap(p, r, h, Z, U, Dnp, cord)   # with - sign: positive laplacian
        PL1D = PL.ravel()
        error = np.linalg.norm(F1D + PL1D)
        if error < best_error:
            best_U = np.copy(U)
            best_error = error
        if error < tol:
            break
    return best_U, best_error


def phi(r, p):
    '''
    computes the duality mapping from L^p to L^q

    Parameters
    ----------
    r       -   function that needs to be mapped
    p       -   p value

    Returns
    -------
    value   -   the duality map: |r|^{p-2}r

    '''
    if p==2:
        value = r
    else:
        value =  np.abs(r) ** (p - 1) * np.sign(r)
    return value


def dphi(r, p):
    '''
    derivative of the duality mapping from L^p to L^q
    we are taking care of the non-differentiable case p<2 by adding a small epsilon value

    Parameters
    ----------
    r       -   function that needs to be mapped
    p       -   p value

    Returns
    value   -   the derivative of duality map: (p-1)|r|^{p-2}
    -------

    '''
    epsilon = 1e-12
    if p<2:
        value = (p - 1) * (np.abs(r) + epsilon) ** (p - 2) #- np.sign(r) * (p-1) * epsilon**(p-2)
    elif p==2:
        value =  np.ones_like(r)
    else:
        value = (p - 1) * np.abs(r) ** (p - 2)
    return value


def disc_p_lap(p, r, h, Z, U, Dnp, cord):
    '''
    calculates the negative discrete p-Laplacian by approximating the p-Laplacian with the del Teso and Lindgren
    mean value approximation

    Parameters
    ----------
    p           -   p value
    r           -   radius of the mean value approximation
    h           -   spatial discretization
    Z           -   binary matrix, which defines the domain
    U           -   discretized function on which the p-Laplacian is applied to
    Dnp         -   scaling parameter for the mean value approximation
    cord        -   neighbouring coordinates within the mean value radius

    Returns
    -------
    negative p-Laplacian of U
    '''
    PL = np.zeros_like(Z, dtype=float)
    for i in range(Z.shape[1]):
        for j in range(Z.shape[0]):
            if Z[j, i] == 1:
                for k in range(cord.shape[0]):
                    x_offset = int(cord[k, 0])
                    y_offset = int(cord[k, 1])
                    U_neighbor = U[j + y_offset, i + x_offset]
                    U_current = U[j, i]
                    PL[j, i] += phi(U_neighbor - U_current, p)
    PL *= h**2 / (np.pi * r**2 * Dnp * r**p)
    # PL = PL #* (Z==1)
    return -PL


def jac_disc_p_lap(p, r, h, Z, U, Dnp, cord):
    '''
    calculates the Jacobian of the discrete approximation of the p-Laplacian of U -- not of negative p-Laplacian!

    Parameters
    ----------
    p       -   p value
    r       -   radius of the mean value approximation
    h       -   spatial discretization
    Z       -   binary matrix, which defines the domain
    U       -   discretized function on which the Jacobian of the discrete p-Laplacian is evaluated
    Dnp     -   scaling parameter for the mean value approximation
    cord    -    neighbouring coordinates within the mean value radius

    Returns
    -------
    a sparse Matrix which represents the Jacobian of the discrete p-Laplacian of U
    '''
    # calculates the jacobian of p-laplacian -- not of negative p-laplacian
    N, M = Z.shape
    J_matrix = lil_matrix((N * M, N * M))  # lil_matrix um die Matrix in COO-Format aufzubauen

    scale_factor = h**2 / (np.pi * r**2 * Dnp * r**p)

    for i in range(M):
        for j in range(N):
            idx = i * N + j  # 1D Index
            if Z[j, i] == 1:
                diag_contrib = 0  # Summe der Diagonalelemente für den Punkt (j, i)
                for k in range(cord.shape[0]):
                    x_offset = int(cord[k, 0])
                    y_offset = int(cord[k, 1])
                    neighbor_x = i + x_offset
                    neighbor_y = j + y_offset

                    neighbor_idx = neighbor_x * N + neighbor_y  # 1D Index des Nachbarn
                    r_val = U[neighbor_y, neighbor_x] - U[j, i]
                    J_value = dphi(r_val, p) * scale_factor
                    J_matrix[idx, neighbor_idx] = J_value
                    diag_contrib -= J_value

                J_matrix[idx, idx] = diag_contrib # Diagonaleintrag beachten
            else:
                J_matrix[idx, idx] = 1/scale_factor

    return J_matrix.tocsr()


def calc_dnp(p):
    '''
    Calculates the scaling parameter of the del Teso and Lindgren mean value approximation

    Parameters
    ----------
    p       -   p value

    Returns
    -------
    Dnp     -   scaling parameter of the mean value approximation
    '''
    if p % 2 == 0:
        P1 = np.arange(p-1, 0, -2)
        P2 = np.arange(p, 1, -2)
        Dnp = 1/(2 + p) * np.prod(P1) / np.prod(P2)
    elif p % 1 == 0:
        P1 = np.arange(p-1, 1, -2)
        P2 = np.arange(p, 2, -2)
        Dnp = 2/((2 + p) * np.pi) * np.prod(P1) / np.prod(P2)
    else:
        d=2
        Dnp = 1/np.sqrt(np.pi) * (p-1)/(d+p) * gamma((p-1)/2)/gamma((d+p)/2)
    return Dnp


def solve_flow_step_w_newton(U, F, Z, tol, p, max_iter, uold, weight, Dnp, cord, h, r):
    '''
    Solves the semi implicit flow step involving the duality map and the p-Laplacian with Newton's method
    Phi(u-u_old) + weight * p-Laplacian(u) = F

    Parameters
    ----------
    U           -   initial guess for Newton's method
    F           -   right hand side of the problem
    Z           -   binary matrix, which defines the domain
    tol         -   tolerance parameter for Newton's method
    p           -   p-value for the p-Laplacian
    max_iter    -   maximum number of iterations for Newton's method
    uold        -   previous iterate
    weight      -   weight for p-Laplacian
    Dnp         -   scaling parameter for the mean value approximation
    cord        -   neighbouring coordinates within the mean value radius
    h           -   spatial discretization
    r           -   radius of the mean value approximation

    Returns
    -------
    best_U      -   the next iterate of the flow (best approximation by the Newton method)
    best_error  -   the approximation error of the solution
    '''
    # Convert to 1D for easier matrix operations
    N, M = Z.shape
    F1D = F.ravel()
    uold1D = uold.ravel()
    error = tol + 1
    PL = -disc_p_lap(p, r, h, Z, U, Dnp, cord)  # with - sign: positive laplacian
    PL1D = PL.ravel()
    U1D = U.ravel()
    best_error = np.linalg.norm(-F1D + phi((U1D - uold1D), p) + PL1D*weight)
    best_U = np.copy(U)
    # Perform Newton Update scheme until convergence or maximum iterations
    for iteration in range(max_iter):  # tqdm(range(max_iter)):  #
        U1D = U.ravel()
        # Compute the Jacobian matrix
        J_matrix = jac_disc_p_lap(p, r, h, Z, U, Dnp, cord) * weight  # with - sign: jacobian of negative laplacian
        J_matrix += diags(dphi((U1D - uold1D), p))
        # right hand side of newton LSE
        rhs = PL1D * weight - F1D + phi((U1D - uold1D), p)
        # Solve newton LSE for the update
        try:
            dU = np.linalg.solve(J_matrix, rhs)
        except:
            #dU = lsqr(J_matrix, rhs)[0]
            #print('lsqr newton step')
            #print('could not solve newton step')
            #return np.ones_like(uold), np.nan
            break
        # Update the solution
        U_new = U1D - dU  # make newton step
        U = U_new.reshape((N, M))
        U = U * (Z == 1)
        U1D = U.ravel()

        # Compute the current absolute error as stopping criteria
        PL = -disc_p_lap(p, r, h, Z, U, Dnp, cord)   # with - sign: positive laplacian
        PL1D = PL.ravel()
        lhs = PL1D * weight + phi((U1D - uold1D), p)
        error = np.linalg.norm(lhs - F1D)
        #print('error', error)
        if error < best_error:
            best_error = error
            best_U = np.copy(U)
        if error < tol:
            #print('newton method is converged')
            break
    #if np.isnan(error):
    #    return np.ones_like(uold), np.nan
    #print('best newton error:', best_error)
    return best_U, best_error


def dNp(u, p):
    '''
    Calculates the derivative of the L^p-norm with respect to u
    Phi(u, p)*||u||_p^{1-p}

    Parameters
    ----------
    u       -   discretized function at which the derivative is evaluated
    p       -   p-value

    Returns
    -------
    the derivative of the L^p-norm evaluated at u
    '''
    u_shape = np.shape(u)
    u = u.ravel()
    dNp = np.linalg.norm(u, ord=p)**(1-p) * phi(u, p)
    return dNp.reshape(u_shape)


def dNplpl(u, p, r, h, Z, Dnp, cord):
    '''
    Calculates the derivative of the L^q-norm of the p-Laplacian of u with respect to u
    Jacobian(-p-Laplacian(u)) * Phi(-p-Laplacian(u), q)*||p-Laplacian(u)||_q^{1-q}

    Parameters
    ----------
    u       -   discretized function at which the derivative is evaluated
    p       -   p-value
    r       -   radius of the mean value approximation
    h       -   spatial discretization
    Z       -   binary matrix, which defines the domain
    Dnp     -   scaling parameter for the mean value approximation
    cord    -   neighbouring coordinates within the mean value radius

    Returns
    -------
    the derivative of the L^q-norm of the p-Laplacian of u with respect to u
    '''
    q = p/(p-1)
    plpl = disc_p_lap(p, r, h, Z, u, Dnp, cord)
    g_plpl = -jac_disc_p_lap(p, r, h, Z, u, Dnp, cord).toarray()
    dNplpl =  np.linalg.norm(plpl.ravel(), ord=q)**(1-q) * np.matmul(g_plpl, phi(plpl.ravel(), q))
    return dNplpl.reshape(np.shape(u)) #* (Z==1)


def obj_function(u_it, p, r, h, Z, Dnp, cord):
    '''
    Evaluated the objective function 1-cosim(u, -p-Laplacian(u))

    Parameters
    ----------
    u_it    -   discretized function at which the objective function is evaluated
    p       -   p-value
    r       -   radius of the mean value approximation
    h       -   spatial discretization
    Z       -   binary matrix, which defines the domain
    Dnp     -   scaling parameter for the mean value approximation
    cord    -   neighbouring coordinates within the mean value radius

    Returns
    -------
    1-cosim(u, -p-Laplacian(u))
    '''
    q = p/(p-1)
    plpl_it = disc_p_lap(p, r, h, Z, u_it, Dnp, cord)  # negative p-Laplace
    nu_it = np.linalg.norm(u_it.ravel(), ord=p)  # norm of U
    nplpl_it = np.linalg.norm(plpl_it.ravel(), ord=q)  # norm of p-Laplace
    return 1-np.inner(plpl_it.ravel(), u_it.ravel())/(nu_it*nplpl_it)  # return 1- cosim


def next_iterate_w_flow(uk, p, tau, rhs, weight, plpl, Z, tol, Nt, Dnp, cord, h, r, explicit=False):
    '''
    For a given tau it computes an initial guess and then calls the solver to solve for the next iterate of the flow

    Parameters
    ----------
    uk      -   discretized function of the current iterate
    p       -   p-value
    tau     -   step size, time discretization of the flow
    rhs     -   right hand side of the semi-implicit update step
    weight  -   weight for the p-Laplacian
    plpl    -   p-Laplacian of u
    Z       -   binary matrix, which defines the domain
    tol     -   tolerance for the Newton solver
    Nt      -   maximum number of iterations for the Newton solver
    Dnp     -   scaling parameter for the mean value approximation
    cord    -   neighbouring coordinates within the mean value radius
    h       -   spatial discretization
    r       -   radius of the mean value approximation
    explizit    if we want to make the step explicitly instead of semi-implicit

    Returns
    -------
    next iterate of the iterations scheme for a given tau

    '''
    q = p/(p-1)
    rhs *= tau  # ** (p - 1)
    weight *= tau  # ** (p - 1)
    initial_guess = phi(rhs + weight * plpl, q) + uk  # explicit euler as initial guess
    if explicit is True:
        return initial_guess
    u_n, newton_error = solve_flow_step_w_newton(initial_guess, rhs, Z, tol, p, Nt, uk, weight, Dnp, cord, h, r)

    if np.isnan(newton_error):
        # print('retrying with different initial guess')
        initial_guess = phi(rhs, q)
        u_n, newton_error = solve_flow_step_w_newton(initial_guess, rhs, Z, tol, p, Nt, uk, weight, Dnp, cord, h, r)
        if np.isnan(newton_error):
            # print('retrying with different initial guess')
            initial_guess = uk
            u_n, newton_error = solve_flow_step_w_newton(initial_guess, rhs, Z, tol, p, Nt, uk, weight, Dnp, cord, h, r)
            if np.isnan(newton_error):
                # print('retrying with smaller step-size')
                return None
    return u_n


def currrent_difference(uk, obj_uk, p, tau, rhs, weight, plpl, Z, tol, Nt, Dnp, cord, h, r, explicit=False):
    '''
    Calculates the difference between the objective function value of the last iterate and a candidate for the next

    Parameters
    ----------
    uk          -   discretized function at which the objective function is evaluated (current candidate)
    obj_uk      -   objective function of the last iterate
    p           -   p-value
    tau         -   step size, time discretization of the flow
    rhs         -   right hand side of the semi-implicit update step
    weight      -   weight for the p-Laplacian
    plpl        -   p-Laplacian of the last iterate
    Z           -   binary matrix, which defines the domain
    tol         -   tolerance for the Newton solver
    Nt          -   maximum number of iterations for the Newton solver
    Dnp         -   scaling parameter for the mean value approximation
    cord        -   neighbouring coordinates within the mean value radius
    h           -   spatial discretization
    r           -   radius of the mean value approximation
    explizit    -   binary value whether the step is performed explicitly or not

    Returns
    -------

    '''
    current_u =  next_iterate_w_flow(uk, p, tau, rhs, weight, plpl, Z, tol, Nt, Dnp, cord, h, r, explicit=explicit)
    if current_u is None:
        return None, None
    current_diff = obj_function(current_u, p, r, h, Z, Dnp, cord) - obj_uk
    return current_diff, current_u


def line_search(uk, rhs, tau0, weight, plpl, p, r, h, Z, Dnp, cord, tol, Nt, explicit=False):
    '''
    Performs a line search to find the optimal step size.
     - decreasing the step size if we are not improving our target function
     - increasing the step size if we are improving our target function
     - and stopping if do not improve on the current best but still improving the target function

    Parameters
    ----------
    uk          -   last iterate
    rhs         -   right hand side of the semi-implicit update step
    tau0        -   initial step size, time discretization of the flow
    weight      -   weight for the p-Laplacian
    plpl        -   p-Laplacian of the last iterate
    p           -   p-value
    r           -   radius of the mean value approximation
    h           -   spatial discretization of the domain
    Z           -   binary matrix, which defines the domain
    Dnp         -   scaling parameter for the mean value approximation
    cord        -   neighbouring coordinates within the mean value radius
    tol         -   tolerance for the Newton solver
    Nt          -   maximum number of iterations for the Newton solver
    explizit    -   binary value whether the step is performed explicitly or not

    Returns
    -------
    next iterate of the iterations scheme, and optimal step size
    '''
    obj_uk = obj_function(uk, p, r, h, Z, Dnp, cord)  # get the current function value
    best_tau = 0  # keep track of the best step size value
    best_dk2 = 0  # greatest optimization
    best_u = np.copy(uk)
    print('tau0', tau0)
    decrease = 0.9
    increase = 1.1
    for linesearch_iteration in range(150):
        if linesearch_iteration == 20:
            decrease = 0.7
            increase = 1.3
        if linesearch_iteration == 40:
            decrease = 0.5
            increase = 1.5
        dk2a, curr_u = currrent_difference(uk, obj_uk, p, tau0, rhs, weight, plpl, Z, tol, Nt, Dnp, cord, h, r, explicit=explicit)  # current descend
        print('curr_optimization', dk2a, 'best_optimization', best_dk2, 'curr_tau', tau0, 'best', best_tau)
        if curr_u is None:
            if best_dk2 < 0:
                break
            tau0 = tau0/2
        elif dk2a < best_dk2:
            # save the current best and try even more by increasing stepsize
            best_tau = np.copy(tau0)
            best_dk2 = np.copy(dk2a)
            best_u = np.copy(curr_u)
            tau0 = tau0*increase
        elif 0 <= dk2a:
            if best_dk2 < 0:
                break
            # If we are increasing our targetfunction -> by Taylor we need to smaller alpha to decrease it
            tau0 = tau0*decrease
        else:
            break
        if linesearch_iteration == 149:
            print('max line search iterations reached')
    print('--results of step sizing--')
    print('opt_tau:', best_tau)
    print('promised ascent', -best_dk2)
    print('old_cosim', 1-obj_uk)
    print('new cosim', 1-obj_function(best_u, p, r, h, Z, Dnp, cord))
    print('--stop step sizing--')
    return best_u, best_tau


def flow_step_2d(uk, p, tau, r, h, Z, Dnp, cord, tol, Nt):
    '''
    performs the semi-implicit update step of the cosim flow approach by making a line search for the optimal step size.

    Parameters
    ----------
    uk      -   last iterate
    p       -   p-value
    tau     -   initial step size, time discretization of the flow
    r       -   radius of the mean value approximation
    h       -   spatial discretization of the domain
    Z       -   binary matrix, which defines the domain
    Dnp     -   scaling parameter for the mean value approximation
    cord    -   neighbouring coordinates within the mean value radius
    tol     -   tolerance for the Newton solver
    Nt      -   maximum number of iterations for the Newton solver

    Returns
    -------
    next iterate of the iterations scheme, and the relevant values and metrics (plpl, cosim, RQ, l2 error), and the optimal step size
    '''

    q=p/(p-1)
    plpl = disc_p_lap(p, r, h, Z, uk, Dnp, cord)  # negative p-Laplace operator of the current iterate uk
    nu =  np.linalg.norm(uk.ravel(), ord=p)  # norm of the current iterate uk
    nplpl = np.linalg.norm(plpl.ravel(), ord=q)  # norm of the p-Laplace of uk
    cosim = np.inner(plpl.ravel(), uk.ravel())/(nu*nplpl)  # track the current cosim value
    error = np.linalg.norm(phi(uk, p).ravel()/(nu**(p-1)) - plpl.ravel()/nplpl)  # track the current l2 error
    coeff = p/(nu*nplpl)
    rhs = -cosim * (dNp(uk,p) * nplpl + nu * dNplpl(uk,p, r, h, Z, Dnp, cord))/(nu*nplpl)
    #rhs = rhs/np.linalg.norm(rhs.ravel(), ord=q)
    #tau = 1 - cosim
    if tau !=0:
        print('--start step sizing with tau0 as old best tau')
        new_uk, tau = line_search(uk, rhs, tau, coeff, plpl, p, r, h, Z, Dnp, cord, tol, Nt, explicit=False)
    if tau ==0:
        print('try explicit step')
        print('cosim', cosim)
        new_uk, tau = line_search(uk, rhs, tau, coeff, plpl, p, r, h, Z, Dnp, cord, tol, Nt, explicit=True)
    return new_uk, plpl, cosim, cosim * nplpl/(nu**(p-1)), error, tau


def u_plus(u):
    '''
    computes only the positiv part of u and sets the negative part to zero

    Parameters
    ----------
    u   - discretized function

    Returns
    -------
    np.where(u >= 0, u, 0)
    '''
    return np.where(u >= 0, u, 0)


def u_minus(u):
    '''
    computes only the negative part of u adn returns it with positive sign and sets the original positive part to zero

    Parameters
    ----------
    u   - discretized function

    Returns
    -------
     np.where(u <= 0, -u, 0)
    '''
    return np.where(u <= 0, -u, 0)


def Rplus(u, p):
    '''
    evaluates the Rayleigh quotient at u_plus

    Parameters
    ----------
    u   - discretized function
    p   - p-value

    Returns
    -------
    R(u^+)
    '''
    u_p = u_plus(u)
    d_u_p = np.gradient(u_p, edge_order=2)
    d_u_p = np.sqrt(d_u_p[0]**2 + d_u_p[1]**2)
    x_p = np.where(u_p > 0, 1, 0)
    #if np.sum(x_p) == 0:
    #    return 0
    if np.sum(x_p) < 6:  # nonzero measure of the positive domain
        return 1e12  # np.inf
    else:
        return np.linalg.norm(d_u_p.flatten(), ord=p)**p/np.linalg.norm(u_p.flatten(), ord=p)**p


def Rminus(u, p):
    '''
    evaluates the Rayleigh quotient at u_minus

    Parameters
    ----------
    u   - discretized function
    p   - p-value

    Returns
    -------
    R(u^-)
    '''
    u_m = u_minus(u)
    d_u_m = np.gradient(u_m, edge_order=2)
    d_u_m = np.sqrt(d_u_m[0]**2 + d_u_m[1]**2)
    x_m = np.where(u_m > 0, 1, 0)
    #if np.sum(x_m) == 0:
    #    return 0
    if np.sum(x_m) < 6:  # nonzero measure of the negative domain
        return 1e12  # np.inf
    else:
        return np.linalg.norm(d_u_m.flatten(), ord=p)**p/np.linalg.norm(u_m.flatten(), ord=p)**p


