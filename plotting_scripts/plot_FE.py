import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from scipy.signal import argrelextrema
from fenics import *
import os
mpl.use('macosx')
os.makedirs('plots/', exist_ok=True)

# load all the necessary data
fenics_left = pd.read_pickle('../study_results/ipm_higher_ex2_FE_left.pk')
fenics_left = fenics_left.fillna(value=np.nan)
fenics_right = pd.read_pickle('../study_results/ipm_higher_ex2_FE_right.pk')
fenics_right = fenics_right.fillna(value=np.nan)
fenics_crossed = pd.read_pickle('../study_results/ipm_higher_ex2_FE_crossed.pk')
fenics_crossed = fenics_crossed.fillna(value=np.nan)
mean_value = pd.read_pickle('../study_results/inv_pm_p_3_higher_ex3_delteso_100_it_50.pk')
mean_value = mean_value.fillna(value=np.nan)

important_quantities = mean_value['etc'].to_numpy()
tau, X, Y, Z, p, R, r, h, U0 = important_quantities[:9]  # padded with none
q = p/(p-1)
# load the iterates u
U_fl = fenics_left['U'].to_numpy()
U_fr = fenics_right['U'].to_numpy()
U_fc = fenics_crossed['U'].to_numpy()
U_mv = mean_value['U'].to_numpy()
U_mv[0] = U_mv[0]/np.linalg.norm(U_mv[0].flatten(), ord=p)

# load the metric cosim
cosim_fl = fenics_left['cosim'].to_numpy()[:-1] # padded with none
cosim_fr = fenics_right['cosim'].to_numpy()[:-1]
cosim_fc = fenics_crossed['cosim'].to_numpy()[:-1]
cosim_mv = mean_value['cosim'].to_numpy()

# load the metric l2 error
error_mv = np.sqrt(mean_value['l2'].to_numpy())
# need to calculate the diff first
J = lambda r: np.abs(r)**(p-1) * np.sign(r)
zeta_fl = fenics_left['zeta'].to_numpy()[:-1]
zeta_fr = fenics_right['zeta'].to_numpy()[:-1]
zeta_fc = fenics_crossed['zeta'].to_numpy()[:-1]
#
error_fl = np.array([np.linalg.norm(J(u)/np.linalg.norm(J(u)) - zeta/np.linalg.norm(zeta)) for u, zeta in zip(U_fl, zeta_fl)])
error_fr = np.array([np.linalg.norm(J(u)/np.linalg.norm(J(u)) - zeta/np.linalg.norm(zeta)) for u, zeta in zip(U_fr, zeta_fr)])
error_fc = np.array([np.linalg.norm(J(u)/np.linalg.norm(J(u)) - zeta/np.linalg.norm(zeta)) for u, zeta in zip(U_fc, zeta_fc)])

# load the metric duality gap
# need to load RQ and dRQ first
rq_fl = fenics_left['RQ'].to_numpy()
drq_fl = fenics_left['dual_Rayleigh_Quotient'].to_numpy()[:-1]
rq_fr = fenics_right['RQ'].to_numpy()
drq_fr = fenics_right['dual_Rayleigh_Quotient'].to_numpy()[:-1]
rq_fc = fenics_crossed['RQ'].to_numpy()
drq_fc = fenics_crossed['dual_Rayleigh_Quotient'].to_numpy()[:-1]
plpl_mv = mean_value['p_laplace'].to_numpy()
rq_mv = np.array([np.inner(u.ravel(), plpl.ravel())/np.linalg.norm(u.ravel(), ord=p)**p for u, plpl in zip(U_mv, plpl_mv)])
drq_mv = np.array([np.inner(u.ravel(), plpl.ravel())/np.linalg.norm(plpl.ravel(), ord=q)**q for u, plpl in zip(U_mv, plpl_mv)])
#
dg_fl = np.array([rq**(-1/p) - drq**(1/q) for rq, drq in zip(rq_fl[1:], drq_fl)])
dg_fr = np.array([rq**(-1/p) - drq**(1/q) for rq, drq in zip(rq_fr[1:], drq_fr)])
dg_fc = np.array([rq**(-1/p) - drq**(1/q) for rq, drq in zip(rq_fc[1:], drq_fc)])
dg_mv = np.array([rq**(-1/p) - drq**(1/q) for rq, drq in zip(rq_mv[1:], drq_mv[1:])])
#
#
# caluclate relevant plateaus
arg_c_mv = argrelextrema(cosim_mv, np.greater)
arg_c_fl = argrelextrema(cosim_fl, np.greater)
arg_c_fr = argrelextrema(cosim_fr, np.greater)
arg_c_fc = argrelextrema(cosim_fc, np.greater)
#
arg_dg_mv = argrelextrema(dg_mv, np.less)
arg_dg_fl = argrelextrema(dg_fl, np.less)
arg_dg_fr = argrelextrema(dg_fr, np.less)
arg_dg_fc = argrelextrema(dg_fc, np.less)
#
arg_l2_mv = argrelextrema(error_mv, np.less)
arg_l2_fl = argrelextrema(error_fl, np.less)
arg_l2_fr = argrelextrema(error_fr, np.less)
arg_l2_fc = argrelextrema(error_fc, np.less)
#
#print(arg_c_mv, '\n', arg_dg_mv, '\n', arg_l2_mv)  # 3, 32, 40, 49
#print(arg_c_fl, '\n', arg_dg_fl, '\n', arg_l2_fl)  # 6, 22, 45, 82, 190
#print(arg_c_fr, '\n', arg_dg_fr, '\n', arg_l2_fr)  # 6, 22, 44, 81, 189
#print(arg_c_fc, '\n', arg_dg_fc, '\n', arg_l2_fc)  # 70, 197
#
cmap = mpl.colormaps['viridis']
colors = cmap(np.linspace(0, 1, 4))
# selected extrema
arg_l2_mv = [0, 3, 49]
arg_l2_fl = [6, 22, 45, 82, 190]
arg_l2_fr = [6, 22, 44, 81, 189]
arg_l2_fc = [34, 70, 197]
########################################################################
########################################################################
########################################################################
######################### Start with the plots #########################
########################################################################
########################################################################
########################################################################
text_width_inches = 6.5
fig = plt.figure(figsize=(text_width_inches, text_width_inches))
plt.rc('font', size=11)
#
ax0 = fig.add_subplot(411)
ax0.set_title('Rayleigh quotient')
#ax0.set_xlabel('iteration')
#ax0.set_ylabel('cosim value [a.u.]')
ax0.set_ylim([0, 150])
#ax0.set_yscale('logit')
ax0.plot(rq_fl, 'x', ls='-', markevery=arg_l2_fl, label='FE left',c = colors[0])
ax0.plot(rq_fr, 'x', ls='-', markevery=arg_l2_fr, label='FE right',c = colors[1])
ax0.plot(rq_fc, 'x', ls='-', markevery=arg_l2_fc, label='FE crossed',c = colors[2])
ax0.plot(rq_mv, 'x', ls='-', markevery=arg_l2_mv, label='MV approx.',c = colors[3])
ax0.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
ax0.legend(loc='upper right', prop={'size': 9})
#
ax1 = fig.add_subplot(412)
ax1.set_title('cosim')
#ax1.set_xlabel('iteration')
#ax1.set_ylabel('cosim value [a.u.]')
ax1.set_ylim([1e-2, 1-1e-12])
ax1.set_yscale('logit')
ax1.plot(cosim_fl, 'x', ls='-', markevery=arg_l2_fl, label='FE left',c = colors[0])
ax1.plot(cosim_fr, 'x', ls='-', markevery=arg_l2_fr, label='FE right',c = colors[1])
ax1.plot(cosim_fc, 'x', ls='-', markevery=arg_l2_fc, label='FE crossed',c = colors[2])
ax1.plot(cosim_mv, 'x', ls='-', markevery=arg_l2_mv, label='MV approx.',c = colors[3])
ax1.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
ax1.legend(loc='lower right', prop={'size': 9})
#
ax2 = fig.add_subplot(413, sharex=ax1)
ax2.set_title('duality gap')
#ax2.set_xlabel('iteration')
#ax2.set_ylabel('duality gap value [a.u.]')
ax2.set_ylim([1e-12, 1])
ax2.set_yscale('log')
ax2.plot(dg_fl, 'x', ls='-', markevery=arg_l2_fl, label='FE left',c = colors[0])
ax2.plot(dg_fr, 'x', ls='-', markevery=arg_l2_fr, label='FE right',c = colors[1])
ax2.plot(dg_fc, 'x', ls='-', markevery=arg_l2_fc, label='FE crossed',c = colors[2])
ax2.plot(dg_mv, 'x', ls='-', markevery=arg_l2_mv, label='MV approx.',c = colors[3])
ax2.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
ax2.legend(loc='upper right', prop={'size': 9})
#
ax3 = fig.add_subplot(414, sharex=ax1)
ax3.set_title(r'$l^2$ error')
ax3.set_xlabel('iteration')
#ax3.set_ylabel('error value [a.u.]')
ax3.set_ylim([1e-12, 1])
ax3.set_yscale('log')
ax3.plot(error_fl, 'x', ls='-', markevery=arg_l2_fl, label='FE left',c = colors[0])
ax3.plot(error_fr, 'x', ls='-', markevery=arg_l2_fr, label='FE right',c = colors[1])
ax3.plot(error_fc, 'x', ls='-', markevery=arg_l2_fc, label='FE crossed',c = colors[2])
ax3.plot(error_mv, 'x', ls='-', markevery=arg_l2_mv, label='MV approx.',c = colors[3])
ax3.legend(loc='lower right', prop={'size': 9})
#
plt.tight_layout(w_pad=3.5, h_pad=0.5,  rect=[0, 0, 0.9, 1])
plt.savefig('plots/higher_order_metrics.pdf', dpi=300)
plt.show()
#
########################################################################
########################################################################
########################################################################
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(text_width_inches, text_width_inches/2), subplot_kw=dict(projection='3d'))
plt.rc('font', size=11)
labels=['initial guess', 'local extremum', 'computed eigenfunction']
for i, (ax, arg) in enumerate(zip(axs.ravel(), arg_l2_mv)):
    ax.plot_surface(X, Y, U_mv[arg], cmap=plt.get_cmap('viridis', 256), edgecolor=None, linewidth=0, antialiased=False)
    ax.set_title(labels[i])
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.set_zlabel(r'$u(x)$')
plt.tight_layout(w_pad=3.5, h_pad=0.5,  rect=[0, 0, 0.93, 1])
plt.savefig('plots/mv_extrema.pdf', dpi=300, transparent=False)
plt.show()
########################################################################
########################################################################
########################################################################
# Beispielmesh: Einheitsquadrat diskretisieren
mesh_ex = UnitSquareMesh(5, 5, 'left')
mesh = RectangleMesh(Point(-1.0, -1.0), Point(1.0, 1.0), 100, 100, 'left')
# Funktionsraum und Diskretisierung
V = FunctionSpace(mesh, 'P', 1)
dof_coordinates = V.tabulate_dof_coordinates().reshape((-1, 2))
x, y = dof_coordinates[:, 0], dof_coordinates[:, 1]
#
fig = plt.figure(figsize=(text_width_inches, text_width_inches*3/2))
plt.rc('font', size=11)
#
# 2D-Plot des Meshes in axs[0,0]
ax0 = fig.add_subplot(3, 2, 1)  # Subplot für 2D-Plot
coords_ex = mesh_ex.coordinates()
x_ex, y_ex = coords_ex[:, 0], coords_ex[:, 1]
cells_ex = mesh_ex.cells()
ax0.triplot(x_ex, y_ex, cells_ex, 'k-')
ax0.set_title("finite element 'left' mesh")
ax0.set_xticks([])  # Entfernt x-Ticks
ax0.set_yticks([])  # Entfernt y-Ticks
ax0.set_frame_on(False)
ax0.set_xlabel(r'$x_1$')
ax0.set_ylabel(r'$x_2$')
# 3D Plots für die anderen Subplots
titles = ['1st extremum', '2nd extremum', '1st plateau', '3rd extremum', 'computed eigenfunction']
for idx, ax_index in enumerate(range(2, 6+1)):  # von 2 bis 6 unter Berücksichtigung eines 3x2 Gitters
    ax = fig.add_subplot(3, 2, ax_index, projection='3d')  # Definiere Achse mit 3D-Projektion
    Z = U_fl[arg_l2_fl[idx]]
    ax.plot_trisurf(x, y, Z, cmap=plt.get_cmap('viridis', 256), edgecolor='none', linewidth=0, antialiased=False, alpha=1)
    ax.set_title(titles[idx])
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.set_zlabel(r'$u(x)$')
#
plt.tight_layout(w_pad=3.5, h_pad=0.5,  rect=[0, 0, 0.93, 1])
plt.savefig('plots/fl_extrema.png', dpi=300, transparent=False)
plt.show()
########################################################################
########################################################################
########################################################################
# Beispielmesh: Einheitsquadrat diskretisieren
mesh_ex = UnitSquareMesh(5, 5, 'right')
mesh = RectangleMesh(Point(-1.0, -1.0), Point(1.0, 1.0), 100, 100, 'right')  # größere Diskretisierung

# Funktionsraum und Diskretisierung
V = FunctionSpace(mesh, 'P', 1)
dof_coordinates = V.tabulate_dof_coordinates().reshape((-1, 2))
x, y = dof_coordinates[:, 0], dof_coordinates[:, 1]

fig = plt.figure(figsize=(text_width_inches, text_width_inches*3/2))
plt.rc('font', size=11)

# 2D-Plot des Meshes in axs[0,0]
ax0 = fig.add_subplot(3, 2, 1)  # Subplot für 2D-Plot
coords_ex = mesh_ex.coordinates()
x_ex, y_ex = coords_ex[:, 0], coords_ex[:, 1]
cells_ex = mesh_ex.cells()
ax0.triplot(x_ex, y_ex, cells_ex, 'k-')
ax0.set_title("finite element 'right' mesh")
ax0.set_xticks([])  # Entfernt x-Ticks
ax0.set_yticks([])  # Entfernt y-Ticks
ax0.set_frame_on(False)
ax0.set_xlabel(r'$x_1$')
ax0.set_ylabel(r'$x_2$')

# 3D Plots für die anderen Subplots
for idx, ax_index in enumerate(range(2, 6+1)):  # von 2 bis 6 unter Berücksichtigung eines 3x2 Gitters
    ax = fig.add_subplot(3, 2, ax_index, projection='3d')  # Definiere Achse mit 3D-Projektion
    Z = U_fr[arg_l2_fr[idx]]
    ax.plot_trisurf(x, y, Z, cmap=plt.get_cmap('viridis', 256), edgecolor='none', linewidth=0, antialiased=False, alpha=1)
    ax.set_title(titles[idx])
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.set_zlabel(r'$u(x)$')

plt.tight_layout(w_pad=3.5, h_pad=0.5,  rect=[0, 0, 0.93, 1])
plt.savefig('plots/fr_extrema.png', dpi=300, transparent=False)
plt.show()
########################################################################
########################################################################
########################################################################
# Beispielmesh: Einheitsquadrat diskretisieren
mesh_ex = UnitSquareMesh(5, 5, 'crossed')
mesh = RectangleMesh(Point(-1.0, -1.0), Point(1.0, 1.0), 100, 100, 'crossed')  # größere Diskretisierung

# Funktionsraum und Diskretisierung
V = FunctionSpace(mesh, 'P', 1)
dof_coordinates = V.tabulate_dof_coordinates().reshape((-1, 2))
x, y = dof_coordinates[:, 0], dof_coordinates[:, 1]

fig = plt.figure(figsize=(text_width_inches, text_width_inches))
plt.rc('font', size=11)
# 2D-Plot des Meshes in axs[0,0]
ax0 = fig.add_subplot(2, 2, 1)  # Subplot für 2D-Plot
coords_ex = mesh_ex.coordinates()
x_ex, y_ex = coords_ex[:, 0], coords_ex[:, 1]
cells_ex = mesh_ex.cells()
ax0.triplot(x_ex, y_ex, cells_ex, 'k-')
ax0.set_title("finite element 'crossed' mesh")
ax0.set_xticks([])  # Entfernt x-Ticks
ax0.set_yticks([])  # Entfernt y-Ticks
ax0.set_frame_on(False)
ax0.set_xlabel(r'$x_1$')
ax0.set_ylabel(r'$x_2$')

titles = ['1st plateau', '1st extremum', 'computed eigenfunction']
# 3D Plots für die anderen Subplots
for idx, ax_index in enumerate(range(2, 4+1)):  # von 2 bis 4 unter Berücksichtigung eines 1x2 Gitters
    ax = fig.add_subplot(2, 2, ax_index, projection='3d')  # Definiere Achse mit 3D-Projektion
    Z = U_fc[arg_l2_fc[idx]]
    ax.plot_trisurf(x, y, Z, cmap=plt.get_cmap('viridis', 256), edgecolor='none', linewidth=0, antialiased=False, alpha=1)
    ax.set_title(titles[idx])
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.set_zlabel(r'$u(x)$')

plt.tight_layout(w_pad=3.5, h_pad=0.5,  rect=[0, 0, 0.93, 1])
plt.savefig('plots/fc_extrema.png', dpi=300, transparent=False)
plt.show()

