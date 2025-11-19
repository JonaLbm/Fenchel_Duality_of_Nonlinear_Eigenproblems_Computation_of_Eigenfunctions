import pandas
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from helpers import phi
import os
mpl.use('macosx')
os.makedirs('plots/', exist_ok=True)

name1 = '../study_results/cosim_flow_ex1.pk'
name2 = '../study_results/cosim_flow_ex2.pk'
########################################################################################################################################
# Extract the data
data_frame1 = pandas.read_pickle(name1)
p1 = data_frame1['p'].to_numpy()[0]
X1, Y1 = data_frame1['xy'].to_numpy()[0]
r1 = data_frame1['r'].to_numpy()[0]
U01 = data_frame1['u0'].to_numpy()[0]
q1 = p1 / (p1 - 1)

U_list_1 = data_frame1['iterates'].to_numpy()[0]
plpl_list_1 = data_frame1['plpl'].to_numpy()[0]
l2_error_list_1 = data_frame1['error'].to_numpy()[0]
cosim_list_1 = data_frame1['COSIM'].to_numpy()[0]
U_list_1 = np.array(U_list_1[:-1])  # the iterates are by one out of sync with the other lists, bc the metrics get computed with the current iterate, but the new iterate gets already added
plpl_list_1 = np.array(plpl_list_1[1:])
l2_error_list_1 = np.array(l2_error_list_1[1:])
cosim_list_1 = np.array(cosim_list_1[1:])

# compute more metrics
RQ_list_1 = np.array([np.inner(plpl.ravel(), u.ravel()) for plpl, u in zip(plpl_list_1, U_list_1)])
dRQ_list_1 = np.array([np.inner(plpl.ravel(), u.ravel())/(np.linalg.norm(plpl.ravel(), ord=q1)**q1) for plpl, u in zip(plpl_list_1, U_list_1)])
diff_list_1 = np.array([plpl/np.linalg.norm(plpl.ravel(), ord=q1) - phi(u,p1) for plpl, u in zip(plpl_list_1, U_list_1)])
duality_gap_1 = np.array([rq_i**(-1/p1) - drq_i**(1/q1) for rq_i, drq_i in zip(RQ_list_1, dRQ_list_1)])
########################################################################################################################################
# Extract the data
data_frame2 = pandas.read_pickle(name2)
p2 = data_frame2['p'].to_numpy()[0]
X2, Y2 = data_frame2['xy'].to_numpy()[0]
r2 = data_frame2['r'].to_numpy()[0]
U02 = data_frame2['u0'].to_numpy()[0]
q2 = p2 / (p2 - 1)

U_list_2 = data_frame2['iterates'].to_numpy()[0]
plpl_list_2 = data_frame2['plpl'].to_numpy()[0]
l2_error_list_2 = data_frame2['error'].to_numpy()[0]
cosim_list_2 = data_frame2['COSIM'].to_numpy()[0]
U_list_2 = np.array(U_list_2[:-1])  # the iterates are by one out of sync with the other lists, bc the metrics get computed with the current iterate, but the new iterate gets already added
plpl_list_2 = np.array(plpl_list_2[1:])
l2_error_list_2 = np.array(l2_error_list_2[1:])
cosim_list_2 = np.array(cosim_list_2[1:])

# compute more metrics
RQ_list_2 = np.array([np.inner(plpl.ravel(), u.ravel()) for plpl, u in zip(plpl_list_2, U_list_2)])
dRQ_list_2 = np.array([np.inner(plpl.ravel(), u.ravel())/(np.linalg.norm(plpl.ravel(), ord=q2)**q2) for plpl, u in zip(plpl_list_2, U_list_2)])
diff_list_2 = np.array([plpl/np.linalg.norm(plpl.ravel(), ord=q2) - phi(u,p2) for plpl, u in zip(plpl_list_2, U_list_2)])
duality_gap_2 = np.array([rq_i**(-1/p2) - drq_i**(1/q2) for rq_i, drq_i in zip(RQ_list_2, dRQ_list_2)])

########################################################################################################################################
# Define the L-shaped domain.
Z1 = np.zeros_like(X1, dtype=bool)
Z1[(Y1 <= 0) & (X1 <= 1.0)] = True  # Bottom part of L
Z1[(Y1 <= 1.0) & (X1 <= 0)] = True  # Left part of L
Z1[(Y1 < -1)] = False
Z1[(X1 < -1)] = False

# # square
Z2 = np.zeros_like(X2, dtype=bool)
Z2[(np.abs(Y2) <= 1.0) & (np.abs(X2) <= 1.0)] = True

# make plot
text_width_inches = 6.5
cmap = mpl.colormaps['viridis']
colors = cmap(np.linspace(0, 1, 2))
fig = plt.figure(figsize=(text_width_inches,text_width_inches))
plt.rc('font', size=11)
#
ax1 = fig.add_subplot(222, projection='3d')
Xs = X1[Z2].reshape((100,100))
Ys = Y1[Z2].reshape((100,100))
Us = U_list_1[-1]
Us = Us[Z2].reshape((100,100))
ax1.plot_surface(Xs, Ys, Us, cmap=plt.get_cmap('viridis', 256), edgecolor=None, linewidth=0, antialiased=False)#, edgecolor=None, linewidth=0, antialiased=False)
ax1.set_title(r'computed eigenfunction')
ax1.set_xlabel(r'$x_1$')
ax1.set_ylabel(r'$x_2$')
#
ax2 = fig.add_subplot(224, projection='3d')
Xs2 = X2[Z2].reshape((100,100))
Ys2 = Y2[Z2].reshape((100,100))
Us2 = U_list_2[-1]
Us2 = Us2[Z2].reshape((100,100))
ax2.plot_surface(Xs2, Ys2,Us2, cmap=plt.get_cmap('viridis', 256), edgecolor=None, linewidth=0, antialiased=False)#, edgecolor=None, linewidth=0, antialiased=False)
ax2.set_title(r'local extrema')
ax2.set_xlabel(r'$x_1$')
ax2.set_ylabel(r'$x_2$')
#
ax3 = fig.add_subplot(221, projection='3d')
Xs = X1[Z2].reshape((100,100))
Ys = Y1[Z2].reshape((100,100))
Us3 = U01
Us3 = Us3[Z2].reshape((100,100))
ax3.plot_surface(Xs, Ys, Us3, cmap=plt.get_cmap('viridis', 256), edgecolor=None, linewidth=0, antialiased=False)#, edgecolor=None, linewidth=0, antialiased=False)
ax3.set_title(r'ex 1')
ax3.set_xlabel(r'$x_1$')
ax3.set_ylabel(r'$x_2$')
#
ax4 = fig.add_subplot(223, projection='3d')
Xs2 = X2[Z2].reshape((100,100))
Ys2 = Y2[Z2].reshape((100,100))
Us4 = U02
Us4 = Us4/np.linalg.norm(Us4.ravel(), ord=3)
Us4 = Us4[Z2].reshape((100,100))
ax4.plot_surface(Xs2, Ys2,Us4, cmap=plt.get_cmap('viridis', 256), edgecolor=None, linewidth=0, antialiased=False)#, edgecolor=None, linewidth=0, antialiased=False)
ax4.set_title(r'ex 2')
ax4.set_xlabel(r'$x_1$')
ax4.set_ylabel(r'$x_2$')
#
plt.tight_layout(w_pad=3.5, h_pad=0.5,  rect=[0, 0, 0.93, 1])
plt.savefig('plots/results_gd.pdf', dpi=300, transparent=False)
plt.show()
#
#
fig = plt.figure(figsize=(text_width_inches,text_width_inches))
plt.rc('font', size=11)
#
ax1 = fig.add_subplot(412)
ax1.set_ylim([1e-2, 1-1e-12])
ax1.set_yscale('logit')
ax1.set_title('cosim')
ax1.plot(cosim_list_1, label=r'ex 1', c=colors[0])
ax1.plot(cosim_list_2, label=r'ex 2', c=colors[1])
ax1.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
ax1.legend(loc='lower right', prop={'size': 9})
#
ax2 = fig.add_subplot(413)
ax2.set_yscale('log')
ax2.set_title('duality gap')
ax2.plot(duality_gap_1, label=r'ex 1', c=colors[0])
ax2.plot(duality_gap_2, label=r'ex 2', c=colors[1])
ax2.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
ax2.legend(loc='upper right', prop={'size': 9})
#
ax3 = fig.add_subplot(414)
ax3.set_yscale('log')
ax3.set_title(r'$l^2$ error')
ax3.plot(l2_error_list_1, label=r'ex 1', c=colors[0])
ax3.plot(l2_error_list_2, label=r'ex 2', c=colors[1])
ax3.legend(loc='upper right', prop={'size': 9})
#
ax4 = fig.add_subplot(411)
ax4.set_title('Rayleigh quotient')
ax4.set_yscale('log')
ax4.plot(RQ_list_1, label=r'ex 1', c=colors[0])
ax4.plot(RQ_list_2, label=r'ex 2', c=colors[1])
ax4.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
ax4.legend(loc='upper right', prop={'size': 9})#
#
plt.tight_layout(w_pad=3.5, h_pad=0.5,  rect=[0, 0, 0.9, 1])
plt.savefig('plots/conv_gd.pdf', dpi=300)
plt.show()
