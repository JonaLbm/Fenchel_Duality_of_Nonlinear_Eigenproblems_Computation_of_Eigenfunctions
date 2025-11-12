import numpy as np
import pandas
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.use('macosx')

name1 = '../study_results/ipm_ex1_p_1,77.pk'
name2 = '../study_results/ipm_ex1_p_2.pk'
name3 = '../study_results/ipm_ex1_p_5.pk'
########################################################################################################################################
# Extract the data
data_frame1 = pandas.read_pickle(name1)
p1 = data_frame1['p'].to_numpy()[0]
X, Y = data_frame1['xy'].to_numpy()[0]
r = data_frame1['r'].to_numpy()[0]
U0 = data_frame1['u0'].to_numpy()[0]
q1 = p1 / (p1 - 1)

U_list_1 = data_frame1['iterates'].to_numpy()[0]
plpl_list_1 = data_frame1['plpl'].to_numpy()[0]
l2_error_list_1 = data_frame1['error'].to_numpy()[0]
cosim_list_1 = data_frame1['COSIM'].to_numpy()[0]
U_list_1 = np.array(U_list_1)
plpl_list_1 = np.array(plpl_list_1)
l2_error_list_1 = np.array(l2_error_list_1)
cosim_list_1 = np.array(cosim_list_1)

# compute more metrics
RQ_list_1 = np.array([np.inner(plpl.ravel(), u.ravel()) for plpl, u in zip(plpl_list_1, U_list_1)])
dRQ_list_1 = np.array([np.inner(plpl.ravel(), u.ravel())/(np.linalg.norm(plpl.ravel(), ord=q)**q) for plpl, u in zip(plpl_list_1, U_list_1)])
########################################################################################################################################

data_frame2 = pandas.read_pickle(name2)
p2 = data_frame2['p'].to_numpy()[0]
X, Y = data_frame2['xy'].to_numpy()[0]
r = data_frame2['r'].to_numpy()[0]
U0 = data_frame2['u0'].to_numpy()[0]
q2 = p2 / (p2 - 1)

U_list_2 = data_frame2['iterates'].to_numpy()[0]
plpl_list_2 = data_frame2['plpl'].to_numpy()[0]
l2_error_list_2 = data_frame2['error'].to_numpy()[0]
cosim_list_2 = data_frame2['COSIM'].to_numpy()[0]
U_list_2 = np.array(U_list_2)
plpl_list_2 = np.array(plpl_list_2)
l2_error_list_2 = np.array(l2_error_list_2)
cosim_list_2 = np.array(cosim_list_2)

RQ_list_2 = np.array([np.inner(plpl.ravel(), u.ravel()) for plpl, u in zip(plpl_list_2, U_list_2)])
dRQ_list_2 = np.array([np.inner(plpl.ravel(), u.ravel())/(np.linalg.norm(plpl.ravel(), ord=q2)**q2) for plpl, u in zip(plpl_list_2, U_list_2)])
########################################################################################################################################
# Extract the data
data_frame3 = pandas.read_pickle(name3)
p3 = data_frame3['p'].to_numpy()[0]
X, Y = data_frame3['xy'].to_numpy()[0]
r = data_frame3['r'].to_numpy()[0]
U0 = data_frame3['u0'].to_numpy()[0]
q3 = p3 / (p3 - 1)

U_list_3 = data_frame3['iterates'].to_numpy()[0]
plpl_list_3 = data_frame3['plpl'].to_numpy()[0]
l2_error_list_3 = data_frame3['error'].to_numpy()[0]
cosim_list_3 = data_frame3['COSIM'].to_numpy()[0]
U_list_3 = np.array(U_list_3)
plpl_list_3 = np.array(plpl_list_3)
l2_error_list_3 = np.array(l2_error_list_3)
cosim_list_3 = np.array(cosim_list_3)

# compute more metrics
RQ_list_3 = np.array([np.inner(plpl.ravel(), u.ravel()) for plpl, u in zip(plpl_list_3, U_list_3)])
dRQ_list_3 = np.array([np.inner(plpl.ravel(), u.ravel())/(np.linalg.norm(plpl.ravel(), ord=q3)**q3) for plpl, u in zip(plpl_list_3, U_list_3)])
########################################################################################################################################
##########################################################################################################################################

p_s = [p1, p2, p3]
cmap = mpl.colormaps['viridis']
colors = cmap(np.linspace(0, 1, 3))

# plotting the convergence
n_index = np.arange(31)
n_inde = 31

text_width_inches = 6.5
fig = plt.figure(figsize=(text_width_inches, text_width_inches))
plt.rc('font', size=11)

ax1 = fig.add_subplot(412)
ax1.set_yscale('logit')
ax1.set_ylim([1e-2, 1 - 1e-12])
ax1.plot(n_index, cosim_list_1[:n_inde], c=colors[0])
ax1.plot(n_index, cosim_list_2[:n_inde], c=colors[1])
ax1.plot(n_index, cosim_list_3[:n_inde], c=colors[2])
plt.legend([r'p=1.77', r'p=2', r'p=5'])
ax1.set_title('cosim')
#
ax2 = fig.add_subplot(414)
ax2.plot(n_index, l2_error_list_1[:n_inde], c=colors[0])
ax2.plot(n_index, l2_error_list_2[:n_inde], c=colors[1])
ax2.plot(n_index, l2_error_list_3[:n_inde], c=colors[2])
ax2.set_yscale('log')
ax2.set_ylim([1e-12, 1])
plt.legend([r'p=1.77', r'p=2', r'p=5'])
ax2.set_title(r'$l^2$ error')
#
ax3 = fig.add_subplot(411)
ax3.plot(n_index, dRQ_list_1[:n_inde] ** (1 / q1), c=colors[0])
ax3.plot(n_index, dRQ_list_2[:n_inde] ** (1 / q2), c=colors[1])
ax3.plot(n_index, dRQ_list_3[:n_inde] ** (1 / q3), c=colors[2])
plt.legend([r'p=1.77', r'p=2', r'p=5'])
ax3.set_title('dual Rayleigh quotient')
#
ax4 = fig.add_subplot(413)
ax4.set_yscale('log')
ax4.set_ylim([1e-12, 1])
ax4.plot(n_index, RQ_list_1 ** (-1 / p1) - dRQ_list_1 ** (1 / q1), c=colors[0])
ax4.plot(n_index, RQ_list_2 ** (-1 / p2) - dRQ_list_2 ** (1 / q2), c=colors[1])
ax4.plot(n_index, RQ_list_3 ** (-1 / p3) - dRQ_list_3 ** (1 / q3), c=colors[2])
plt.legend([r'p=1.77', r'p=2', r'p=5'])
ax4.set_title('duality gap')
plt.tight_layout(w_pad=3.5, h_pad=0.5, rect=[0, 0, 0.9, 1])
plt.savefig('plots/inv_pm_conv.pdf', dpi=300)
plt.show()

# Plotting solutions and errors
max_index = -1  # np.nanargmin(l2_error_list_1)
U1 = U_list_1[max_index]
print(f'p=1.77: Error: {l2_error_list_1[max_index]}, cosim: {cosim_list_1[max_index]}')

max_index = np.nanargmin(l2_error_list_2)
U2 = U_list_2[max_index]
print(max_index)
print(f'p=2: Error: {l2_error_list_2[max_index]}, cosim: {cosim_list_2[max_index]}')

max_index = np.nanargmin(l2_error_list_3)
U3 = U_list_3[max_index]
print(max_index)
print(f'p=5: Error: {l2_error_list_3[max_index]}, cosim: {cosim_list_3[max_index]}')

text_width_inches = 6.5
fig = plt.figure(figsize=(text_width_inches, text_width_inches))
plt.rc('font', size=11)
ax1 = fig.add_subplot(221, projection='3d')
ax1.plot_surface(X, Y, U0, cmap="viridis", edgecolor=None, linewidth=0, antialiased=False)
ax1.set_title("initial datum")
ax1.set_xlabel(r'$x_1$')
ax1.set_ylabel(r'$x_2$')
ax1.set_zlabel(r'$u(x)$')

ax2 = fig.add_subplot(222, projection='3d')
ax2.plot_surface(X, Y, U1, cmap="viridis", edgecolor=None, linewidth=0, antialiased=False)
ax2.set_title("p=1.77")
ax2.set_xlabel(r'$x_1$')
ax2.set_ylabel(r'$x_2$')
ax2.set_zlabel(r'$u(x)$')

ax3 = fig.add_subplot(223, projection='3d')
ax3.plot_surface(X, Y, U2, cmap="viridis", edgecolor=None, linewidth=0, antialiased=False)
ax3.set_title("p=2")
ax3.set_xlabel(r'$x_1$')
ax3.set_ylabel(r'$x_2$')
ax3.set_zlabel(r'$u(x)$')

ax4 = fig.add_subplot(224, projection='3d')
ax4.plot_surface(X, Y, U3, cmap="viridis")
ax4.set_title("p=5")
ax4.set_xlabel(r'$x_1$')
ax4.set_ylabel(r'$x_2$')
ax4.set_zlabel(r'$u(x)$')

plt.tight_layout(w_pad=3.5, h_pad=0.5, rect=[0, 0, 0.93, 1])
plt.savefig('plots/ground_states.pdf', dpi=300, transparent=False)
plt.show()

