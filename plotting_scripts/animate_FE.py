import numpy as np
import pandas
from fenics import *

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
meshshape = 'crossed' # 'left' 'right' 'crossed'
name1 = '../study_results/ipm_higher_ex2_FE_'+meshshape+'.pk'
########################################################################################################################################
data_frame1 = pandas.read_pickle(name1)
data_frame1 = data_frame1.fillna(value=np.nan)
#important_quantities = data_frame1['etc'].to_numpy()
#tau, X, Y, Z, p, R, r, h, U0 = important_quantities[:9]  # padded with none
p=3
q = p/(p-1)
print(p)
U_list_1 = data_frame1['U'].to_numpy()
zeta_list_1 = data_frame1['zeta'].to_numpy()[:-1] # padded with none
cosim_list_1 = data_frame1['cosim'].to_numpy()[:-1] # padded with none
RQ_list_1 = data_frame1['RQ'].to_numpy()
dRQ_list_1 = data_frame1['dual_Rayleigh_Quotient'].to_numpy()[:-1]  # padded with none
print(len(cosim_list_1))
print(len(RQ_list_1))
print(len(dRQ_list_1))

print(len(U_list_1))
print(len(zeta_list_1))
print(np.shape(U_list_1[0]))
print(np.shape(zeta_list_1[0]))
print(zeta_list_1[-1])
J = lambda r: np.abs(r)**(p-1) * np.sign(r)

diff_list = np.array([J(u)/np.linalg.norm(J(u)) - zeta/np.linalg.norm(zeta) for u, zeta in zip(U_list_1[1:], zeta_list_1)])
error_list = np.array([np.linalg.norm(diff) for diff in diff_list])
dg_list = np.array([rq**(-1/p) - drq**(1/q) for rq, drq in zip(RQ_list_1[1:], dRQ_list_1)])
print(np.shape(dg_list))
u_sol = U_list_1[0]
mesh = RectangleMesh(Point(-1.0, -1.0), Point(1.0, 1.0), 100, 100, meshshape)  # more connections with crossed
#mesh = UnitSquareMesh(100, 100)
# finite-element-space
V = FunctionSpace(mesh, 'P', 1)

# Extrahiere die Koordinaten der Freiheitsgrade
dof_coordinates = V.tabulate_dof_coordinates().reshape((-1, 2))
x, y = dof_coordinates[:, 0], dof_coordinates[:, 1]

# Anzeige der Lösung
#fig = plt.figure()
#ax = fig.add_subplot(1, 1, 1)
#pl1 = ax.tricontourf(x, y, u_sol, 60, cmap='viridis')  # Verwende tricontourf da es sich um ein trianguliertes mesh handelt
#ax.colorbar(label='Solution value')
#ax.xlabel('x')
#ax.ylabel('y')
#ax.title('PDE Solution')

#t_max = len(U_list_1) - 1

# to control the animation
#def update_time():
#    t = 0
#    while t<t_max:
#        t += anim.direction
#        yield t
# animate the plots in the list
#def animate_1(k):
#    ax.set_title(f'{k}')
#    pl1 = ax.tricontourf(x, y, U_list_1[k], 60, cmap='viridis')
#    return pl1


#def on_press(event):
#    if event.key.isspace():
#        if anim.running:
#            anim.event_source.stop()
#        else:
#            anim.event_source.start()
#        anim.running ^= True
#    elif event.key == 'left':
#        anim.direction = -1
#    elif event.key == 'right':
#        anim.direction = +1

    # Manually update the plot
#    if event.key in ['left','right']:
#        t = anim.frame_seq.__next__()
#        animate_1(t)
#        plt.draw()
#fig.canvas.mpl_connect('key_press_event', on_press)

#anim = animation.FuncAnimation(fig, animate_1, frames=update_time, interval=15, repeat=True, save_count=t_max)
#anim.running = True
#anim.direction = +1

# To save the animation using Pillow as a gif
# writer = animation.PillowWriter(fps=15,
#                                 metadata=dict(artist='Me'),
#                                 bitrate=1800)
# anim.save('evolution.gif', writer=writer)

#plt.show()




#
'''
# 3D-Plot der Lösung
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')  # 3D-Achsen

# Plotten der Oberfläche
ax.plot_trisurf(x, y, u_sol, cmap='viridis', edgecolor='none')

# Labels und Titel
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('Solution value')
ax.set_title('3D Surface Plot of PDE Solution')

plt.show()
'''

t_max = len(U_list_1) - 2

# create a figure with two subfigures
fig = plt.figure(figsize=(12, 10))
subfigs = fig.subfigures(2, 1)
axs_top_1 = subfigs[0].add_subplot(1, 2, 1, projection='3d')
axs_top_2 = subfigs[0].add_subplot(1, 2, 2, projection='3d')
# plot the 3d plots in top figure
ps1 = axs_top_1.tricontourf(x, y, U_list_1[0], 60, cmap='viridis')  # Verwende tricontourf da es sich um ein trianguliertes mesh handelt
axs_top_1.set_title(r'iterates $u^{k}$')
#axs_top_1.set_zlim(np.min(U_list_1[0]), np.max(U_list_1[0]))
ps2 = axs_top_2.tricontourf(x, y, diff_list[0], 60, cmap='viridis')  # Verwende tricontourf da es sich um ein trianguliertes mesh handelt
axs_top_2.set_title(r'error')
#axs_top_2.set_zlim(np.min(diff_list[0]), np.max(diff_list[0]))
# plot the normal plots in lower figure
axs = subfigs[1].subplots(1,3)
axs[0].set_yscale('log')
axs[0].set_ylim(1e-12, 1)
pl1, = axs[0].plot([], [])
#axs[0].set_title('dual Rayleigh quotient')
axs[0].set_title('L2-error')
#axs[0].set_ylim(np.min(dRQ_list_1)*0.9, np.max(dRQ_list_1)*1.05)
pl2, = axs[1].plot([], [])
axs[1].set_title('duality gap')
#axs[1].set_ylim(0, np.max(np.abs(RQ_v_list_1**(-1/p) - dRQ_list_1**(1/q)))*1.05)
#axs[1].set_ylim(-1,1)
pl3, = axs[2].plot([], [])
axs[2].set_title(r'cosine similarity of $-\Delta_p(u^k)$ and $u^k$')
axs[2].set_ylim(0.5, 1.02)
# pl4, = axs[3].plot([], [])
# axs[3].set_title('primal Rayleigh quotient')
# axs[3].set_ylim(np.min(RQ_v_list_1)*0.9, np.max(RQ_v_list_1)*1.05)
# add all plots to a list
pls = [pl1, pl2, pl3, ps1, ps2]
# set limits
for ax in axs:
    ax.set_xlim(0, t_max)

# to control the animation
def update_time():
    t = 0
    while t<t_max:
        t += anim.direction
        yield t
# animate the plots in the list
def animate_1(k):
    axs_top_1.set_title(f'{k}')
    #pls[0].set_data(np.arange(k), dRQ_list_1[:k])
    pls[0].set_data(np.arange(k), error_list[:k])
    pls[1].set_data(np.arange(k), dg_list[:k])
    pls[2].set_data(np.arange(k), cosim_list_1[:k])
    #pls[3].set_data(np.arange(k), RQ_v_list_1[:k])
    pls[3].remove()
    pls[3] = axs_top_1.tricontourf(x, y, U_list_1[k+1], 60, cmap='viridis')
    pls[4].remove()
    pls[4] =axs_top_2.tricontourf(x, y, diff_list[k], 60, cmap='viridis')
    return pls


def on_press(event):
    if event.key.isspace():
        if anim.running:
            anim.event_source.stop()
        else:
            anim.event_source.start()
        anim.running ^= True
    elif event.key == 'left':
        anim.direction = -1
    elif event.key == 'right':
        anim.direction = +1

    # Manually update the plot
    if event.key in ['left','right']:
        t = anim.frame_seq.__next__()
        animate_1(t)
        plt.draw()
fig.canvas.mpl_connect('key_press_event', on_press)

anim = animation.FuncAnimation(fig, animate_1, frames=update_time, interval=15, repeat=True, save_count=t_max)
anim.running = True
anim.direction = +1

# To save the animation using Pillow as a gif
# writer = animation.PillowWriter(fps=15,
#                                 metadata=dict(artist='Me'),
#                                 bitrate=1800)
# anim.save('plots/FE_evolution.gif', writer=writer)

plt.show()

