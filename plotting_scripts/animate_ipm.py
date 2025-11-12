import numpy as np
import pandas
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from plotting_scripts.plot_ipm import diff_list_1

mpl.use('macosx')

name1 = '../study_results/ipm_ex1_p_2.pk'

########################################################################################################################################
# Extract the data
data_frame1 = pandas.read_pickle(name1)
p = data_frame1['p'].to_numpy()[0]
X, Y = data_frame1['xy'].to_numpy()[0]
r = data_frame1['r'].to_numpy()[0]
U0 = data_frame1['u0'].to_numpy()[0]
q = p / (p - 1)

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
diff_list_1 = np.array([plpl/np.linalg.norm(plpl.ravel(), ord=q) - u for plpl, u in zip(plpl_list_1, U_list_1)])
########################################################################################################################################

t_max = len(U_list_1) - 1

# create a figure with two subfigures
fig = plt.figure(figsize=(12, 10))
subfigs = fig.subfigures(2, 1)
axs_top_1 = subfigs[0].add_subplot(1, 2, 1, projection='3d')
axs_top_2 = subfigs[0].add_subplot(1, 2, 2, projection='3d')
# plot the 3d plots in top figure
ps1 = axs_top_1.plot_surface(X, Y, U_list_1[0], cmap="viridis")
axs_top_1.set_title(r'iterates $u^{k}$')
ps2 = axs_top_2.plot_surface(X, Y, diff_list_1[0], cmap="viridis")
axs_top_2.set_title(r'error')
axs_top_2.set_zlim(np.min(diff_list_1[0]), np.max(diff_list_1[0]))
# plot the normal plots in lower figure
axs = subfigs[1].subplots(1,3)
axs[0].set_yscale('log')
axs[0].set_ylim(1e-12, 1)
pl1, = axs[0].plot([], [])
axs[0].set_title('L2-error')
pl2, = axs[1].plot([], [])
axs[1].set_title('duality gap')
pl3, = axs[2].plot([], [])
axs[2].set_title(r'cosine similarity of $-\Delta_p(u^k)$ and $u^k$')
axs[2].set_ylim(0.5, 1.02)

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
    pls[0].set_data(np.arange(k), l2_error_list_1[:k])
    pls[1].set_data(np.arange(k), RQ_list_1[:k]**(-1/p) - dRQ_list_1[:k]**(1/q))
    pls[2].set_data(np.arange(k), cosim_list_1[:k])
    pls[3].remove()
    pls[3] = axs_top_1.plot_surface(X, Y, U_list_1[k], cmap="viridis")
    pls[4].remove()
    pls[4] = axs_top_2.plot_surface(X, Y, diff_list_1[k], cmap="viridis")
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

anim = animation.FuncAnimation(fig, animate_1, frames=update_time, interval=300, repeat=True, save_count=t_max)
anim.running = True
anim.direction = +1

# To save the animation using Pillow as a gif
# writer = animation.PillowWriter(fps=15,
#                                 metadata=dict(artist='Me'),
#                                 bitrate=1800)
# anim.save('plots/ipm_evolution.gif', writer=writer)

plt.show()

