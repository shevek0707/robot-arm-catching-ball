import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches

def forward_kinematics(lengths: list[float], theta: list[float]):
    positions = [(0, 0, 0)]
    x, y = 0, 0
    for i in range(len(lengths)):
        next_x = x + lengths[i]*np.cos(theta[i])
        next_y = y + lengths[i]*np.sin(theta[i])
        x, y = next_x, next_y
        positions.append((x, y, 0))
    return positions


def move_arm_with_sliders(num_arms: int=4):

    def update(val):
        positions = forward_kinematics(
            [ls.val for ls in length_sliders], [ts.val/180*np.pi for ts in theta_sliders]
        )
        for i, (start, end) in enumerate(zip(positions[:-1], positions[1:])):
            print(lines[i])
            lines[i].set_xdata([start[0], end[0]])
            lines[i].set_ydata([start[1], end[1]])
    fig, ax = plt.subplots(dpi=150)
    ax.set_aspect('equal')
    ax.set_title('Click to change alpha')
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    length_sliders_axes = [
        plt.axes([0.05, 0.7 - i*0.12, 0.15, 0.03])
        for i in range(num_arms)
    ]
    theta_sliders_axes = [
        plt.axes([0.8, 0.7 - i*0.12, 0.15, 0.03])
        for i in range(num_arms)
    ]
    length_sliders = [
        Slider(length_sliders_axes[i], f'length{i+1}', 0.5, 30, valinit=5, valstep=0.5)
        for i in range(num_arms)
    ]
    theta_sliders = [
        Slider(theta_sliders_axes[i], f'theta{i+1}', 0, 360, valinit=0, valstep=5)
        for i in range(num_arms)
    ]
    lines = [ax.plot([0, 0], [0, 0], 'o-', lw=2, c='r')[0] for _ in range(num_arms)]
    update(0)
    for s in length_sliders + theta_sliders:
        s.on_changed(update)
    plt.show()

def single_particle_pso(velocity, best, global_best, W, c1, c2):
    fig, ax = plt.subplots(dpi=150)
    ax.set_aspect('equal')
    ax.set_title('Click to change alpha')
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)

    def update(frame):
        nonlocal velocity, best, global_best, particle_arrow
        r1, r2 = np.ones(2)
        velocity = (
            W * velocity
            + c1 * r1 * (best - np.array([0, 0]))
            + c2 * r2 * (global_best - np.array([0, 0]))
        )
        new_position = np.array([0, 0]) + velocity
        #ax.clear()
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
        ax.plot(new_position[0], new_position[1], 'ro')
        # Update best and global best if new position is better
        new_arrow = ax.arrow
        return ax,

    best_dot = ax.plot(best[0], best[1], 'go')
    global_best_dot = ax.plot(global_best[0], global_best[1], 'bo')
    # draw particle as an arrow with a dot at the beginning and a triangle tip at the end, its direction and length is represented by velocity
    particle_arrow = ax.arrow(0, 0, velocity[0], velocity[1], head_width=1, head_length=1, fc='r', ec='r')
    ani = FuncAnimation(fig, update, frames=100, interval=100, blit=True)
    plt.show()



single_particle_pso(
    velocity=np.array([1, 1]), 
    best=np.array([5, 5]), 
    global_best=np.array([10, 10]), 
    W=0.7, 
    c1=1.5, 
    c2=1.5
)
