import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, Circle, Arrow
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation


def fitness(lengths, xg, yg, alpha):
    x = np.sum(lengths * np.cos(alpha), axis=1)
    y = np.sum(lengths * np.sin(alpha), axis=1)
    res = (x - xg)**2 + (y - yg)**2
    
    return res

def pso(lengths, xg, yg):
    num_particles = 500
    num_dimensions = len(lengths)
    num_iterations = 300
    w = 0.7
    c1 = 1.5
    c2 = 1.5
    bounds = np.array([[0, 2*np.pi]] * len(lengths))
    positions = np.random.uniform(bounds[:, 0], bounds[:, 1], (num_particles, num_dimensions))
    velocities = np.zeros((num_particles, num_dimensions))
    personal_best_positions = positions.copy()
    personal_best_scores = fitness(lengths, xg, yg, positions)
    global_best_idx = np.argmin(personal_best_scores)
    global_best_position = personal_best_positions[global_best_idx].copy()
    for _ in range(num_iterations):
        r1 = np.random.rand(num_particles, num_dimensions)
        r2 = np.random.rand(num_particles, num_dimensions)
        velocities = (
            w * velocities
            + c1 * r1 * (personal_best_positions - positions)
            + c2 * r2 * (global_best_position - positions)
        )
        positions += velocities
        positions = np.clip(positions, bounds[:, 0], bounds[:, 1])
        scores = fitness(lengths, xg, yg, positions)
        mask = scores < personal_best_scores
        personal_best_scores[mask] = scores[mask]
        personal_best_positions[mask] = positions[mask]
        best_idx = np.argmin(personal_best_scores)
        if personal_best_scores[best_idx] < fitness(lengths, xg, yg, global_best_position.reshape(1, -1))[0]:
            global_best_position = personal_best_positions[best_idx].copy()
    return global_best_position


def find_alpha(lengths, xg, yg):
    # generate n delta alpha and keep best one
    n = 10
    best_fitness = fitness(lengths, xg, yg, alpha)
    best_alpha = alpha.copy()
    alpha_copy = alpha.copy()
    for _ in range(n):
        delta_alpha = np.random.uniform(-2*np.pi*0.03, 2*np.pi*0.03, len(lengths))
        alpha_copy += delta_alpha
        current_fitness = fitness(lengths, xg, yg, alpha_copy)
        if current_fitness < best_fitness:
            best_fitness = current_fitness
            best_alpha = alpha_copy.copy()
        alpha_copy -= delta_alpha
    return best_alpha

def update(frame, slider, ball, ball_pos, lines, lengths, alpha):
    x, y = 0, 0
    global perc
    global temp_alpha
    perc += (1 - perc)/slider.val
    delta = (alpha - prev_alpha + np.pi) % (2*np.pi) - np.pi
    
    temp_alpha[:] = (prev_alpha + perc * delta) % (2*np.pi)
    
    for i, line in enumerate(lines):
        next_x, next_y = x + lengths[i] * np.cos(temp_alpha[i]), y + lengths[i] * np.sin(temp_alpha[i])
        line.set_xdata([x, next_x])
        line.set_ydata([y, next_y])
        x, y = next_x, next_y
    ball.set_center(([ball_pos[0]], [ball_pos[1]]))
    return *lines, ball,

def on_click(event, axes, ball_pos, alpha):
    # if event is not in axes, return
    if event.inaxes is None or event.inaxes != axes:
        return
    global perc
    global prev_alpha
    prev_alpha[:] = alpha
    perc = 0
    print(dir(event))
    ball_pos[0] = event.xdata
    ball_pos[1] = event.ydata
    alpha[:] = pso(lengths, ball_pos[0], ball_pos[1])


if __name__ == '__main__':
    
    lengths = np.array([10, 20, 6, 8, 3])
    ball_pos = [0, sum(lengths)*2/3]
    alpha = pso(lengths, *ball_pos)
    prev_alpha = np.zeros(len(lengths))
    temp_alpha = np.zeros(len(lengths))
    perc = 0
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_title('Click to change alpha')
    ax.set_xlim(-sum(lengths)*1.1, sum(lengths)*1.1)
    ax.set_ylim(-sum(lengths)*1.1, sum(lengths)*1.1)
    x, y = 0, 0
    lines = []
    for i, length in enumerate(lengths):
        next_x, next_y = x + length * np.cos(alpha[i]), y + length * np.sin(alpha[i])
        line, = ax.plot([x, next_x], [y, next_y], 'o-', lw=2, c='r')
        lines.append(line)
        x, y = next_x, next_y
    
    ball = Circle(ball_pos, 1.2)
    ax.add_patch(ball)
    # add an input slider between 1.1 and 10 named inverse speed
    ax_slider = plt.axes([0.15, 0.01, 0.7, 0.03])
    slider = Slider(ax_slider, 'Inverse Speed', 1.1, 10, valinit=5, valstep=0.01)


    ani = FuncAnimation(fig, lambda f: update(f, slider, ball, ball_pos, lines, lengths, alpha), interval=1000/60, blit=True)
    fig.canvas.mpl_connect('button_press_event', lambda e: on_click(e, ax, ball_pos, alpha))
    plt.show()
