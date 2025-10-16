import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, Circle, Arrow
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation


def fitness(lengths, xg, yg, theta):
    x = np.sum(lengths * np.cos(theta), axis=1)
    y = np.sum(lengths * np.sin(theta), axis=1)
    res = (x - xg)**2 + (y - yg)**2
    return res


def pso(lengths, xg, yg, num_particles=100, num_iterations=50, w=0.7, c1=1.5, c2=1.5):

    num_dimensions = len(lengths)
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



def update(frame, ball, ball_pos, lines, lengths, theta, a=5):
    x, y = 0, 0
    global perc
    global temp_theta
    perc += (1 - perc)/a
    delta = (theta - prev_theta + np.pi) % (2*np.pi) - np.pi
    temp_theta[:] = (prev_theta + perc * delta) % (2*np.pi)
    
    for i, line in enumerate(lines):
        next_x, next_y = x + lengths[i] * np.cos(temp_theta[i]), y + lengths[i] * np.sin(temp_theta[i])
        line.set_xdata([x, next_x])
        line.set_ydata([y, next_y])
        x, y = next_x, next_y
    ball.set_center(([ball_pos[0]], [ball_pos[1]]))
    return *lines, ball,


def on_click(event, axes, ball_pos, theta):

    if event.inaxes is None or event.inaxes != axes:
        return
    global perc
    global prev_theta
    prev_theta[:] = theta
    perc = 0
    ball_pos[0] = event.xdata
    ball_pos[1] = event.ydata
    theta[:] = pso(lengths, ball_pos[0], ball_pos[1])


if __name__ == '__main__':
    
    lengths = np.array([10, 8, 5, 2])
    ball_pos = [0, sum(lengths)*2/3]

    a = 5
    num_particles = 200
    num_iterations = 100
    w = 0.7
    c1 = 1.5
    c2 = 1.5

    theta = pso(lengths, *ball_pos, num_particles, num_iterations, w, c1, c2)
    prev_theta = np.zeros(len(lengths))
    temp_theta = np.zeros(len(lengths))
    perc = 0
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_title('Click to change ball position')
    ax.set_xlim(-sum(lengths)*1.1, sum(lengths)*1.1)
    ax.set_ylim(-sum(lengths)*1.1, sum(lengths)*1.1)
    fig.patch.set_facecolor('#110914')
    ax.set_facecolor('#110914')
    ax.title.set_color('white')
    ax.tick_params(colors='white')
    ax.spines['top'].set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')

    ground = ax.plot([-sum(lengths)*0.3, sum(lengths)*0.3], [0, 0], lw=1, c='white', alpha=0.5)[0]
    x, y = 0, 0
    lines = []
    for i, length in enumerate(lengths):
        next_x, next_y = x + length * np.cos(theta[i]), y + length * np.sin(theta[i])
        line, = ax.plot([x, next_x], [y, next_y], 'o-', lw=3, c="#caace2")
        lines.append(line)
        x, y = next_x, next_y
    
    ball = Circle(ball_pos, 1, color="#55cd97")
    ax.add_patch(ball)

    ani = FuncAnimation(fig, lambda f: update(f, ball, ball_pos, lines, lengths, theta, a), interval=1000/60, blit=True)
    fig.canvas.mpl_connect('button_press_event', lambda e: on_click(e, ax, ball_pos, theta))
    plt.show()
