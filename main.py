import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, Circle, Arrow
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation

# potential to optimize: add 1,the penalty for the change between two states 2,the penalty for the intersection of the arms
def distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

# the minimum distance demanded maybe better than the intersection of the arms(more reasonable and practical)
def is_intersecting(p1, p2, p3, p4):
    def cross_product(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

    return (cross_product(p1, p2, p3) * cross_product(p1, p2, p4) < 0 and
            cross_product(p3, p4, p1) * cross_product(p3, p4, p2) < 0)

def check_self_collision(lengths, theta):
    n = len(lengths)
    x = np.cumsum(np.insert(lengths * np.cos(theta), 0, 0))
    y = np.cumsum(np.insert(lengths * np.sin(theta), 0, 0))

    for i in range(n):
        for j in range(i + 1, n):
            p1 = (x[i], y[i])
            p2 = (x[i+1], y[i+1])
            p3 = (x[j], y[j])
            p4 = (x[j+1], y[j+1])
            
            if is_intersecting(p1, p2, p3, p4):
                return True 
    return False

def fitness(lengths, xg, yg, theta,last_theta=None,lambda_=0.001):
    #1, the basic cost of distance
    x = np.sum(lengths * np.cos(theta), axis=1)
    y = np.sum(lengths * np.sin(theta), axis=1)
    res = distance(x, y, xg, yg)**2
    
    #2，the penalty for the change between two states
    if last_theta is not None:
        delta = (theta - last_theta + np.pi) % (2*np.pi) - np.pi
        res += lambda_*np.sum(np.abs(delta))

    #3，the penalty for the intersection of the arms
    if check_self_collision(lengths,theta):
        res += 1e10
    
    
    return res


def pso(lengths, xg, yg, num_particles=100, num_iterations=20, w=0.7, c1=1.5, c2=1.5, last_pos=None):

    num_dimensions = len(lengths)
    bounds = np.array([[0, 2*np.pi]] * len(lengths))
    #positions = np.random.uniform(bounds[:, 0], bounds[:, 1], (num_particles, num_dimensions))
    if last_pos is not None:
        positions = np.zeros((num_particles, num_dimensions))
        half = num_particles // 2
        positions[:half] = np.clip(last_pos + np.random.uniform(-0.5, 0.5, (half, num_dimensions)), 0, 2*np.pi)
        positions[half:] = np.random.uniform(bounds[:, 0], bounds[:, 1], (num_particles - half, num_dimensions))
    else:
        positions = np.random.uniform(bounds[:, 0], bounds[:, 1], (num_particles, num_dimensions))
    velocities = np.zeros((num_particles, num_dimensions))
    personal_best_positions = positions.copy()
    personal_best_scores = fitness(lengths, xg, yg, positions, last_theta=last_pos)
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
        scores = fitness(lengths, xg, yg, positions, last_theta=last_pos)
        mask = scores < personal_best_scores
        personal_best_scores[mask] = scores[mask]
        personal_best_positions[mask] = positions[mask]
        best_idx = np.argmin(personal_best_scores)
        if personal_best_scores[best_idx] < fitness(lengths, xg, yg, global_best_position.reshape(1, -1), last_theta=last_pos)[0]:
            global_best_position = personal_best_positions[best_idx].copy()
        
    return global_best_position



def update(frame, ball, ball_pos, lines, lengths, theta, a=50):
    x, y = 0, 0
    global perc
    global temp_theta
    perc += (1 - perc)/a
    delta = (theta - prev_theta + np.pi) % (2*np.pi) - np.pi
    # a basic but good method avoiding the problem of the angle overestimated
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
    #using the last states to penalty the change between two states
    last_stable_theta = theta.copy()
    perc = 0
    ball_pos[0] = event.xdata
    ball_pos[1] = event.ydata
    theta[:] = pso(lengths, ball_pos[0], ball_pos[1],last_pos=last_stable_theta)


if __name__ == '__main__':
    
    lengths = np.array([7, 3, 5, 4, 4])
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
    fig.patch.set_facecolor("#1C1E1D")
    ax.set_facecolor("#1A1B1B")
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
