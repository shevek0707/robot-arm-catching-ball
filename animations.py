from manim import *
import numpy as np


def forward_kinematics(lengths: list[float], theta: list[float]):
    positions = [(0, 0, 0)]
    x, y = 0, 0
    for i in range(len(lengths)):
        next_x = x + lengths[i]*np.cos(theta[i])
        next_y = y + lengths[i]*np.sin(theta[i])
        x, y = next_x, next_y
        positions.append((x, y, 0))
    return positions

def move_arm(lengths, theta, joints: list[Dot]):
    positions = forward_kinematics(lengths, theta)
    return AnimationGroup([
        joints[i].animate.move_to(positions[i])
        for i in range(len(joints))
    ])

class SceneOne(Scene):
    def construct(self):
        # black screen
        self.wait(2)
        lengths = [1, 2, 0.5]
        theta = [PI, PI/4, 0]
        positions = forward_kinematics(lengths, theta)
        ground = Line(3*LEFT, 3*RIGHT, stroke_width=1,z_index=float('-inf'))
        joints = [Dot(pos) for pos in positions]
        parts = [
            always_redraw(lambda i=i: Line(joints[i].get_center(), joints[i+1].get_center(), z_index=float('-inf')))
            for i in range(len(joints)-1)
        ]
        # draw robot arm
        self.play(
            Create(VGroup(joints)), Create(VGroup(parts)), Create(ground)
        )
        self.wait(1)

        direction = lambda line: line.get_end() - line.get_start()
        length_labels = [
            always_redraw(lambda i=i: Tex(f'$l_{i+1}$', font_size=20).move_to(
                parts[i].get_center() + 0.2*rotate_vector(direction(parts[i])/np.linalg.norm(direction(parts[i])), PI/2)
            )) for i in range(len(parts))
        ]
        arrows_start = 4*RIGHT + 2*DOWN
        arrows_to_length_labels = [
            ArcBetweenPoints(arrows_start, parts[0].get_center(), stroke_width=2),
            Arrow(4*RIGHT+2*DOWN, parts[1].get_center(), stroke_width=2, tip_length=0.2, tip_width=0.2),
            ArcBetweenPoints(arrows_start, parts[2].get_center(), stroke_width=2)
        ]
        
        parts_label = Text('parts', font_size=40).next_to(4*RIGHT+2*DOWN)
        self.play(
            Create(VGroup(length_labels)), Create(VGroup(arrows_to_length_labels)), Create(parts_label)
        )
        theta = [PI/1.5, PI/3, -PI/2]
        self.play(move_arm(lengths, theta, joints))
        self.wait(1)
        theta = [-PI/3, -PI/4, PI]
        self.play(move_arm(lengths, theta, joints))


lengths = [1, 2, 0.5]
theta = [np.pi, np.pi/4, 0]
print(forward_kinematics(lengths, theta))