import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# 定义三个物体的初始位置、质量和速度
positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 1]], dtype='float64')  # 初始位置
masses = np.array([800000000, 20000000000, 1500000000])  # 质量
velocities = np.array([[0, 0.2, 0.9], [-0.1, 0, -0.1], [0.1, -0.2, 0]], dtype='float64')  # 初始速度
collision_distance = 0.1  # 碰撞检测距离阈值

# 设置模拟参数
G = 0.0000000000667430  # 引力05常数
dt = 0.05  # 时间步长
num_steps = 1000  # 模拟步数

# 创建3D画布
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-2, 2)

# 初始化物体的散点和轨迹线
scatters = [ax.plot([], [], [], 'o', markersize=8)[0] for _ in range(3)]
lines = [ax.plot([], [], [], lw=0.5)[0] for _ in range(3)]
trajectories = [[], [], []]
active = [True, True, True]  # 用于记录每个物体是否仍在运动

def init():
    for line in lines:
        line.set_data([], [])
        line.set_3d_properties([])
    for scatter in scatters:
        scatter.set_data([], [])
        scatter.set_3d_properties([])
    return lines + scatters

def compute_forces(positions, masses):
    num_bodies = len(masses)
    forces = np.zeros_like(positions)
    for i in range(num_bodies):
        for j in range(i + 1, num_bodies):
            if active[i] and active[j]:  # 只计算活跃物体间的引力
                r_vector = positions[j] - positions[i]
                distance = np.linalg.norm(r_vector)
                if distance == 0:
                    continue
                force_magnitude = G * masses[i] * masses[j] / distance**2
                force_direction = r_vector / distance
                force = force_magnitude * force_direction
                forces[i] += force
                forces[j] -= force
    return forces

def update(frame):
    global positions, velocities

    # 检测碰撞
    for i in range(3):
        for j in range(i + 1, 3):
            if active[i] and active[j]:
                distance = np.linalg.norm(positions[i] - positions[j])
                if distance < collision_distance:  # 检测到碰撞
                    active[i] = False
                    active[j] = False

    # 计算力并更新位置和速度
    forces = compute_forces(positions, masses)
    for i in range(3):
        if active[i]:  # 仅更新活跃的物体
            velocities[i] += forces[i] / masses[i] * dt
            positions[i] += velocities[i] * dt
            trajectories[i].append(positions[i].copy())

    # 更新散点和轨迹线
    for i in range(3):
        if active[i]:  # 显示活跃物体
            scatters[i].set_data(positions[i, 0], positions[i, 1])
            scatters[i].set_3d_properties(positions[i, 2])
            line_x, line_y, line_z = zip(*trajectories[i])
            lines[i].set_data(line_x, line_y)
            lines[i].set_3d_properties(line_z)
        else:  # 移除碰撞后消失的物体
            scatters[i].set_data([], [])
            scatters[i].set_3d_properties([])
            lines[i].set_data([], [])
            lines[i].set_3d_properties([])

    return lines + scatters

# 创建动画
ani = FuncAnimation(fig, update, frames=num_steps, init_func=init, blit=True, interval=20)
plt.show()
