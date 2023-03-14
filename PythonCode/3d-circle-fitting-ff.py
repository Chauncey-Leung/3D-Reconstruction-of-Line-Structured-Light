import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D

# 空间圆的参数方程P(t) = r*cos(t)*u + r*sin(t)*(n x u) + C
# 一下分别通过两组不同的参数生成空间圆


def generate_circle_by_vectors(t, C, r, n, u):
    n = n / np.linalg.norm(n)
    u = u / np.linalg.norm(u)
    P_circle = r * np.cos(t)[:, np.newaxis] * u + r * \
        np.sin(t)[:, np.newaxis] * np.cross(n, u) + C
    return P_circle


def generate_circle_by_angles(t, C, r, theta, phi):
    # n, u垂直，内积为0
    n = np.array([np.cos(phi) * np.sin(theta), np.sin(phi)
                 * np.sin(theta), np.cos(theta)])
    u = np.array([-np.sin(phi), np.cos(phi), 0])
    P_circle = r * np.cos(t)[:, np.newaxis] * u + r * \
        np.sin(t)[:, np.newaxis] * np.cross(n, u) + C
    return P_circle


def fit_circle_2d(x, y, w=[]):
    """
    最佳拟合平面内二维圆的拟合，
    (x-xc)^2 + (y-yc)^2 = r^2
    (2*xc)*x + (2*yc)*y + (r^2-xc^2-yc^2) = x^2+y^2
    c[0]*x + c[1]*y + c[2] = x^2+y^2
    :param x: 如上
    :param y: 如上
    :param w: 权重 不设置则等权重
    :return: (xc, yc) 圆心坐标  r 半径
    """
    A = np.array([x, y, np.ones(len(x))]).T
    b = x ** 2 + y ** 2
    # A = [x y 1], b = [x^2+y^2]
    # if len(w) == len(x):
    #     W = np.diag(w)
    #     A = np.dot(W, A)
    #     b = np.dot(W, b)
    # 最小二乘法求解 A*c = b,
    # c = argmin(||A*c - b||^2)
    #
    c = np.linalg.lstsq(A, b, rcond=None)[0]

    xc = c[0] / 2
    yc = c[1] / 2
    r = np.sqrt(c[2] + xc ** 2 + yc ** 2)
    return xc, yc, r


def rodrigues_rot(P, n0, n1):
    """
    P_rot = P*cos(theta) + (k × P)*sin(theta) + k*(k·P)*(1-cos(theta))
    :param P: 三维点坐标集合
    :param n0: 最佳拟合平面法向量
    :param n1: 原平面法向量
    :return P_rot: 旋转后的点集合
    """
    # 如果P是一维的，则扩展成矩阵方便后面运算
    if P.ndim == 1:
        P = P[np.newaxis, :]
    # 归一化
    n0 = n0 / np.linalg.norm(n0)
    n1 = n1 / np.linalg.norm(n1)
    # k为旋转轴
    k = np.cross(n0, n1)
    k = k / np.linalg.norm(k)
    # theta为旋转角度
    theta = np.arccos(np.dot(n0, n1))

    P_rot = np.zeros((len(P), 3))
    for i in range(len(P)):
        P_rot[i] = P[i] * np.cos(theta) + np.cross(k, P[i]) * \
            np.sin(theta) + k * np.dot(k, P[i]) * (1 - np.cos(theta))

    return P_rot


def angle_between(u, v, n=None):
    """
    得到向量u v的夹角
    """
    if n is None:
        return np.arctan2(np.linalg.norm(np.cross(u, v)), np.dot(u, v))
    else:
        return np.arctan2(np.dot(n, np.cross(u, v)), np.dot(u, v))


# def set_axes_equal_3d(ax):
#     limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
#     spans = abs(limits[:, 0] - limits[:, 1])
#     centers = np.mean(limits, axis=1)
#     radius = 0.5 * max(spans)
#     ax.set_xlim3d([centers[0] - radius, centers[0] + radius])
#     ax.set_ylim3d([centers[1] - radius, centers[1] + radius])
#     ax.set_zlim3d([centers[2] - radius, centers[2] + radius])

'''以下通过正态分布噪声模拟实际散点'''
# =============================================================================
# r = 2.5  # Radius
# C = np.array([3, 3, 4])  # Center
# theta = 45 / 180 * np.pi  # Azimuth 方位角
# phi = -30 / 180 * np.pi  # Zenith 天顶角
# t = np.linspace(0, 2 * np.pi, 100) # 返回均匀间隔的数据
# P_gen = generate_circle_by_angles(t, C, r, theta, phi)
# t = np.linspace(-np.pi, -0.25 * np.pi, 100)
# n = len(t)
# P = generate_circle_by_angles(t, C, r, theta, phi)
# P += np.random.normal(size=P.shape) * 0.1 # 加一个正态分布的噪声
# # 在棋盘格上找一个点
# =============================================================================
''''''''''''''''''''''''''

input_path = r"..\Image\RotationAxisCalibration\RotationAxisWorldPoint.txt"
P = np.loadtxt(input_path, skiprows=0)


fig = plt.figure(figsize=(15, 11))
alpha_pts = 0.5
figshape = (2, 3)
ax = [None] * 4

# 绘制坐标系、点云和理想圆
ax[0] = plt.subplot2grid(figshape, loc=(0, 0), colspan=2)
ax[1] = plt.subplot2grid(figshape, loc=(1, 0))
ax[2] = plt.subplot2grid(figshape, loc=(1, 1))
ax[3] = plt.subplot2grid(figshape, loc=(1, 2))
i = 0
ax[i].set_title('Fitting circle in 2D coords projected onto fitting plane')
ax[i].set_xlabel('x')
ax[i].set_ylabel('y')
ax[i].set_aspect('equal', 'datalim')  # 设置层次
ax[i].margins(.1, .1)
ax[i].grid()
i = 1
# ax[i].plot(P_gen[:, 0], P_gen[:, 1], 'y-', lw=3, label='Generating circle')
ax[i].scatter(P[:, 0], P[:, 1], alpha=alpha_pts, label='Cluster points P')
ax[i].set_title('View X-Y')
ax[i].set_xlabel('x')
ax[i].set_ylabel('y')
ax[i].set_aspect('equal', 'datalim')
ax[i].margins(.1, .1)
ax[i].grid()
i = 2
# ax[i].plot(P_gen[:, 0], P_gen[:, 2], 'y-', lw=3, label='Generating circle')
ax[i].scatter(P[:, 0], P[:, 2], alpha=alpha_pts, label='Cluster points P')
ax[i].set_title('View X-Z')
ax[i].set_xlabel('x')
ax[i].set_ylabel('z')
ax[i].set_aspect('equal', 'datalim')
ax[i].margins(.1, .1)
ax[i].grid()
i = 3
# ax[i].plot(P_gen[:, 1], P_gen[:, 2], 'y-', lw=3, label='Generating circle')
ax[i].scatter(P[:, 1], P[:, 2], alpha=alpha_pts, label='Cluster points P')
ax[i].set_title('View Y-Z')
ax[i].set_xlabel('y')
ax[i].set_ylabel('z')
ax[i].set_aspect('equal', 'datalim')
ax[i].margins(.1, .1)
ax[i].grid()
# plt.show()


# 空间坐标中心化
P_mean = P.mean(axis=0)
P_centered = P - P_mean
# SVD分解
U, s, V = np.linalg.svd(P_centered)

normal = V[2, :]  # V^T的第3列为最佳拟合平面的法向量
d = -np.dot(P_mean, normal)  # d = -<p,n>

# 将中心化坐标点投影到最佳拟合平面
P_xy = rodrigues_rot(P_centered, normal, [0, 0, 1])

ax[0].scatter(P_xy[:, 0], P_xy[:, 1], alpha=alpha_pts,
              label='Projected points')
# 生成散点？

# 通过最小二乘法在最佳拟合平面上拟合一个圆
xc, yc, r = fit_circle_2d(P_xy[:, 0], P_xy[:, 1])
# 生成二维点云
t = np.linspace(0, 2 * np.pi, 100)  # 这里为什么取100：采样率100
xx = xc + r * np.cos(t)
yy = yc + r * np.sin(t)

ax[0].plot(xx, yy, 'k--', lw=2, label='Fitting circle')
ax[0].plot(xc, yc, 'k+', ms=10)
ax[0].legend()

# 将二维圆坐标转化为三维坐,同时去中心化
C = rodrigues_rot(np.array([xc, yc, 0]), [0, 0, 1], normal) + P_mean
C = C.flatten()  # 降维到一维

# 生成拟合后的空间圆点云
t = np.linspace(0, 2 * np.pi, 100)
u = P[0] - C
P_fitcircle = generate_circle_by_vectors(t, C, r, normal, u)

# 绘制最佳拟合平面上的点，以及拟和的圆弧
ax[1].plot(P_fitcircle[:, 0], P_fitcircle[:, 1],
           'k--', lw=2, label='Fitting circle')
ax[2].plot(P_fitcircle[:, 0], P_fitcircle[:, 2],
           'k--', lw=2, label='Fitting circle')
ax[3].plot(P_fitcircle[:, 1], P_fitcircle[:, 2],
           'k--', lw=2, label='Fitting circle')
ax[3].legend()


# 生成拟合的弧
u = P[0] - C
v = P[-1] - C  # -1 最后一个
theta = angle_between(u, v, normal)

t = np.linspace(0, theta, 100)
P_fitarc = generate_circle_by_vectors(t, C, r, normal, u)

# 绘制拟合生成的空间圆
ax[1].plot(P_fitarc[:, 0], P_fitarc[:, 1], 'k-', lw=3, label='Fitting arc')
ax[2].plot(P_fitarc[:, 0], P_fitarc[:, 2], 'k-', lw=3, label='Fitting arc')
ax[3].plot(P_fitarc[:, 1], P_fitarc[:, 2], 'k-', lw=3, label='Fitting arc')
ax[1].plot(C[0], C[1], 'k+', ms=10)
ax[2].plot(C[0], C[2], 'k+', ms=10)
ax[3].plot(C[1], C[2], 'k+', ms=10)
ax[3].legend()
# plt.show()

# 相关三维绘图
fig3D = plt.figure()
ax3D = Axes3D(fig3D)
# set_axes_equal_3d(ax3D)
# ax3D.plot(P_gen[:, 0], P_gen[:, 1], P_gen[:, 2], 'y-', lw=3, label='Generating circle') # 用线拟合
ax3D.scatter(P[:, 0], P[:, 1], P[:, 2], alpha=alpha_pts,
             label='Cluster points P')  # 画散点图 点云
# ax3D.plot(P_fitarc[:, 0], P_fitcircle[:, 1], P_fitcircle[:, 2], 'k--', lw=2, label='Fitting circle')
ax3D.plot(P_fitarc[:, 0], P_fitarc[:, 1], P_fitarc[:, 2],
          'k-', lw=3, label='Fitting arc')  # k- 图例
plt.show()
print('Fitting plane: n = %s' % np.array_str(normal, precision=4))
print('Fitting circle: center = %s, r = %.4g' %
      (np.array_str(C, precision=4), r))
print('Fitting arc: u = %s, θ = %.4g' %
      (np.array_str(u, precision=4), theta * 180 / np.pi))

# 保存数据到txt
save = np.array([[C[0], C[1], C[2]],
                [normal[0], normal[1], normal[2]]],
                dtype=np.float64)
output_path = r"..\Image\RotationAxisCalibration\MATLABdata.txt"
np.savetxt(output_path, save, fmt='%1.8f')
