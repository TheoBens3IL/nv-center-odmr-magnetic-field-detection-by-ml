from matplotlib import pyplot as plt
import numpy as np

TlabNV = [
    np.array([
        [ 0.5,        -0.70710678,  0.5       ],
        [ 0.5,         0.70710678,  0.5       ],
        [-0.70710678,  0.0,         0.70710678]
    ]),
    np.array([
        [-0.5,         0.70710678, -0.5       ],
        [-0.5,        -0.70710678, -0.5       ],
        [-0.70710678,  0.0,         0.70710678]
    ]),
    
    np.array([
        [-0.5,         0.70710678,  0.5       ],
        [ 0.5,         0.70710678, -0.5       ],
        [-0.70710678,  0.0,        -0.70710678]
    ]),

    np.array([
        [ 0.5,        -0.70710678, -0.5       ],
        [-0.5,        -0.70710678,  0.5       ],
        [-0.70710678,  0.0,        -0.70710678]
    ])
    ]

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection='3d')

colors = ['green', 'blue', 'violet', 'red']

# --- function to draw a sphere ---
def draw_sphere(ax, center, radius, color, alpha=0.6):
    u = np.linspace(0, 2*np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = radius * np.outer(np.ones_like(u), np.cos(v)) + center[2]
    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0)

# --- draw origin sphere ---
draw_sphere(ax, center=[0,0,0], radius=0.07, color='black', alpha=1)

# --- draw NV vectors and tip spheres ---
for i, (T, c) in enumerate(zip(TlabNV, colors)):
    z_vec = 0.9 * T[:, 2]
    
    # vector
    ax.quiver(0, 0, 0,
              z_vec[0], z_vec[1], z_vec[2],
              color=c,
              linewidth=2,
              label=f'NV {i+1}',
              arrow_length_ratio=0.1)
    
    # sphere at tip
    draw_sphere(ax, center=z_vec, radius=0.08, color=c, alpha=0.8)
# --- Ejes coordenados en gris ---
axis_min = -0.05
axis_max = 1
# --- Ejes coordenados con punta ---
axis_min = -0.05
axis_max = 1

# X axis
ax.quiver(axis_min, 0, 0,
          axis_max-axis_min, 0, 0,
          color='gray',
          linewidth=1.5,
          arrow_length_ratio=0.05)

# Y axis
ax.quiver(0, axis_min, 0,
          0, axis_max-axis_min, 0,
          color='gray',
          linewidth=1.5,
          arrow_length_ratio=0.05)

# Z axis
ax.quiver(0, 0, axis_min,
          0, 0, axis_max-axis_min,
          color='gray',
          linewidth=1.5,
          arrow_length_ratio=0.05)
# --- Plano verde en z = 0 ---
xx, yy = np.meshgrid(np.linspace(-1, 1, 30),
                     np.linspace(-1, 1, 30))
zz = np.zeros_like(xx)

ax.plot_surface(xx, yy, zz,
                color='blue',
                alpha=0.1,
                linewidth=0)

for i, (T, c) in enumerate(zip(TlabNV, colors)):
    z_vec = 0.9 * T[:, 2]
    
    # vector
    ax.quiver(0, 0, 0,
              z_vec[0], z_vec[1], z_vec[2],
              color=c,
              linewidth=2,
              arrow_length_ratio=0.1)
    
    # sphere at tip
    draw_sphere(ax, center=z_vec, radius=0.08, color=c, alpha=0.8)
    
    # ---- label next to sphere ----
    offset = 0.08  # pequeño corrimiento
    ax.text(z_vec[0] + offset,
            z_vec[1] + offset,
            z_vec[2] + offset,
            f'NV{i+1}',
            color=c,
            fontsize=11)
    
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

# --- axis limits ---
ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
ax.set_zlim([-1,1])

ax.text(1.05, 0, 0, 'xlab', color='gray', fontsize=12)
ax.text(0, 1.05, 0, 'ylab', color='gray', fontsize=12)
ax.text(0, 0, 1.05, 'zlab', color='gray', fontsize=12)

ax.set_box_aspect([1,1,1])
# ax.view_init(elev=30, azim=60)



plt.show() 