"""
two_body_3d.py
3-D two-body orbit integrator & animator
Author:  You
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import tqdm   # pip install tqdm (optional)

# ------------------ physical parameters ------------------
G  = 1.0
m1 = 1.0
m2 = 0.5
M  = m1 + m2
mu = m1*m2/M

# ------------------ initial conditions -------------------
# choose a bound orbit: a = 1, e = 0.4, inclined 30°
a  = 1.0
e  = 0.4
h  = np.sqrt(G*M*a*(1-e**2))          # specific ang. mom. magnitude
inc = np.radians(30)                 # inclination
Omega = 0.0                          # RAAN
omega = 0.0                          # argument of periapsis

# classical orbital elements → state vector
# helper: rotation matrices
def Rz(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])
def Ry(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])

# perifocal frame position & velocity at periapsis
r0_pqw = np.array([a*(1-e), 0, 0])
v0_pqw = np.array([0, np.sqrt(G*M*(1+e)/(a*(1-e))), 0])

# rotate to inertial frame (J2000-equatorial for demo)
R = Rz(-Omega) @ Ry(-inc) @ Rz(-omega)
r_rel = R @ r0_pqw
v_rel = R @ v0_pqw

# centre-of-mass frame velocities
v1 =  (m2/M) * v_rel
v2 = -(m1/M) * v_rel

# absolute positions (CM at origin)
x1 = - (m2/M) * r_rel
x2 =   (m1/M) * r_rel

# -------------------- integrator -------------------------
dt      = 0.01
t_max   = 8*np.pi*np.sqrt(a**3/(G*M))   # 4 periods
steps   = int(t_max/dt)

x1_hist, x2_hist = [], []

def accel(r):
    r_norm = np.linalg.norm(r)
    return -G*M * r / r_norm**3

def verlet_step(x1, x2, v1, v2):
    r12 = x2 - x1
    a1 =  (m2/M) * accel(r12)
    a2 = -(m1/M) * accel(r12)

    x1 += v1*dt + 0.5*a1*dt**2
    x2 += v2*dt + 0.5*a2*dt**2

    r12_new = x2 - x1
    a1_new  =  (m2/M) * accel(r12_new)
    a2_new  = -(m1/M) * accel(r12_new)

    v1 += 0.5*(a1 + a1_new)*dt
    v2 += 0.5*(a2 + a2_new)*dt
    return x1, x2, v1, v2

print("Integrating …")
for _ in tqdm(range(steps)):
    x1, x2, v1, v2 = verlet_step(x1, x2, v1, v2)
    x1_hist.append(x1.copy())
    x2_hist.append(x2.copy())

x1_hist = np.array(x1_hist)
x2_hist = np.array(x2_hist)
d= np.linalg.norm(x2_hist - x1_hist, axis=1)
print(f"Max separation: {np.max(d):.3f} (should be ~{a*(1+e):.3f})")
print(f"Min separation: {np.min(d):.3f} (should be ~{a*(1-e):.3f})")
# ------------------ 3-D animation ------------------------
fig = plt.figure(figsize=(7,7))
ax  = fig.add_subplot(111, projection='3d')
trail_len = 150
ax.set_xlim(-1.5*a, 1.5*a)
ax.set_ylim(-1.5*a, 1.5*a)
ax.set_zlim(-1.5*a, 1.5*a)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3-D Two-Body Orbit")

line1, = ax.plot([], [], [], lw=1, color='tab:blue')
line2, = ax.plot([], [], [], lw=1, color='tab:orange')
dot1,  = ax.plot([], [], [], 'o', color='tab:blue', ms=8)
dot2,  = ax.plot([], [], [], 'o', color='tab:orange', ms=6)

def init():
    line1.set_data([], [])
    line2.set_data([], [])
    dot1.set_data([], [])
    dot2.set_data([], [])
    line1.set_3d_properties([])
    line2.set_3d_properties([])
    dot1.set_3d_properties([])
    dot2.set_3d_properties([])
    return line1, line2, dot1, dot2

def update(frame):
    start = 0
    # trails
    line1.set_data(x1_hist[start:frame,0], x1_hist[start:frame,1])
    line1.set_3d_properties(x1_hist[start:frame,2])
    line2.set_data(x2_hist[start:frame,0], x2_hist[start:frame,1])
    line2.set_3d_properties(x2_hist[start:frame,2])
    # dots
    dot1.set_data([x1_hist[frame,0]], [x1_hist[frame,1]])
    dot1.set_3d_properties([x1_hist[frame,2]])
    dot2.set_data([x2_hist[frame,0]], [x2_hist[frame,1]])
    dot2.set_3d_properties([x2_hist[frame,2]])
    return line1, line2, dot1, dot2

ani = FuncAnimation(fig, update, frames=steps,
                    init_func=init, blit=True, interval=20)

# ------------------ save or show -------------------------
# Uncomment ONE line below to save (mp4 or gif):
# ani.save("two_body_3d.mp4", writer="ffmpeg", fps=30)
ani.save("two_body_3d.gif", writer=PillowWriter(fps=30))

plt.show()