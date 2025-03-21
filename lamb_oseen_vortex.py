import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import RegularGridInterpolator
import scipy.io as sio

desired_pixels = 256  
dpi = 100  
figsize = desired_pixels / dpi  
initial_t = 1.0

def lamb_oseen_vortex(x, y, t, Gamma, nu):
    """
    Calculate the velocity field of a Lamb-Oseen vortex, returning velocities in Cartesian coordinates

    Parameters:
    x : float or numpy.ndarray
        x-coordinate
    y : float or numpy.ndarray
        y-coordinate
    t : float
        time
    Gamma : float
        circulation of the vortex
    nu : float
        kinematic viscosity

    Returns:
    vx, vy : tuple of float or numpy.ndarray
        velocity components in x and y directions
    """

    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    g = 1 - np.exp(-r**2 / (4 * nu * t))

    vtheta = (Gamma / (2 * np.pi * r)) * g

    vr = np.zeros_like(r)

    vx = -vtheta * np.sin(theta)
    vy = vtheta * np.cos(theta)

    return vx, vy

def lamb_oseen_pressure(x, y, t, Gamma, nu, rho=1.0):
    """
    Calculate the pressure field of a Lamb-Oseen vortex

    Parameters:
        x, y : coordinates
        t : time
        Gamma : circulation
        nu : kinematic viscosity
        rho : density (default is 1.0)

    Returns:
        p : pressure field
        """

    r = np.sqrt(x**2 + y**2)
    
    r = np.maximum(r, 1e-10)
    
    g = 1 - np.exp(-r**2 / (4 * nu * t))
    
    vtheta = (Gamma / (2 * np.pi * r)) * g
    
    # p = p∞ + ρ∫(vθ²/r)dr
    p = -rho * (Gamma / (2 * np.pi))**2 / (4 * np.pi * r**2) * (
        1 - 2 * np.exp(-r**2 / (4 * nu * t)) + 
        np.exp(-r**2 / (2 * nu * t))
    )
    
    return p

def advect_scalar(scalar, u, v, dx, dy, dt):
    """
    Advect a scalar field using the semi-Lagrangian method with open boundary conditions
    """
    nx, ny = scalar.shape
    x, y = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    
    x_back = x - u * dt / dx
    y_back = y - v * dt / dy
    
    x_back = np.clip(x_back, 0, nx - 1)
    y_back = np.clip(y_back, 0, ny - 1)
    
    x0 = np.floor(x_back).astype(int)
    y0 = np.floor(y_back).astype(int)
    x1 = np.minimum(x0 + 1, nx - 1)
    y1 = np.minimum(y0 + 1, ny - 1)
    
    wx = x_back - x0
    wy = y_back - y0
    
    new_scalar = (scalar[x0, y0] * (1-wx) * (1-wy) +
                  scalar[x1, y0] * wx * (1-wy) +
                  scalar[x0, y1] * (1-wx) * wy +
                  scalar[x1, y1] * wx * wy)
    
    return new_scalar

def compute_vorticity(U, V, dx, dy):
    """
    Calculate the vorticity field
    """
    dV_dx = np.gradient(V, dx, axis=0)
    dU_dy = np.gradient(U, dy, axis=1)
    return dV_dx - dU_dy

nu = 1e-2  
dt = 0.01  
total_time = 1  
Gamma = 20*np.pi # Re = Gamma / (2*Pi*nu)

n = 256  
x = np.linspace(-np.pi, np.pi, n)
y = np.linspace(-np.pi, np.pi, n)
X, Y = np.meshgrid(x, y, indexing='ij')
dx = x[1] - x[0]
dy = y[1] - y[0]

x_pixel = np.linspace(-np.pi + (2*np.pi)/(2*desired_pixels), np.pi - (2*np.pi)/(2*desired_pixels), desired_pixels)
y_pixel = np.linspace(-np.pi + (2*np.pi)/(2*desired_pixels), np.pi - (2*np.pi)/(2*desired_pixels), desired_pixels)
X_pixel, Y_pixel = np.meshgrid(x_pixel, y_pixel, indexing='ij')

scalar = np.sin(X) * np.sin(Y)

dir = "lamb_oseen_vortex_Re=1000"
os.makedirs(dir, exist_ok=True)

velocity_frames = []
vorticity_frames = []

scalar_frames = []
t_star = []
x_star = x_pixel.reshape(-1, 1)
y_star = y_pixel.reshape(-1, 1)

for i in range(int(total_time/dt) + 1):
    t = i * dt + initial_t
    t_star.append(t)
    
    U, V = lamb_oseen_vortex(X, Y, t, Gamma, nu)
    
    vorticity = compute_vorticity(U, V, dx, dy)
    
    U_interp = RegularGridInterpolator((x, y), U, method='linear', bounds_error=False)
    V_interp = RegularGridInterpolator((x, y), V, method='linear', bounds_error=False)
    vorticity_interp = RegularGridInterpolator((x, y), vorticity, method='linear', bounds_error=False)

    points = np.stack((X_pixel, Y_pixel), axis=-1)
    U_pixel = U_interp(points)
    V_pixel = V_interp(points)
    vorticity_pixel = vorticity_interp(points)
    
    velocity_frames.append(np.stack((U_pixel, V_pixel), axis=-1))
    vorticity_frames.append(vorticity_pixel[:, :, np.newaxis])
    
    scalar = advect_scalar(scalar, U, V, dx, dy, dt)
    
    scalar_frames.append(scalar.reshape(-1, 1))
    
    fig, ax = plt.subplots(figsize=(figsize, figsize), dpi=dpi)
    im = ax.pcolormesh(X.T, Y.T, scalar.T, cmap='viridis', shading='gouraud')
    ax.set_aspect('equal')
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(f'{dir}/lamb_oseen_vortex_{i:03d}.png', 
                dpi=dpi, 
                bbox_inches='tight', 
                pad_inches=0)
    plt.close(fig)
    
    pressure = lamb_oseen_pressure(X, Y, t, Gamma, nu)
    
    if i == 0:
        pressure_frames = []
        u_frames = []
        v_frames = []
    pressure_frames.append(pressure.reshape(-1, 1))
    u_frames.append(U.reshape(-1, 1))
    v_frames.append(V.reshape(-1, 1))
    
velocity_data = np.stack(velocity_frames, axis=0)
vorticity_data = np.stack(vorticity_frames, axis=0)

np.save(f"{dir}/velocity_data.npy", velocity_data)
np.save(f"{dir}/vorticity_data.npy", vorticity_data)

c_star = np.hstack(scalar_frames)
p_star = np.hstack(pressure_frames)
u_star = np.hstack(u_frames)
v_star = np.hstack(v_frames)
t_star = np.array(t_star).reshape(-1, 1)

sio.savemat(f'{dir}/data.mat', {
    'x_star': x_star,
    'y_star': y_star,
    't_star': t_star,
    'C_star': c_star,
    'P_star': p_star,
    'U_star': u_star,
    'V_star': v_star,
})