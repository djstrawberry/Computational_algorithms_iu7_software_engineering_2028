# newton.py
import numpy as np

def divided_differences(x_vals, y_vals):
    n = len(x_vals)
    diff = np.zeros((n, n))
    diff[:, 0] = y_vals
    
    for j in range(1, n):
        for i in range(n - j):
            diff[i, j] = (diff[i+1, j-1] - diff[i, j-1]) / (x_vals[i+j] - x_vals[i])
    
    return diff[0]

def newton_interpolate_1d(x_vals, y_vals, x_target):

    n = len(x_vals)
    coeff = divided_differences(x_vals, y_vals)
    
    result = coeff[n-1]
    for i in range(n-2, -1, -1):
        result = result * (x_target - x_vals[i]) + coeff[i]
    
    return result

def newton_interpolate_3d(x, y, z, u, x0, y0, z0, nx, ny, nz):

    nx = min(nx, len(x) - 1)
    ny = min(ny, len(y) - 1)
    nz = min(nz, len(z) - 1)
    
    u_z = np.zeros(nz + 1)
    for i in range(nz + 1):
        u_zy = np.zeros(ny + 1)
        for j in range(ny + 1):
            x_vals = x[:nx + 1]
            y_vals = u[i][j][:nx + 1]
            u_zy[j] = newton_interpolate_1d(x_vals, y_vals, x0)
        
        y_vals = y[:ny + 1]
        u_z[i] = newton_interpolate_1d(y_vals, u_zy, y0)
    
    z_vals = z[:nz + 1]
    return newton_interpolate_1d(z_vals, u_z, z0)