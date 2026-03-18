# spline.py
import numpy as np
from scipy import interpolate

def spline_interpolate_1d(x_vals, y_vals, x_target):
   
    spline = interpolate.CubicSpline(x_vals, y_vals, bc_type='natural')
    return spline(x_target)

def spline_interpolate_3d(x, y, z, u, x0, y0, z0):
   
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Транспонируем u для правильной размерности
    u_reshaped = u.transpose(2, 1, 0)  # [x][y][z]
    
    interp_3d = interpolate.RegularGridInterpolator(
        (x, y, z), 
        u_reshaped,
        method='cubic'
    )
    
    return interp_3d([x0, y0, z0])[0]