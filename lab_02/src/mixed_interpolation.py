# mixed.py
import numpy as np
from scipy import interpolate
from newton_interpolation import newton_interpolate_1d

def mixed_interpolate_3d(x, y, z, u, x0, y0, z0, ny_spline=True):
  
    if ny_spline:
        # Сплайн по y, полиномы по x и z
        # Интерполяция по x для фиксированных y и z
        u_x = np.zeros(len(x))
        for i in range(len(x)):
            # Для каждого x фиксируем y0 и интерполируем по y сплайном
            u_yz = np.zeros((len(z), len(y)))
            for k in range(len(z)):
                for j in range(len(y)):
                    u_yz[k][j] = u[k][j][i]
            
            # Интерполяция по y сплайном
            u_z = np.zeros(len(z))
            for k in range(len(z)):
                spline_y = interpolate.CubicSpline(y, u_yz[k])
                u_z[k] = spline_y(y0)
            
            # Интерполяция по z полиномом
            u_x[i] = newton_interpolate_1d(z, u_z, z0)
        
        # Интерполяция по x полиномом
        return newton_interpolate_1d(x, u_x, x0)
    
    else:
        # Сплайн по x, полиномы по y и z
        # Интерполяция по y для фиксированных x и z
        u_y = np.zeros(len(y))
        for j in range(len(y)):
            # Для каждого y фиксируем x0 и интерполируем по x сплайном
            u_xz = np.zeros((len(z), len(x)))
            for k in range(len(z)):
                for i in range(len(x)):
                    u_xz[k][i] = u[k][j][i]
            
            # Интерполяция по x сплайном
            u_z = np.zeros(len(z))
            for k in range(len(z)):
                spline_x = interpolate.CubicSpline(x, u_xz[k])
                u_z[k] = spline_x(x0)
            
            # Интерполяция по z полиномом
            u_y[j] = newton_interpolate_1d(z, u_z, z0)
        
        # Интерполяция по y полиномом
        return newton_interpolate_1d(y, u_y, y0)