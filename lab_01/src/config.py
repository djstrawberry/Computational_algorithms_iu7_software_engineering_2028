INTERPOLATION_DEGREES = {
    "I_t_degree": 2,
    "Nh_T_degree": 2,
    "Nh_p_degree": 2,
    "sigma_T_degree": 2,
    "sigma_p_degree": 2,
    "c_T_degree": 2,
    "c_p_degree": 2,
    "q_T_degree": 2,
    "q_p_degree": 2,
}

TX = 300.0  
PX = 0.04  

T0 = 5400.0      
T_START = 14e-6    
T_END = 450e-6   
TAU = 1e-6        

P_MIN = 0.3
P_MAX = 2.5

R = 0.25          
L = 0.012             

K_N = 7.242e4

EPS = 1e-6

def build_params():

    N_known = K_N * PX / TX

    return {
        "t0": T_START,
        "tk": T_END,
        "tau": TAU,

        "T0": T0,

        "N_known": N_known,

        "p_min": P_MIN,
        "p_max": P_MAX,

        "R": R,
        "l": L
    }