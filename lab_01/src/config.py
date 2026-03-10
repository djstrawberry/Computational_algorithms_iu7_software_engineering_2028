# TODO(change config to user input)

INTERPOLATION_DEGREES = {
    "I_t": 3,
    "Nh": 2,
    "sigma": 2,
    "c": 2,
    "q": 2
}

TX = 300.0  
PX = 40000.0    

T0 = 5400.0      
T_START = 14e-6    
T_END = 450e-6   
TAU = 1e-6        

P_MIN = 0.3
P_MAX = 2.5

R = 0.0025          
L = 0.012             

K_N = 7.242e4

EPS = 1e-6

def build_params():

    N_known = K_N / (TX * PX)

    return {
        "t0": T_START,
        "tk": T_END,
        "tau": TAU,

        "T0": T0,

        "degree": INTERPOLATION_DEGREES["I_t"],

        "N_known": N_known,

        "p_min": P_MIN,
        "p_max": P_MAX,

        "R": R,
        "l": L
    }