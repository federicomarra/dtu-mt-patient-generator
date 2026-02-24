import numpy as np

def measure_glycemia(x, params, noise_std=0.0) -> float:
    """
    param sol:
    param params:
    param noise_std:
    return:
    """

    Q1 = x[0]           # [mmol] glucose in blood
    VG = params["VG"]   # [L / kg] glucose distribution volume
    BW = params["BW"]   # [kg] body weight

    # eq: 2.3: glucose concentration in mmol / L
    G = Q1 / (VG * BW)  # [mmol / L] = [mmol] * [kg / L] * [1 / kg]

    # Add noise to the sensor reading if specified
    if noise_std > 0:
        noise = np.random.normal(0, noise_std)
        G += noise
        #print(f"Added noise: {noise:.4f} mmol/L, Noisy Sensor Reading: {G[-1]:.4f} mmol/L")

    return G