import numpy as np

def get_base_params():
    """Returns standard parameters for a standard patient."""
    return {
        'MwG': 180.16,      # [mg/mmol] molecular weight of glucose

        'EGP0': 0.0161,     # [mmol/kg/min] endogenous glucose production
        'F01': 0.0097,      # [mmol/kg/min] non-insulin dependent flux

        'k12': 0.0649,      # [1/min] transfer rate
        'ka1': 0.0055,      # [1/min] deactivation rate for transport
        'ka2': 0.0683,      # [1/min] deactivation rate for disposal
        'ka3': 0.0304,      # [1/min] deactivation rate for EGP

        'SI1': 51.2,        # [L/min/mU] sensitivity transport
        'SI2': 8.2,         # [L/min/mU] sensitivity disposal
        'SI3': 520.0,       # [L/mU] sensitivity EGP

        'kb1': 0.0055 * 51.2,   # [min^-2/(mU/L)] sensitivity transport
        'kb2': 0.0683 * 8.2,    # [min^-2/(mU/L)] sensitivity disposal
        'kb3': 0.0304 * 520,    # [min^-1/(mU/L)] sensitivity EGP

        'ke': 0.138,        # [1/min] elimination rate

        'VI': 0.12,         # [L/kg] insulin distribution volume
        'VG': 0.1484,       # [L/kg] glucose distribution volume

        'tauI': 55.871,    # [min] absorption time constant
        'tauG': 39.908,    # [min] glucose absorption time

        'Ag': 0.7943,       # [1] glucose absorption rate
        'BW': 80            # [kg] Body weight
    }


def generate_monte_carlo_patients(n=10):
    """
    Generates N patients by perturbing sensitivity parameters
    using a Log-Normal distribution (prevents negative values).
    """
    patients = []
    base = get_base_params()

    for _ in range(n):
        p = base.copy()
        # Randomize Insulin Sensitivities (kb1, kb2, kb3)
        # CV of 30% is typical for inter-patient variability
        p['kb1'] = np.random.lognormal(np.log(base['kb1']), 0.3)
        p['kb2'] = np.random.lognormal(np.log(base['kb2']), 0.3)
        p['kb3'] = np.random.lognormal(np.log(base['kb3']), 0.3)

        # Glucose Parameters (Normal Distributions)
        p['EGP0'] = np.random.normal(0.0161, 0.0039)
        p['F01'] = np.random.normal(0.0097, 0.0022)
        p['k12'] = np.random.normal(0.0649, 0.0282)

        # Activation rates (ka)
        p['ka1'] = np.random.normal(0.0055, 0.0056)
        p['ka2'] = np.random.normal(0.0683, 0.0507)
        p['ka3'] = np.random.normal(0.0304, 0.0235)

        # Insulin Sensitivities (SI)
        p['SI1'] = np.random.normal(51.2, 32.09)
        p['SI2'] = np.random.normal(8.2, 7.84)
        p['SI3'] = np.random.normal(520.0, 306.2)

        # Elimination and Volumes
        p['ke'] = np.random.normal(0.14, 0.035)
        p['VI'] = np.random.normal(0.12, 0.012)

        # VG is derived from: exp(VG) ~ N(1.16, 0.23^2)
        # We sample the normal distribution first, then take the log
        p['VG'] = np.log(np.random.normal(1.16, 0.23))

        # Time constants (tau)
        # tau_I is derived from: 1/tau_I ~ N(0.018, 0.0045^2)
        p['tauI'] = 1 / np.random.normal(0.018, 0.0045)

        # tau_G is derived from: ln(tau_G) ~ N(3.689, 0.25^2)
        p['tauG'] = np.exp(np.random.normal(3.689, 0.25))

        # Carbohydrate Bioavailability and Body Weight (Uniform Distributions)
        p['Ag'] = np.random.uniform(0.7, 1.2)
        p['BW'] = np.random.uniform(65, 95)

        patients.append(p)
    return patients

def get_base_params_gemini(bw=70):
    """Returns standard parameters for a 70kg patient."""
    return {
        'MwG': 180.16,  # mg/mmol
        'BW': bw,
        # Glucose params
        'EGP0': 16.9,  # endogenous glucose production
        'F01': 9.7,  # non-insulin dependent flux
        'k12': 0.066,  # transfer rate
        'VG': 0.16 * bw,  # distribution volume
        # Insulin Sensitivity Parameters (These are often randomized)
        'kb1': 14.6e-4 * 60,  # sensitivity transport
        'kb2': 8.2e-4 * 60,  # sensitivity disposal
        'kb3': 52e-5 * 60,  # sensitivity EGP
        'ka1': 0.006 * 60,  # deactivation rate
        'ka2': 0.06 * 60,
        'ka3': 0.03 * 60,
        # Insulin Kinetics
        'ke': 0.138,  # elimination
        'VI': 0.12 * bw,  # insulin volume
        'tauI': 55,  # absorption time constant
        # Meal Kinetics
        'tauG': 40,  # meal absorption time
        'Ag': 0.8  # bioavailability
    }

def generate_monte_carlo_patients_gemini(n=10):
    """
    Generates N patients by perturbing sensitivity parameters
    using a Log-Normal distribution (prevents negative values).
    """
    patients = []
    base = get_base_params_gemini()

    for _ in range(n):
        p = base.copy()
        # Randomize Insulin Sensitivities (kb1, kb2, kb3)
        # CV of 30% is typical for inter-patient variability
        p['kb1'] = np.random.lognormal(np.log(base['kb1']), 0.3)
        p['kb2'] = np.random.lognormal(np.log(base['kb2']), 0.3)
        p['kb3'] = np.random.lognormal(np.log(base['kb3']), 0.3)

        # Randomize Meal Absorption time slightly
        p['tauG'] = np.random.normal(base['tauG'], 5)

        patients.append(p)
    return patients