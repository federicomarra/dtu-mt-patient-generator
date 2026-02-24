from sympy.strategies.core import switch

N_SCENARIOS = 5  # Number of different input scenarios (e.g., different meal times, insulin doses)


def ttm(h: int, m: int) -> int:
    """Convert hours and minutes to total minutes."""
    return h * 60 + m


def scenario_normal(t, b):
    u = b  # Start with basal insulin
    d = 0

    # Breakfast of 50g at t=8:00 for 15 min, 5U insulin
    if ttm(8, 0) <= t <= ttm(8, 15): d = 50 * 1000 / 15
    if ttm(8, 0) <= t <= ttm(8, 1): u += 5 * 1000

    # Snack of 20g at t=10:00 for 5 min, 2U insulin
    if ttm(10, 0) <= t <= ttm(10, 5): d = 20 * 1000 / 5
    if ttm(10, 0) <= t <= ttm(10, 1): u += 2 * 1000

    # Lunch of 70g at t=13:00 for 20 min, 7U insulin
    if ttm(13, 0) <= t <= ttm(13, 20): d = 70 * 1000 / 20
    if ttm(13, 0) <= t <= ttm(13, 1): u += 7 * 1000

    # Dinner of 40g at t=20:00 for 50 min, 4U insulin
    if ttm(20, 0) <= t <= ttm(20, 50): d = 40 * 1000 / 50
    if ttm(20, 0) <= t <= ttm(20, 1): u += 4 * 1000

    return u, d


def scenario_sedentary(t, b):
    u = b * 0.8 # Start with basal insulin
    d = 0

    # Breakfast of 60g at t=9:00 for 20 min, 5.5U insulin
    if ttm(9, 0) <= t <= ttm(9, 20): d = 60 * 1000 / 20
    if ttm(9, 0) <= t <= ttm(9, 1): u += 5.5 * 1000

    # Snack of 40g at t=11:00 for 10 min, 3.5U insulin
    if ttm(11, 0) <= t <= ttm(11, 10): d = 40 * 1000 / 10
    if ttm(11, 0) <= t <= ttm(11, 1): u += 3.5 * 1000

    # Lunch of 90g at t=13:00 for 20 min, 8.5U insulin
    if ttm(13, 30) <= t <= ttm(13, 50): d = 90 * 1000 / 20
    if ttm(13, 30) <= t <= ttm(13, 31): u += 8.5 * 1000

    # Dinner of 55g at t=20:45 for 1h, 5.5U insulin
    if ttm(20, 45) <= t <= ttm(21, 45): d = 55 * 1000 / 60
    if ttm(20, 45) <= t <= ttm(20, 1): u += 5. * 1000

    return u, d


def scenario_active(t, b):
    u = b * 1.2  # Start with basal insulin
    d = 0

    # Breakfast of 30g at t=7:00 for 5 min, 2U insulin
    if ttm(7, 0) <= t <= ttm(7, 5): d = 30 * 1000 / 5
    if ttm(7, 0) <= t <= ttm(7, 1): u += 2 * 1000

    # Run at t=11:00 for 50 min, (like 2.5U insulin)
    if ttm(11, 0) <= t <= ttm(11, 50): u += 2.5 * 1000

    # TODO: MORE HERE DOWN
    # Lunch of 90g at t=13:00 for 20 min, 8.5U insulin
    if ttm(13, 30) <= t <= ttm(13, 50): d = 90 * 1000 / 20
    if ttm(13, 30) <= t <= ttm(13, 31): u += 8.5 * 1000

    # Dinner of 55g at t=20:45 for 1h, 5.5U insulin
    if ttm(20, 45) <= t <= ttm(21, 45): d = 55 * 1000 / 60
    if ttm(20, 45) <= t <= ttm(20, 1): u += 5. * 1000

    return u, d


def scenario_missed_bolus(t, b):
    pass


def scenario_long_lunch(t, b):
    pass


def scenario_inputs(t, basal=0.5, scenario=1):
    """
    Defines what happens at time t.
    Time is in minutes. 0 to 1440 (24 hours).
    """
    # Basal Insulin (constant background)
    # Example input u is in mU/min. 0.5 U/hr = 500 mU / 60 min = 8.33
    b = basal * 1000 / 60  # Convert U/hr to mU/min

    if scenario == 1:
        return scenario_normal(t, b)
    elif scenario == 2:
        return scenario_active(t, b)
    elif scenario == 3:
        return scenario_sedentary(t, b)
    elif scenario == 4:
        return scenario_missed_bolus(t, b)
    elif scenario == 5:
        return scenario_long_lunch(t, b)
    else:
        print(f"Invalid scenario, defaulting to normal scenario.")
        return scenario_normal(t, b)


def scenario_inputs_gemini(t, scenario=1):
    """
    Defines what happens at time t.
    Time is in minutes. 0 to 1440 (24 hours).
    """
    # Basal Insulin (constant background)
    u_basal = 0.5  # U/hr -> converted to mU/min later if needed.
    # Let's assume input u is in mU/min. 0.5 U/hr = 500 mU / 60 min = 8.33
    u_val = 8.33

    # Meal at t=120 min (2 hours in)
    # 50g carbs eaten over 15 mins
    d_val = 0
    if 120 <= t <= 135:
        d_val = 50000 / 15  # mg/min (Total 50g)

    # Meal Bolus Insulin at t=120
    # 5 Units bolus delivered over 1 min
    if 120 <= t <= 121:
        u_val += 5000  # 5 U = 5000 mU

    return u_val, d_val
