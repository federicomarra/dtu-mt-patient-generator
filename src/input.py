N_SCENARIOS = 6  # Number of different input scenarios (e.g., different meal times, insulin doses)

def time_to_minutes(h: int, m: int) -> int:
    """Convert hours and minutes to total minutes."""
    return h * 60 + m

def minutes_to_time(minutes: int) -> str:
    """Convert total minutes to HH:MM format."""
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours:02d}:{mins:02d}"

BREAKFAST_CARBS_MIN = 30
BREAKFAST_CARBS_MAX = 60
BREAKFAST_TIME_MIN = time_to_minutes(7, 0)
BREAKFAST_TIME_MAX = time_to_minutes(9, 0)
BREAKFAST_DURATION_MIN = 10
BREAKFAST_DURATION_MAX = 15

LUNCH_CARBS_MIN = 50
LUNCH_CARBS_MAX = 100
LUNCH_TIME_MIN = time_to_minutes(12, 0)
LUNCH_TIME_MAX = time_to_minutes(14, 0)
LUNCH_DURATION_MIN = 25
LUNCH_DURATION_MAX = 30

DINNER_CARBS_MIN = 50
DINNER_CARBS_MAX = 80
DINNER_TIME_MIN = time_to_minutes(18, 0)
DINNER_TIME_MAX = time_to_minutes(21, 0)
DINNER_DURATION_MIN = 10
DINNER_DURATION_MAX = 15

SNACK_CARBS_MIN = 10
SNACK_CARBS_MAX = 30
SNACK_TIME_MIN = time_to_minutes(10, 0)
SNACK_TIME_MAX = time_to_minutes(11, 0)
SNACK_DURATION_MIN = 5
SNACK_DURATION_MAX = 10

PREANNOUNCED_BOLUS_TIME = 15
BOLUS_DURATION = 1


def scenario_normal(time: int, basal: float, insulin_sensitivity: float):
    """
    Simulates a normal daily routine by calculating insulin delivery (u) and carbohydrate intake (d) 
    at a specific time (t), incorporating randomized meals.
    """
    u = basal  # Start with basal insulin [mU/min]
    d = 0      # [mg/min]

    # Breakfast
    breakfast_time = np.random.randint(BREAKFAST_TIME_MIN, BREAKFAST_TIME_MAX)
    breakfast_duration = np.random.randint(BREAKFAST_DURATION_MIN, BREAKFAST_DURATION_MAX)
    breakfast_carbs = np.random.randint(BREAKFAST_CARBS_MIN, BREAKFAST_CARBS_MAX)
    breakfast_insulin = breakfast_carbs / insulin_sensitivity
    if breakfast_time <= time <= breakfast_time + breakfast_duration:
        d = breakfast_carbs * 1000 / breakfast_duration
    if breakfast_time - PREANNOUNCED_BOLUS_TIME <= time <= breakfast_time - PREANNOUNCED_BOLUS_TIME + BOLUS_DURATION:
        u += breakfast_insulin * 1000

    # Snack
    snack_time = np.random.randint(SNACK_TIME_MIN, SNACK_TIME_MAX)
    snack_duration = np.random.randint(SNACK_DURATION_MIN, SNACK_DURATION_MAX)
    snack_carbs = np.random.randint(SNACK_CARBS_MIN, SNACK_CARBS_MAX)
    snack_insulin = snack_carbs / insulin_sensitivity
    if snack_time <= time <= snack_time + snack_duration:
        d = snack_carbs * 1000 / snack_duration
    if snack_time - PREANNOUNCED_BOLUS_TIME <= time <= snack_time - PREANNOUNCED_BOLUS_TIME + BOLUS_DURATION:
        u += snack_insulin * 1000

    # Lunch
    lunch_time = np.random.randint(LUNCH_TIME_MIN, LUNCH_TIME_MAX)
    lunch_duration = np.random.randint(LUNCH_DURATION_MIN, LUNCH_DURATION_MAX)
    lunch_carbs = np.random.randint(LUNCH_CARBS_MIN, LUNCH_CARBS_MAX)
    lunch_insulin = lunch_carbs / insulin_sensitivity
    if lunch_time <= time <= lunch_time + lunch_duration:
        d = lunch_carbs * 1000 / lunch_duration
    if lunch_time - PREANNOUNCED_BOLUS_TIME <= time <= lunch_time - PREANNOUNCED_BOLUS_TIME + BOLUS_DURATION:
        u += lunch_insulin * 1000

    # Dinner
    dinner_time = np.random.randint(DINNER_TIME_MIN, DINNER_TIME_MAX)
    dinner_duration = np.random.randint(DINNER_DURATION_MIN, DINNER_DURATION_MAX)
    dinner_carbs = np.random.randint(DINNER_CARBS_MIN, DINNER_CARBS_MAX)
    dinner_insulin = dinner_carbs / insulin_sensitivity
    if dinner_time <= time <= dinner_time + dinner_duration:
        d = dinner_carbs * 1000 / dinner_duration
    if dinner_time - PREANNOUNCED_BOLUS_TIME <= time <= dinner_time - PREANNOUNCED_BOLUS_TIME + BOLUS_DURATION:
        u += dinner_insulin * 1000

    return u, d


def scenario_sedentary(time: int, basal: float, insulin_sensitivity: float):
    u = basal  # Start with basal insulin [mU/min]
    d = 0      # [mg/min]

    # Breakfast
    breakfast_time = np.random.randint(BREAKFAST_TIME_MIN, BREAKFAST_TIME_MAX)
    breakfast_duration = np.random.randint(BREAKFAST_DURATION_MIN, BREAKFAST_DURATION_MAX)
    breakfast_carbs = np.random.randint(BREAKFAST_CARBS_MIN, BREAKFAST_CARBS_MAX)
    breakfast_insulin = breakfast_carbs / insulin_sensitivity
    if breakfast_time <= time <= breakfast_time + breakfast_duration:
        d = breakfast_carbs * 1000 / breakfast_duration
    if breakfast_time - PREANNOUNCED_BOLUS_TIME <= time <= breakfast_time - PREANNOUNCED_BOLUS_TIME + BOLUS_DURATION:
        u += breakfast_insulin * 1000

    # Snack
    snack_time = np.random.randint(SNACK_TIME_MIN, SNACK_TIME_MAX)
    snack_duration = np.random.randint(SNACK_DURATION_MIN, SNACK_DURATION_MAX)
    snack_carbs = np.random.randint(SNACK_CARBS_MIN, SNACK_CARBS_MAX)
    snack_insulin = snack_carbs / insulin_sensitivity
    if snack_time <= time <= snack_time + snack_duration:
        d = snack_carbs * 1000 / snack_duration
    if snack_time - PREANNOUNCED_BOLUS_TIME <= time <= snack_time - PREANNOUNCED_BOLUS_TIME + BOLUS_DURATION:
        u += snack_insulin * 1000

    # Lunch
    lunch_time = np.random.randint(LUNCH_TIME_MIN, LUNCH_TIME_MAX)
    lunch_duration = np.random.randint(LUNCH_DURATION_MIN, LUNCH_DURATION_MAX)
    lunch_carbs = np.random.randint(LUNCH_CARBS_MIN, LUNCH_CARBS_MAX)
    lunch_insulin = lunch_carbs / insulin_sensitivity
    if lunch_time <= time <= lunch_time + lunch_duration:
        d = lunch_carbs * 1000 / lunch_duration
    if lunch_time - PREANNOUNCED_BOLUS_TIME <= time <= lunch_time - PREANNOUNCED_BOLUS_TIME + BOLUS_DURATION:
        u += lunch_insulin * 1000

    # Dinner
    dinner_time = np.random.randint(DINNER_TIME_MIN, DINNER_TIME_MAX)
    dinner_duration = np.random.randint(DINNER_DURATION_MIN, DINNER_DURATION_MAX)
    dinner_carbs = np.random.randint(DINNER_CARBS_MIN, DINNER_CARBS_MAX)
    dinner_insulin = dinner_carbs / insulin_sensitivity
    if dinner_time <= time <= dinner_time + dinner_duration:
        d = dinner_carbs * 1000 / dinner_duration
    if dinner_time - PREANNOUNCED_BOLUS_TIME <= time <= dinner_time - PREANNOUNCED_BOLUS_TIME + BOLUS_DURATION:
        u += dinner_insulin * 1000

    return u, d


def scenario_active(time: int, basal: float, insulin_sensitivity: float):
    u = basal  # Start with basal insulin [mU/min]
    d = 0      # [mg/min]

    # Breakfast
    breakfast_time = np.random.randint(BREAKFAST_TIME_MIN, BREAKFAST_TIME_MAX)
    breakfast_duration = np.random.randint(BREAKFAST_DURATION_MIN, BREAKFAST_DURATION_MAX)
    breakfast_carbs = np.random.randint(BREAKFAST_CARBS_MIN, BREAKFAST_CARBS_MAX)
    breakfast_insulin = breakfast_carbs / insulin_sensitivity
    if breakfast_time <= time <= breakfast_time + breakfast_duration:
        d = breakfast_carbs * 1000 / breakfast_duration
    if breakfast_time - PREANNOUNCED_BOLUS_TIME <= time <= breakfast_time - PREANNOUNCED_BOLUS_TIME + BOLUS_DURATION:
        u += breakfast_insulin * 1000

    # Snack
    snack_time = np.random.randint(SNACK_TIME_MIN, SNACK_TIME_MAX)
    snack_duration = np.random.randint(SNACK_DURATION_MIN, SNACK_DURATION_MAX)
    snack_carbs = np.random.randint(SNACK_CARBS_MIN, SNACK_CARBS_MAX)
    snack_insulin = snack_carbs / insulin_sensitivity
    if snack_time <= time <= snack_time + snack_duration:
        d = snack_carbs * 1000 / snack_duration
    if snack_time - PREANNOUNCED_BOLUS_TIME <= time <= snack_time - PREANNOUNCED_BOLUS_TIME + BOLUS_DURATION:
        u += snack_insulin * 1000

    # Lunch
    lunch_time = np.random.randint(LUNCH_TIME_MIN, LUNCH_TIME_MAX)
    lunch_duration = np.random.randint(LUNCH_DURATION_MIN, LUNCH_DURATION_MAX)
    lunch_carbs = np.random.randint(LUNCH_CARBS_MIN, LUNCH_CARBS_MAX)
    lunch_insulin = lunch_carbs / insulin_sensitivity
    if lunch_time <= time <= lunch_time + lunch_duration:
        d = lunch_carbs * 1000 / lunch_duration
    if lunch_time - PREANNOUNCED_BOLUS_TIME <= time <= lunch_time - PREANNOUNCED_BOLUS_TIME + BOLUS_DURATION:
        u += lunch_insulin * 1000

    # Dinner
    dinner_time = np.random.randint(DINNER_TIME_MIN, DINNER_TIME_MAX)
    dinner_duration = np.random.randint(DINNER_DURATION_MIN, DINNER_DURATION_MAX)
    dinner_carbs = np.random.randint(DINNER_CARBS_MIN, DINNER_CARBS_MAX)
    dinner_insulin = dinner_carbs / insulin_sensitivity
    if dinner_time <= time <= dinner_time + dinner_duration:
        d = dinner_carbs * 1000 / dinner_duration
    if dinner_time - PREANNOUNCED_BOLUS_TIME <= time <= dinner_time - PREANNOUNCED_BOLUS_TIME + BOLUS_DURATION:
        u += dinner_insulin * 1000

    return u, d


def scenario_missed_bolus(time: int, basal: float, insulin_sensitivity: float):
    u = basal  # Start with basal insulin [mU/min]
    d = 0      # [mg/min]

    missed_meal_id = np.random.randint(1, 4)

    # Breakfast
    breakfast_time = np.random.randint(BREAKFAST_TIME_MIN, BREAKFAST_TIME_MAX)
    breakfast_duration = np.random.randint(BREAKFAST_DURATION_MIN, BREAKFAST_DURATION_MAX)
    breakfast_carbs = np.random.randint(BREAKFAST_CARBS_MIN, BREAKFAST_CARBS_MAX)
    breakfast_insulin = breakfast_carbs / insulin_sensitivity
    if breakfast_time <= time <= breakfast_time + breakfast_duration:
        d = breakfast_carbs * 1000 / breakfast_duration
    if breakfast_time - PREANNOUNCED_BOLUS_TIME <= time <= breakfast_time - PREANNOUNCED_BOLUS_TIME + BOLUS_DURATION and missed_meal_id != 1:
        u += breakfast_insulin * 1000

    # Snack
    snack_time = np.random.randint(SNACK_TIME_MIN, SNACK_TIME_MAX)
    snack_duration = np.random.randint(SNACK_DURATION_MIN, SNACK_DURATION_MAX)
    snack_carbs = np.random.randint(SNACK_CARBS_MIN, SNACK_CARBS_MAX)
    snack_insulin = snack_carbs / insulin_sensitivity
    if snack_time <= time <= snack_time + snack_duration:
        d = snack_carbs * 1000 / snack_duration
    if snack_time - PREANNOUNCED_BOLUS_TIME <= time <= snack_time - PREANNOUNCED_BOLUS_TIME + BOLUS_DURATION and missed_meal_id != 2:
        u += snack_insulin * 1000

    # Lunch
    lunch_time = np.random.randint(LUNCH_TIME_MIN, LUNCH_TIME_MAX)
    lunch_duration = np.random.randint(LUNCH_DURATION_MIN, LUNCH_DURATION_MAX)
    lunch_carbs = np.random.randint(LUNCH_CARBS_MIN, LUNCH_CARBS_MAX)
    lunch_insulin = lunch_carbs / insulin_sensitivity
    if lunch_time <= time <= lunch_time + lunch_duration:
        d = lunch_carbs * 1000 / lunch_duration
    if lunch_time - PREANNOUNCED_BOLUS_TIME <= time <= lunch_time - PREANNOUNCED_BOLUS_TIME + BOLUS_DURATION and missed_meal_id != 3:
        u += lunch_insulin * 1000

    # Dinner
    dinner_time = np.random.randint(DINNER_TIME_MIN, DINNER_TIME_MAX)
    dinner_duration = np.random.randint(DINNER_DURATION_MIN, DINNER_DURATION_MAX)
    dinner_carbs = np.random.randint(DINNER_CARBS_MIN, DINNER_CARBS_MAX)
    dinner_insulin = dinner_carbs / insulin_sensitivity
    if dinner_time <= time <= dinner_time + dinner_duration:
        d = dinner_carbs * 1000 / dinner_duration
    if dinner_time - PREANNOUNCED_BOLUS_TIME <= time <= dinner_time - PREANNOUNCED_BOLUS_TIME + BOLUS_DURATION and missed_meal_id != 4:
        u += dinner_insulin * 1000

    return u, d

def scenario_late_bolus(time: int, basal: float, insulin_sensitivity: float):
    u = basal  # Start with basal insulin [mU/min]
    d = 0      # [mg/min]

    late_bolus_id = np.random.randint(1, 4)

    # Breakfast
    breakfast_time = np.random.randint(BREAKFAST_TIME_MIN, BREAKFAST_TIME_MAX)
    breakfast_duration = np.random.randint(BREAKFAST_DURATION_MIN, BREAKFAST_DURATION_MAX)
    breakfast_carbs = np.random.randint(BREAKFAST_CARBS_MIN, BREAKFAST_CARBS_MAX)
    breakfast_insulin = breakfast_carbs / insulin_sensitivity
    if breakfast_time <= time <= breakfast_time + breakfast_duration:
        d = breakfast_carbs * 1000 / breakfast_duration
    if late_bolus_id != 1 and breakfast_time - PREANNOUNCED_BOLUS_TIME <= time <= breakfast_time - PREANNOUNCED_BOLUS_TIME + BOLUS_DURATION:
        u += breakfast_insulin * 1000
    if late_bolus_id == 1 and breakfast_time <= time <= breakfast_time + BOLUS_DURATION:
        u += breakfast_insulin * 1000

    # Snack
    snack_time = np.random.randint(SNACK_TIME_MIN, SNACK_TIME_MAX)
    snack_duration = np.random.randint(SNACK_DURATION_MIN, SNACK_DURATION_MAX)
    snack_carbs = np.random.randint(SNACK_CARBS_MIN, SNACK_CARBS_MAX)
    snack_insulin = snack_carbs / insulin_sensitivity
    if snack_time <= time <= snack_time + snack_duration:
        d = snack_carbs * 1000 / snack_duration
    if late_bolus_id != 2 and snack_time - PREANNOUNCED_BOLUS_TIME <= time <= snack_time - PREANNOUNCED_BOLUS_TIME + BOLUS_DURATION:
        u += snack_insulin * 1000
    if late_bolus_id == 2 and snack_time <= time <= snack_time + BOLUS_DURATION:
        u += snack_insulin * 1000

    # Lunch
    lunch_time = np.random.randint(LUNCH_TIME_MIN, LUNCH_TIME_MAX)
    lunch_duration = np.random.randint(LUNCH_DURATION_MIN, LUNCH_DURATION_MAX)
    lunch_carbs = np.random.randint(LUNCH_CARBS_MIN, LUNCH_CARBS_MAX)
    lunch_insulin = lunch_carbs / insulin_sensitivity
    if lunch_time <= time <= lunch_time + lunch_duration:
        d = lunch_carbs * 1000 / lunch_duration
    if late_bolus_id != 3 and lunch_time - PREANNOUNCED_BOLUS_TIME <= time <= lunch_time - PREANNOUNCED_BOLUS_TIME + BOLUS_DURATION:
        u += lunch_insulin * 1000
    if late_bolus_id == 3 and lunch_time <= time <= lunch_time + BOLUS_DURATION:
        u += lunch_insulin * 1000

    # Dinner
    dinner_time = np.random.randint(DINNER_TIME_MIN, DINNER_TIME_MAX)
    dinner_duration = np.random.randint(DINNER_DURATION_MIN, DINNER_DURATION_MAX)
    dinner_carbs = np.random.randint(DINNER_CARBS_MIN, DINNER_CARBS_MAX)
    dinner_insulin = dinner_carbs / insulin_sensitivity
    if dinner_time <= time <= dinner_time + dinner_duration:
        d = dinner_carbs * 1000 / dinner_duration
    if late_bolus_id != 4 and dinner_time - PREANNOUNCED_BOLUS_TIME <= time <= dinner_time - PREANNOUNCED_BOLUS_TIME + BOLUS_DURATION:
        u += dinner_insulin * 1000
    if late_bolus_id == 4 and dinner_time <= time <= dinner_time + BOLUS_DURATION:
        u += dinner_insulin * 1000

    return u, d


def scenario_long_lunch(time: int, basal: float, insulin_sensitivity: float):
    u = basal  # Start with basal insulin [mU/min]
    d = 0      # [mg/min]

    # Breakfast
    breakfast_time = np.random.randint(BREAKFAST_TIME_MIN, BREAKFAST_TIME_MAX)
    breakfast_duration = np.random.randint(BREAKFAST_DURATION_MIN, BREAKFAST_DURATION_MAX)
    breakfast_carbs = np.random.randint(BREAKFAST_CARBS_MIN, BREAKFAST_CARBS_MAX)
    breakfast_insulin = breakfast_carbs / insulin_sensitivity
    if breakfast_time <= time <= breakfast_time + breakfast_duration:
        d = breakfast_carbs * 1000 / breakfast_duration
    if breakfast_time - PREANNOUNCED_BOLUS_TIME <= time <= breakfast_time - PREANNOUNCED_BOLUS_TIME + BOLUS_DURATION:
        u += breakfast_insulin * 1000

    # Snack
    snack_time = np.random.randint(SNACK_TIME_MIN, SNACK_TIME_MAX)
    snack_duration = np.random.randint(SNACK_DURATION_MIN, SNACK_DURATION_MAX)
    snack_carbs = np.random.randint(SNACK_CARBS_MIN, SNACK_CARBS_MAX)
    snack_insulin = snack_carbs / insulin_sensitivity
    if snack_time <= time <= snack_time + snack_duration:
        d = snack_carbs * 1000 / snack_duration
    if snack_time - PREANNOUNCED_BOLUS_TIME <= time <= snack_time - PREANNOUNCED_BOLUS_TIME + BOLUS_DURATION:
        u += snack_insulin * 1000

    # Lunch
    lunch_time = np.random.randint(LUNCH_TIME_MIN, LUNCH_TIME_MAX)
    lunch_duration = np.random.randint(LUNCH_DURATION_MIN, LUNCH_DURATION_MAX) * 3
    lunch_carbs = np.random.randint(LUNCH_CARBS_MIN, LUNCH_CARBS_MAX)
    lunch_insulin = lunch_carbs / insulin_sensitivity
    if lunch_time <= time <= lunch_time + lunch_duration:
        d = lunch_carbs * 1000 / lunch_duration
    if lunch_time - PREANNOUNCED_BOLUS_TIME <= time <= lunch_time - PREANNOUNCED_BOLUS_TIME + BOLUS_DURATION:
        u += lunch_insulin * 1000

    # Dinner
    dinner_time = np.random.randint(DINNER_TIME_MIN, DINNER_TIME_MAX)
    dinner_duration = np.random.randint(DINNER_DURATION_MIN, DINNER_DURATION_MAX)
    dinner_carbs = np.random.randint(DINNER_CARBS_MIN, DINNER_CARBS_MAX)
    dinner_insulin = dinner_carbs / insulin_sensitivity
    if dinner_time <= time <= dinner_time + dinner_duration:
        d = dinner_carbs * 1000 / dinner_duration
    if dinner_time - PREANNOUNCED_BOLUS_TIME <= time <= dinner_time - PREANNOUNCED_BOLUS_TIME + BOLUS_DURATION:
        u += dinner_insulin * 1000

    return u, d


def scenario_inputs(time: int, basal_hourly: float=0.5, scenario: int=1, insulin_sensitivity: float = 2.0):
    """
    Defines what happens at time t.
    Time is in minutes. 0 to 24*60=1440 (24 hours).
    """
    # Basal Insulin (constant background)
    # Example input u is in mU/min. 0.5 U/hr = 500 mU / 60 min = 8.33
    basal = basal_hourly * 1000 / 60  # Convert U/hr to mU/min

    if scenario == 1:
        return scenario_normal(time, basal, insulin_sensitivity)
    elif scenario == 2:
        return scenario_active(time, basal, insulin_sensitivity)
    elif scenario == 3:
        return scenario_sedentary(time, basal, insulin_sensitivity)
    elif scenario == 4:
        return scenario_missed_bolus(time, basal, insulin_sensitivity)
    elif scenario == 5:
        return scenario_late_bolus(time, basal, insulin_sensitivity)
    elif scenario == 6:
        return scenario_long_lunch(time, basal, insulin_sensitivity)
    else:
        print(f"Invalid scenario, defaulting to normal scenario.")
        return scenario_normal(time, basal, insulin_sensitivity)







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
