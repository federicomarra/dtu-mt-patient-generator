### Tau_I halved, should the results and the TIR be better?

Short answer: no, not necessarily.

Halving tauI in this simulator does not mean "more insulin" or automatically "better control." It means insulin moves from the subcutaneous depot into plasma faster in the ODE at model.py. That changes timing, not total dose. The dataset summary still looks pretty much the same, that is believable.

The bigger reason is compensation. After tauI is set, the simulator recalibrates each patient’s basal, ICR, and ISF before running the 14-day Monte Carlo: see simulation.py. On top of that, the fasting steady-state algebra cancels tauI itself in model.py:

$$
S_{eq}=\tau_I u,\quad
I_{eq}=\frac{S_{eq}}{k_e \tau_I V_I BW}
=\frac{u}{k_e V_I BW}
$$

So baseline plasma insulin is basically unchanged by tauI. TauI mostly affects transient post-meal shape, not the long-run baseline.

There is also a safety stack that compresses differences: calibrated basal, hypo guard, and extra correction dosing with target are on at simulation_config.py. So if faster insulin creates a bit less hyper, the controller may just give less correction; if it creates more early hypo, guard/rescue may cancel that too. Net result: similar TIR, hypo, hyper.

If you want to see the real effect of tauI, the right checks are not the 14-day global averages. Use these instead:

1. Compare identical patients and identical seeds with and without recalibrating basal, ICR, and ISF.
2. Look at meal-window metrics: 0 to 4 hour postprandial peak, time-to-peak, and area above 10 mmol/L.
3. Track guard/rescue/correction event counts, because tauI may be changing controller behavior even if headline TIR barely moves.
