# Virtual Patient Simulator: Mathematical Equations

This file consolidates the equations implemented across the simulator codebase.

Primary implementation sources:

- src/model.py
- src/hovorka_exercise.py
- src/input.py
- src/simulation_control.py
- src/sensor.py
- src/simulation_utils.py
- src/sensitivity.py
- src/parameters.py

## 1) State Vector And Units

The integrated state is:

$$
x = [Q_1, Q_2, S_1, S_2, I, x_1, x_2, x_3, D_1, D_2, Y, Z, rGU, rGP, tPA, PAint, rdepl, th]
$$

where:

- $Q_1, Q_2$: glucose masses [mmol]
- $S_1, S_2$: subcutaneous insulin depots [mU]
- $I$: plasma insulin concentration [mU/L]
- $x_1, x_2, x_3$: insulin action states
- $D_1, D_2$: gut glucose compartments [mmol]
- $Y$: short-term PA insulin sensitivity accumulator [count·min]
- $Z$: long-term post-exercise SI elevation [count·min]
- $rGU$: exercise glucose utilization rate [1/min]
- $rGP$: exercise glucose production rate [1/min]
- $tPA$: PA tracking state (1=active, 0=rest) [-]
- $PAint$: cumulative PA intensity integral [count·min]
- $rdepl$: glycogen depletion rate [1/min]
- $th$: high-intensity duration accumulator (drives aerobic→anaerobic transition) [-]

## 2) Input Channels

Per minute, the input layer provides:

- $U(t)$: insulin delivery [mU/min]
- $D_{mg}(t)$: carbohydrate intake [mg/min]
- $AC(t)$: accelerometer counts [count/min] — exercise intensity input to the ETH model

Carbohydrate conversion used in model:

$$
D(t) = \frac{D_{mg}(t)}{M_w^G}
$$

with $M_w^G$ in [mg/mmol], so $D(t)$ is [mmol/min].

Basal conversion:

$$
U_{basal} = \frac{U_{hr} \cdot 1000}{60}
$$

where $U_{hr}$ is [U/hr], $U_{basal}$ is [mU/min].

## 3) Hovorka Core ODEs

Let:

$$
G = \frac{Q_1}{V_G \cdot BW}
$$

### 3.1 Gut absorption

$$
\dot D_1 = A_g D - \frac{1}{\tau_G}D_1
$$

$$
\dot D_2 = \frac{1}{\tau_G}(D_1 - D_2)
$$

$$
U_G = \frac{1}{\tau_G}D_2
$$

### 3.2 Insulin absorption/plasma insulin

$$
\dot S_1 = U - \frac{1}{\tau_I}S_1
$$

$$
\dot S_2 = \frac{1}{\tau_I}(S_1 - S_2)
$$

$$
U_I = \frac{1}{\tau_I}S_2
$$

$$
\dot I = \frac{U_I}{V_I \cdot BW} - k_e I
$$

### 3.3 Non-insulin glucose utilization and renal clearance

$$
F_{01}^c =
\begin{cases}
F_{01} \cdot BW, & G \ge 4.5 \\
\max\left(0, F_{01} \cdot BW \cdot \frac{\max(0,G)}{4.5}\right), & G < 4.5
\end{cases}
$$

$$
F_R =
\begin{cases}
0.003\,(G-9.0)\,V_G\,BW, & G \ge 9.0 \\
0, & G < 9.0
\end{cases}
$$

### 3.4 Insulin action states

$$
k_{b1} = SI_1 k_{a1}, \quad k_{b2} = SI_2 k_{a2}, \quad k_{b3} = SI_3 k_{a3}
$$

$$
\dot x_1 = k_{b1}I - k_{a1}x_1
$$

$$
\dot x_2 = k_{b2}I - k_{a2}x_2
$$

$$
\dot x_3 = k_{b3}I - k_{a3}x_3
$$

### 3.5 Glucose compartments (before exercise terms)

The implemented transfer form is:

$$
R_{12} = x_1 Q_1 - k_{12}Q_2, \quad R_2 = x_2Q_2
$$

Note: in this implementation, $R_{12}$ is a **net inter-compartment transfer term** (not a one-way flux).

$$
EGP_c = EGP_0 \cdot BW \cdot \max(0, 1 - x_3)
$$

## 4) Exercise Extension (ETH Deichmann)

Based on: Deichmann et al., PLOS Comput Biol 2023. T1D-validated on aerobic parameters (5 Basel children). Anaerobic parameters (marked EXPERIMENTAL) use population values from params_standard.csv.

Input: $AC(t)$ — accelerometer counts at minute $t$.

### 4.1 Sigmoidal transfer functions

$$
f_Y = \frac{(Y/a_Y)^{n_1}}{1+(Y/a_Y)^{n_1}}, \quad
f_{AC} = \frac{(AC/a_{AC})^{n_2}}{1+(AC/a_{AC})^{n_2}}, \quad
f_{HI} = \frac{(AC/a_h)^{n_2}}{1+(AC/a_h)^{n_2}}
$$

$$
f_p = \frac{(th/\tau_p)^{n_2}}{1+(th/\tau_p)^{n_2}}
$$

Glycogen depletion fraction (when $tPA>0$ and $PAint>0$):

$$
\bar{I}_{avg} = \frac{PAint}{tPA}, \quad
t_{depl} = \max(10^{-3},\; -a_{depl}\bar{I}_{avg}+b_{depl})
$$

$$
f_t = \frac{(tPA/t_{depl})^{n_1}}{1+(tPA/t_{depl})^{n_1}}
$$

### 4.2 Intensity-blended rGP parameters

$$
q_3 = (1-f_p)\,q_{3l} + f_p\,q_{3h}, \quad q_4 = (1-f_p)\,q_{4l} + f_p\,q_{4h}
$$

where subscript $l$ = aerobic (T1D-validated), $h$ = anaerobic (EXPERIMENTAL).

### 4.3 State derivatives

$$
\dot Y = -\frac{1}{\tau_{AC}}Y + \frac{1}{\tau_{AC}}AC
$$

$$
\dot Z = b\,f_Y Y - \frac{1-f_Y}{\tau_Z}Z
$$

$$
\dot{rGU} = q_1 f_Y Y - q_2\,rGU
$$

$$
\dot{rGP} = q_3 f_Y Y - q_4\,rGP
$$

$$
\dot{tPA} = f_{AC} - (1-f_{AC})\,tPA
$$

$$
\dot{PAint} = f_{AC}\cdot AC - (1-f_{AC})\,PAint
$$

$$
\dot{rdepl} = q_6\,(f_t\,rGP - rdepl)
$$

$$
\dot{th} = f_{HI} - (1-f_{HI})\,q_5\,th
$$

### 4.4 Q1 interaction terms

$$
\text{exercise\_uptake} = rGU \cdot Q_1
$$

$$
\text{exercise\_prod} = \max(0,\; rGP-rdepl)\cdot Q_1
$$

$$
\text{exercise\_si} = Z \cdot x_1 \cdot Q_1
$$

These modify glucose dynamics:

$$
\dot Q_1 = U_G + EGP_c - R_{12} - F_{01}^c - F_R - \text{exercise\_uptake} + \text{exercise\_prod} - \text{exercise\_si}
$$

$$
\dot Q_2 = R_{12} - R_2
$$

## 5) Steady-State Initialization

### 5.1 Fasting state from basal insulin

Given basal insulin $u_{\mu}$ [mU/min], the implementation enforces fasting equilibrium assumptions:

$$
\dot S_1=\dot S_2=\dot I=\dot x_1=\dot x_2=\dot x_3=\dot D_1=\dot D_2=0,\quad \dot Y=\dot Z=\dot{rGU}=\dot{rGP}=\dot{tPA}=\dot{PAint}=\dot{rdepl}=\dot{th}=0
$$

$$
D_1=D_2=Y=Z=rGU=rGP=tPA=PAint=rdepl=th=0 \quad (AC(t)=0 \text{ at fasting})
$$

and computes:

$$
S_{eq} = \tau_I u_{\mu}
$$

$$
I_{eq} = \frac{S_{eq}}{k_e\tau_I V_I BW} = \frac{u_{\mu}}{k_e V_I BW}
$$

$$
x_{1,eq}=SI_1I_{eq},\quad x_{2,eq}=SI_2I_{eq},\quad x_{3,eq}=SI_3I_{eq}
$$

With:

$$
F_{01}^c = F_{01}BW,
\quad
EGP_c = EGP_0BW\max(0,1-x_{3,eq})
$$

Assumption for this initialization step: fasting-like baseline around normoglycemia, so renal excretion is inactive ($F_R\approx 0$) and the non-insulin-dependent utilization uses its non-hypoglycemic branch ($F_{01}^c=F_{01}BW$).

Solve linear system for $(Q_{1,eq},Q_{2,eq})$:

$$
\begin{bmatrix}
-x_{1,eq} & k_{12} \\
x_{1,eq} & -(k_{12}+x_{2,eq})
\end{bmatrix}
\begin{bmatrix}Q_{1,eq} \\ Q_{2,eq}\end{bmatrix}
=
\begin{bmatrix}F_{01}^c-EGP_c \\ 0\end{bmatrix}
$$

Numerically, if the $2\times2$ system is singular, least-squares is used as fallback, and the final values are clipped to nonnegative:

$$
Q_{1,eq}=\max(0,Q_{1,eq}),\quad Q_{2,eq}=\max(0,Q_{2,eq})
$$

### 5.2 Target-glucose solve (full multivariate Newton)

The implemented unknown vector is:

$$
z = [x_1,\ldots,x_{18},u]^T
$$

where $x=[Q_1,Q_2,S_1,S_2,I,x_1,x_2,x_3,D_1,D_2,Y,Z,rGU,rGP,tPA,PAint,rdepl,th]$ and $u$ is basal insulin input (mU/min).

The nonlinear root system is:

$$
F(z)=
\begin{bmatrix}
f_{Hovorka+ETH}(x,u,p) \\
G(x)-G_{target}
\end{bmatrix}=0
$$

where $f_{Hovorka+ETH}(x,u,p)$ is the 18-state ODE right-hand side evaluated at steady-state inputs.

**Notation note on $p$:** $p$ denotes the fixed patient-specific model parameters (e.g. $V_G$, $k_{12}$, $SI_1$, ...). They are written explicitly in $f(x,u,p)$, following standard dynamical-systems convention, to make clear what the function depends on, but they are *not* unknowns of the Newton system. The Newton iteration solves only for $z=[x,u]$; $p$ remains constant throughout.

Steady-state inputs used during the solve:

$$
D(t)=0,\quad AC(t)=0
$$

At fasting ($AC=0$), all ETH states are identically zero and their ODEs decouple from the Hovorka core. The 18-state Jacobian block for ETH states reduces to a negative-diagonal matrix of decay rates, which is well-conditioned and does not degrade Newton convergence.

and $G(x)=Q_1/(V_GBW)$.

Implementation architecture is method-general via pluggable callbacks:

- residual $(z,p)\rightarrow F\in\mathbb{R}^{19}$
- jacobian $(z,F,p)\rightarrow J\in\mathbb{R}^{19\times19}$
- project $(z)\rightarrow z_{constrained}$
- observable $(z,p)\rightarrow G(z)$

where, project(z) is a constraint enforcement callback in the Newton solver that clips the solution vector to physically valid ranges after each iteration.

This keeps the Newton workflow reusable across models (for example, a future UVA/Padova adapter) while using Hovorka-specific callbacks today.

Newton iteration:

$$
J(z_k)\,\Delta z_k = -F(z_k),\quad z_{k+1}=z_k+\lambda\Delta z_k
$$

Two Jacobian options are supported conceptually:

Analytical Jacobian:

$$
J_{ij}(z)=\frac{\partial F_i}{\partial z_j}
$$

Numerical Jacobian (forward finite differences):

$$
\frac{\partial F}{\partial z_j}\approx\frac{F(z+h_j e_j)-F(z)}{h_j},
\quad
h_j=\max(10^{-6},10^{-4}\max(1,|z_j|))
$$

Choice for this codebase:

- We use the analytical Jacobian for Hovorka (faster and more stable for a fixed, known model).
- Numerical Jacobian remains the model-general fallback strategy when model equations are frequently changing or when derivatives are not yet derived.

If $J$ is singular or ill-conditioned, least-squares is used instead of direct solve.

Damping/backtracking factors:

$$
\lambda \in \{1,0.5,0.25,0.125,0.0625\}
$$

Acceptance metric:

$$
\text{score}(z)=\max\left(\lVert f_{Hovorka}(x,u,p)\rVert_\infty,\;|G(x)-G_{target}|\right)
$$

Update is accepted when trial score decreases.

Constraints during evaluation:

- nonnegative clipping on physiological nonnegative states ($Q_1,Q_2,S_1,S_2,I,D_1,D_2,Y,Z,rGU,rGP,tPA,PAint,rdepl,th$)
- insulin clipping: $u\in[10^{-6},200]$

Stopping criteria:

- $\lVert f_{Hovorka}(x,u,p)\rVert_\infty \le 10^{-6}$
- and $|G(x)-G_{target}|\le tol_G$

with:

- $tol_G=0.1$ mmol/L for mmol/L targets
- mg/dL targets converted by $G_{mmol}=G_{mg/dL}/(M_w^G/10)$ with consistent tolerance conversion

Initialization uses a deterministic warm start from the fasting algebraic construction at $u=10$ mU/min, then sets initial $Q_1$ from the target glucose relation $Q_1=G_{target}V_GBW$.

## 6) Scenario And Exercise Sampling Math

### 6.1 Scenario weight normalization

Raw weights $w_i$ are normalized:

$$
p_i = \frac{w_i}{\sum_j w_j}
$$

and scenario sampled as categorical draw with probabilities $p_i$.

### 6.2 Exercise intensity and scenario mapping

Exercise intensity is represented directly as accelerometer counts $AC(t)$, the native input to the ETH model. Scenario-specific AC ranges (all in [count/min]):

| Scenario | Type | Duration | AC range |
| --- | --- | --- | --- |
| 2 — active | moderate aerobic | 30–75 min | 1200–2000 |
| 7 — prolonged aerobic | long aerobic | 60–90 min | 1500–2500 |
| 8 — anaerobic (EXPERIMENTAL) | resistance/HIIT | 30–60 min | 6000–9000 |
| 9 — exercise + missed bolus | moderate aerobic | 30–75 min | 1000–1800 |

Reference thresholds: $a_{AC}=1000$ counts → $f_{AC}=0.5$ (aerobic onset); $a_h=5600$ counts → $f_{HI}=0.5$ (anaerobic onset).

Total AC at minute $t$:

$$
AC(t) = AC_{baseline}(t,\text{scenario}) + AC_{session}(t,\text{schedule})
$$

where $AC_{baseline}$ models incidental movement synchronized with meal windows (typically 75–500 counts for normal/active, 75–200 for sedentary, 0 for meal-perturbation scenarios 4–6), and $AC_{session}$ is the planned exercise session intensity (including burst modulation for scenario 8).

Anaerobic burst pattern (scenario 8): 2 min on / 3 min rest cycle, $AC_{burst} = 1.5 \times AC_{session}$.

## 7) Sensor And Measurement Equations

### 7.1 Glucose from state

$$
G_{true,mmol/L}=\frac{Q_1}{V_G\cdot BW}
$$

Unit conversion:

$$
G_{mg/dL}=G_{mmol/L}\cdot\frac{M_w^G}{10}
$$

### 7.2 Sensor noise modes

- none: $G_{meas}=G_{true}$
- gaussian: $G_{meas}=G_{true}+\epsilon$, $\epsilon\sim\mathcal N(0,\sigma^2)$
- bias_gaussian: $G_{meas}=G_{true}+b+\epsilon$
- lagged:

$$
G_{lag}(t)=G_{disp}(t-1)+\alpha_{lag}(G_{true}(t)-G_{disp}(t-1))
$$

$$
e_t = \phi e_{t-1}+\eta_t,
\quad
\eta_t\sim\mathcal N\left(0,\sigma^2(1-\phi^2)\right)
$$

$$
G_{meas}(t)=G_{lag}(t)+b+e_t
$$

## 8) Simulation Noise Process (Day Trajectory)

AR(1) process used in simulation loop:

$$
n_t = \rho n_{t-1} + \xi_t,
\quad
\xi_t\sim\mathcal N\left(0,\sigma^2(1-\rho^2)\right)
$$

Daily glycemia sample used for export/plots:

$$
G_t = \frac{Q_{1,t}}{V_G\cdot BW} + n_t
$$

## 9) Control Layer Equations

## 9.1 Hypo guard latch

Guard activates when:

- $G \le G_{guard}$
- not in existing guard window
- cooldown elapsed

Then basal is forced to zero during guard window.

## 9.2 IOB bolus attenuation

Using thresholds $IOB_{guard}$ and $IOB_{full}$:

$$
frac = \text{clip}\left(\frac{IOB-IOB_{guard}}{IOB_{full}-IOB_{guard}},0,1\right)
$$

$$
ICR_{eff} = ICR\cdot\left(1 + frac\cdot(m_{ICR,max}-1)\right)
$$

## 9.3 ISF correction dose

If $G>G_{target}^{corr}$ and cooldown allows:

$$
U_{raw} = \frac{G-G_{target}^{corr}}{ISF}
$$

$$
U_{IOBcredit} = \max(0, IOB-IOB_{free})
$$

$$
U_{net} = \max(0, U_{raw}-U_{IOBcredit})
$$

$$
U_{dose}=\min(U_{max},U_{net}),\quad U_{dose}\ge U_{min}\ \text{to trigger}
$$

Delivered as rate over correction duration $T_{corr}$:

$$
rate_{corr,mU/min} = \frac{U_{dose}\cdot 1000}{T_{corr}}
$$

and added to effective basal while active:

$$
U_{basal,eff}^{(U/hr)} \leftarrow U_{basal,eff}^{(U/hr)} + rate_{corr,mU/min}\cdot\frac{60}{1000}
$$

## 9.4 Hypo rescue carbs

When rescue is active:

$$
D_{rescue,mg/min}=\frac{CHO_{rescue,g}\cdot 1000}{T_{rescue}}
$$

Injected into gut derivative:

$$
\dot D_1 \leftarrow \dot D_1 + A_g\cdot\frac{D_{rescue,mg/min}}{M_w^G}
$$

## 10) Calibration (Sensitivity Module)

## 10.1 ICR identification (bisection on bolus)

Given fixed meal CHO and target postprandial glucose:

- bisection over bolus $U_{bolus}$
- simulate for horizon
- objective: minimize $|G_{final}-G_{target}|$

Final estimate:

$$
ICR = \frac{CHO_{grams}}{U_{bolus,U}}
$$

## 10.2 ISF identification (bisection on correction bolus)

Given high initial glucose and no CHO:

- bisection over correction bolus
- objective: final glucose near target

Final estimate:

$$
ISF = \frac{G_{initial}-G_{final}}{U_{bolus,U}}
$$

## 11) Rejection Criteria (Accepted Cohort)

Patient/day trajectories are accepted only if all criteria hold:

- initial glucose within

$$
G_{init}\in [G_{init,min}, G_{init,max}]
$$

- instability constraints

$$
\max(G) \le G_{max,instability}
$$

$$
\%Hyper \le \theta_{hyper,instability}
$$

- quality constraints

$$
\%Hypo \le \theta_{hypo,quality}
$$

$$
\%Hyper \le \theta_{hyper,quality}
$$

All thresholds come from SimulationConfig.

## 12) Additional Utility Equations

- IOB from depot masses:

$$
IOB[U] = \frac{\max(0,S_1+S_2)}{1000}
$$

- Basal estimate from state:

$$
U_{basal,mU/min}=\frac{S_1}{\tau_I}
$$
