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

### 4.5 Model coupling architecture

The ETH Deichmann paper (2023) presents a complete 14-state glucose-insulin-exercise model: 6 simplified glucose-insulin states plus the 8 exercise states above. We do **not** use the ETH glucose-insulin core. Instead, we keep the Hovorka 10-state glucose-insulin model (which is more physiologically detailed and better validated for closed-loop insulin delivery) and graft only the ETH exercise interface onto it.

The grafting principle: extract the three coupling terms from the ETH Q1 equation, discard the rest of the ETH glucose core, and inject those terms into Hovorka's $\dot Q_1$.

**Why only $Q_1$ and $x_1$, not the other states?**

$Q_1$ is the accessible glucose compartment — it represents circulating plasma glucose, which is the quantity directly affected by muscle uptake, liver output, and insulin action. All three exercise coupling terms act on $Q_1$ because that is where their physiological effects manifest:

- `exercise_uptake` ($rGU \cdot Q_1$): muscles consume glucose from the circulating pool during exercise. Proportional to $Q_1$ because at higher blood glucose there is more substrate available.
- `exercise_prod` ($\max(0, rGP-rdepl) \cdot Q_1$): the liver releases glucose during exercise (exercise-driven EGP). The $Q_1$ proportionality is a linearisation used in the original ETH model.
- `exercise_si` ($Z \cdot x_1 \cdot Q_1$): post-exercise elevated insulin sensitivity. The state $Z$ amplifies the existing Hovorka insulin action state $x_1$ on $Q_1$. $x_1$ specifically governs glucose transport (inter-compartment flux $R_{12}$), which is where the post-exercise GLUT4 upregulation effect is best captured. $x_2$ (glucose disposal from $Q_2$) and $x_3$ (EGP suppression) are not directly enhanced by exercise in the same way.

$Q_2$ is a peripheral tissue compartment. It only exchanges with $Q_1$ through the $R_{12}$ term; exercise does not act on it directly, so $\dot Q_2 = R_{12} - R_2$ remains unchanged.

**Note on `eth_alpha`:** The ETH paper defines a scaling constant $\alpha = 0.27$ used in some formulations. It is stored in the patient parameter set (`eth_alpha`) for completeness but is not part of the coupling equations implemented here, where the three interaction terms are taken directly from the published ODE form without additional scaling.

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

- mmol/L path: $tol_G = 0.1$ mmol/L
- mg/dL path: input $G_{mg/dL}$ is first converted to mmol/L via $G_{mmol}=G_{mg/dL}\,/\,(M_w^G/10)$, then $tol_G = 0.1\,/\,(M_w^G/10) \approx 0.006$ mmol/L

The mg/dL tolerance is tighter by unit-conversion, but in practice the $\lVert f \rVert_\infty \le 10^{-6}$ ODE residual bound is always the binding constraint — both paths converge to residuals of order $10^{-8}$ to $10^{-7}$.

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

Three sequential stages. A patient is regenerated if any stage fails.

**Stage 1 — Initial-state:**

$$
G_{init}\in [G_{init,min},\; G_{init,max}]
$$

**Stage 2 — Instability (evaluated over full horizon):**

$$
\max(G) \le G_{max,instability}, \quad \%Hyper \le \theta_{hyper,instability}
$$

**Stage 3 — Quality (evaluated per recorded day):**

Each recorded day $d$ with scenario $s_d$ is evaluated independently. Exercise scenarios ($s_d \in \{2,7,8,9\}$) use a relaxed hypo threshold because physiological hypoglycaemia during or after vigorous exercise is expected and handled by the rescue system:

$$
\%Hypo_d \le
\begin{cases}
\theta_{hypo,exercise} & \text{if } s_d \in \{2,7,8,9\} \\
\theta_{hypo,quality}  & \text{otherwise}
\end{cases}
$$

$$
\%Hyper_d \le \theta_{hyper,quality} \quad \forall\, d
$$

**Hard glucose floor (any day, any scenario):**

$$
\min_t G(t) \ge G_{floor}
$$

A patient is rejected if any single day violates its threshold, or if glucose drops below the floor at any minute. This prevents deeply hypoglycaemic spikes from entering the accepted cohort even when the day-average hypo% would pass.

Default values: $\theta_{hypo,quality}=4\%$, $\theta_{hypo,exercise}=8\%$, $\theta_{hyper,quality}=12\%$, $\theta_{hyper,instability}=30\%$, $G_{floor}=3.0$ mmol/L. All thresholds are config-driven from `SimulationConfig`.

## 12) Monte Carlo Parameter Distributions

Patient variability is introduced by sampling physiological parameters from published distributions. The following table documents base values, sampling distributions, and sources. Fixed parameters are the same for all patients; sampled parameters are drawn independently per patient.

### 12.1 Hovorka core parameters

| Parameter | Base value | Unit | MC distribution | Source / notes |
| --- | --- | --- | --- | --- |
| `MwG` | 180.16 | mg/mmol | Fixed | Molecular weight of glucose (C₆H₁₂O₆) |
| `EGP0` | 0.0161 | mmol/kg/min | N(0.0161, 0.0039²) | Hovorka 2004, Table 1 |
| `F01` | 0.0097 | mmol/kg/min | N(0.0097, 0.0022²) | Hovorka 2004, Table 1 |
| `k12` | 0.0649 | 1/min | N(0.0649, 0.0282²) | Hovorka 2004, Table 1 |
| `ka1` | 0.0055 | 1/min | N(0.0055, 0.0006²) | Hovorka 2004; CV reduced to ~10% (see note A) |
| `ka2` | 0.0683 | 1/min | N(0.0683, 0.0068²) | Hovorka 2004; CV reduced to ~10% |
| `ka3` | 0.0304 | 1/min | N(0.0304, 0.0030²) | Hovorka 2004; CV reduced to ~10% |
| `SI1` | 32.0×10⁻⁴ | L/min/mU | N(32e-4, 20e-4²) | Scaled ~38% down from Hovorka 2004 (see note B) |
| `SI2` | 5.1×10⁻⁴ | L/min/mU | N(5.1e-4, 4.9e-4²) | Hovorka 2004 / Boiroux thesis Table 2.1 |
| `SI3` | 325×10⁻⁴ | L/mU | N(325e-4, 191e-4²) | Scaled ~37% down from Hovorka 2004 |
| `ke` | 0.138 | 1/min | N(0.14, 0.035²) | Hovorka 2004; plasma insulin half-life ≈ 5 min |
| `VI` | 0.12 | L/kg | N(0.12, 0.012²) | Hovorka 2004, Table 1 |
| `VG` | 0.1484 | L/kg | log(exp(VG)) ~ N(1.16, 0.23²) | Hovorka 2004; exp(mean)=1.16 → VG=log(1.16)=0.148 |
| `tauI` | 55.871 | min | 1/tauI ~ N(0.018, 0.0045²) | Hovorka 2004; mean rate 0.018 → tauI≈55.6 min |
| `tauG` | 39.908 | min | ln(tauG) ~ N(3.689, 0.25²) | Hovorka 2004; exp(3.689)≈40 min |
| `Ag` | 0.7943 | — | U(0.70, 0.90) | Physiological CHO bioavailability in T1D adults |
| `BW` | 80.0 | kg | U(65, 95) | Adult T1D outpatient cohort; male-range centred |
| `age_years` | 35.0 | years | N(35, 12²) clipped [18, 65] | Adult T1D population assumption |

**Note A — ka variability reduction:** The Boiroux thesis reports CVs of ~100%, 74%, 77% for ka1/2/3 from a 7-patient cohort. Because the ODE uses $k_b = SI \cdot k_a$, sampling both SI and ka with high independent variance compounds the effective $k_b$ spread as $\sqrt{CV_{SI}^2 + CV_{ka}^2}$, producing physiologically implausible insulin action. CV is reduced to ~10% here, concentrating variability in SI (the clinically meaningful parameter), which better matches published ICR/ISF distributions (Dalla Man et al. 2007).

**Note B — SI1/SI3 scaling:** The 7-patient Hovorka 2004 cohort had higher mean SI1 (~51×10⁻⁴) than the broader adult T1D population. Values are scaled down ~38% to target a mean ICR of ~12 g/U, consistent with Dalla Man et al. 2007 (IEEE TBME, Table II). Stds are scaled proportionally to preserve published CVs.

### 12.2 ETH Deichmann exercise parameters

Aerobic parameters are validated on 5 T1D children from the Basel University Children's Hospital (Deichmann et al. 2023, gitlab.com/csb.ethz/t1d-exercise-model). Anaerobic parameters are marked **EXPERIMENTAL** — not validated on T1D patients; values from `params_standard.csv`.

| Parameter | Base value | Unit | MC distribution | Notes |
| --- | --- | --- | --- | --- |
| `eth_tau_AC` | 5.0 | min | Fixed | AC → Y time constant; consistent across all patient files |
| `eth_tau_Z` | 600.0 | min | Fixed | Post-exercise SI decay (~10h); consistent across patients |
| `eth_b` | 3.0×10⁻⁶ | 1/(count·min) | N(3e-6, 1e-6²) | Z drive; mean of patients 1–5 and T1D-V1 |
| `eth_q1` | 1.0×10⁻⁶ | 1/(count·min²) | N(1e-6, 8e-7²) | rGU drive; mean of T1D patients |
| `eth_q2` | 0.10 | 1/min | N(0.10, 0.08²) | rGU decay; mean of T1D patients |
| `eth_q3l` | 3.0×10⁻⁷ | — | N(3e-7, 1.5e-7²) | rGP aerobic drive; mean of T1D patients |
| `eth_q4l` | 0.060 | 1/min | N(0.060, 0.018²) | rGP aerobic decay; mean of T1D patients |
| `eth_adepl` | 0.0108 | min/count | Fixed | Depletion threshold slope; fixed across patients |
| `eth_bdepl` | 180.6 | count·min | Fixed | Depletion threshold intercept; fixed across patients |
| `eth_aY` | 1500.0 | count·min | Fixed | fY half-saturation; fixed across patients |
| `eth_aAC` | 1000.0 | count | Fixed | fAC onset; ~1000 counts ≈ moderate walking |
| `eth_ah` | 5600.0 | count | Fixed | Anaerobic threshold; ~5600 counts ≈ high-intensity run |
| `eth_n1` | 20.0 | — | Fixed | fY Hill coefficient; steep switch behaviour |
| `eth_n2` | 100.0 | — | Fixed | fAC/fHI Hill coefficient; near-binary switch |
| `eth_tp` | 2.0 | min | Fixed | fp half-saturation |
| `eth_q3h` | 1.17×10⁻⁶ | — | Fixed (EXPERIMENTAL) | rGP anaerobic drive; params_standard.csv |
| `eth_q4h` | 0.0705 | 1/min | Fixed (EXPERIMENTAL) | rGP anaerobic decay; params_standard.csv |
| `eth_q5` | 0.03 | 1/min | Fixed (EXPERIMENTAL) | th decay rate; params_standard.csv |
| `eth_q6` | 0.0 | 1/min | Fixed | Glycogen depletion rate; 0 for aerobic-only validated path |

**Note on eth_n1/n2:** The very large Hill coefficients (20, 100) make the sigmoidal transfer functions behave as near-binary switches. This is intentional in the ETH formulation: $f_{AC}$ switches sharply at $a_{AC}=1000$ counts to mark the onset of activity, and $f_{HI}$ switches sharply at $a_h=5600$ counts to mark the anaerobic transition.

**Note on eth_q6=0:** In the T1D-validated aerobic parameter files, $q_6=0$, meaning glycogen depletion ($rdepl$) is inactive and $\dot{rdepl}=0$ always. This is consistent with the observation that moderate aerobic exercise in T1D children does not produce significant glycogen depletion within typical session durations. The EXPERIMENTAL anaerobic path uses a non-zero $q_6$ for prolonged/intense sessions (scenario 7/8), but this pathway is not yet clinically validated.

## 13) Additional Utility Equations

- IOB from depot masses:

$$
IOB[U] = \frac{\max(0,S_1+S_2)}{1000}
$$

- Basal estimate from state:

$$
U_{basal,mU/min}=\frac{S_1}{\tau_I}
$$
