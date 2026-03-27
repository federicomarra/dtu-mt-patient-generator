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
D(t) = \frac{D_{mg}(t)}{M_{wG}}
$$

with $M_{wG}$ in [mg/mmol], so $D(t)$ is [mmol/min].

Basal conversion:

$$
U(t) = \frac{U_{hr} \cdot 1000}{60}
$$

where $U_{hr}$ is [U/hr], $U(t)$ is [mU/min].

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
\dot I = \frac{U_I}{V_I \cdot BW} - k_e\cdot I
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
\dot{rdepl} = q_6\,(f_t\,r_m - rdepl)
$$

where $r_m = rGP$ (current exercise glucose production rate, used as ceiling for depletion).

**Approximation note:** In the original Deichmann model, $r_m = \beta\left(\frac{q_3}{q_4}Y + (1-\alpha)(p_1+X_b)\right)$, where $p_1$ and $X_b$ are basal glucose production/utilization terms from the Deichmann glucose-insulin core, and $\beta$ is an additional scaling parameter. Since we do not use the Deichmann glucose core (we use the Hovorka 10-state model), these parameters have no direct equivalent. We substitute $r_m = rGP$ (the current exercise EGP rate), which at quasi-steady-state satisfies $rGP \approx q_3 f_Y Y / q_4$, matching the dominant term $q_3/q_4 \cdot Y$ from the Deichmann formula. The basal contribution $(1-\alpha)(p_1+X_b)$ is small relative to the exercise-driven term during active exercise, and introducing a Hovorka-compatible approximation (e.g. $F_{01}^c/Q_1$) would add Hovorka-specific coupling that the original ETH model does not assume. The simplification is therefore deliberate and has negligible impact for aerobic scenarios where $q_6=0$ makes $rdepl$ identically zero regardless of $r_m$.

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

> **Implementation note:** The $\max(0,\cdot)$ clamp on $(rGP-rdepl)$ is an addition not present in the original Deichmann ODE, where the term is taken as-is. It is included here as a safety guard to prevent exercise-driven EGP from becoming negative (which would act as an unphysical glucose sink) in edge cases where numerical drift allows $rdepl > rGP$. Both states are non-negative by projection, so this situation is rare but can arise transiently during ODE integration.

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

The ETH Deichmann paper (2023) presents a complete 14-state glucose-insulin-exercise model: 6 simplified glucose-insulin states plus the 8 exercise states above. We do **not** use the ETH glucose-insulin core. Instead, we keep the Hovorka 10-state glucose-insulin model (which is more physiologically detailed and better validated for closed-loop insulin delivery) and fit only the ETH exercise interface onto it.

The fitting principle consists in extracting the three coupling terms from the ETH Q1 equation, discard the rest of the ETH glucose core, and inject those terms into Hovorka's $\dot Q_1$.

**Why only $Q_1$, not the other states?**

$Q_1$ is the accessible glucose compartment; it represents circulating plasma glucose, which is the quantity directly affected by muscle uptake, liver output, and insulin action. All three exercise coupling terms act on $Q_1$ because that is where their physiological effects manifest:

- `exercise_uptake` ($rGU \cdot Q_1$): muscles consume glucose from the circulating pool during exercise. Proportional to $Q_1$ because at higher blood glucose there is more substrate available.
- `exercise_prod` ($\max(0, rGP-rdepl) \cdot Q_1$): the liver releases glucose during exercise (exercise-driven EGP). The $Q_1$ proportionality is a linearisation used in the original ETH model.
- `exercise_si` ($Z \cdot x_1 \cdot Q_1$): post-exercise elevated insulin sensitivity. The state $Z$ amplifies the existing Hovorka insulin action state $x_1$ on $Q_1$. $x_1$ specifically governs glucose transport (inter-compartment flux $R_{12}$), which is where the post-exercise GLUT4 upregulation effect is best captured. $x_2$ (glucose disposal from $Q_2$) and $x_3$ (EGP suppression) are not directly enhanced by exercise in the same way.

$Q_2$ is a peripheral tissue compartment. It only exchanges with $Q_1$ through the $R_{12}$ term; exercise does not act on it directly, so $\dot Q_2 = R_{12} - R_2$ remains unchanged.

**Note on `eth_alpha` and `eth_beta`:** The ETH paper uses $\alpha = 0.27$ and $\beta$ in the $r_m$ formula for the glycogen depletion ODE (see the `rm` approximation note in section 4.3). `eth_alpha` is stored in the patient parameter set for completeness but is not used as an additional scaling factor in any of the three $Q_1$ coupling terms above. `eth_beta` is not stored because the entire $r_m$ formula is replaced by the $r_m = rGP$ approximation.

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
- mg/dL path: input $G_{mg/dL}$ is first converted to mmol/L via $G_{mmol}=G_{mg/dL}\,/\,(M_w^G/10)$, then $tol_G = 0.1$ mmol/L (= 1.8 mg/dL). Both unit paths share the same 0.1 mmol/L tolerance; it is incorrect to divide the tolerance by the conversion factor (doing so would give a $18\times$ tighter bound of $\approx 0.0055$ mmol/L with no physiological justification).

In practice the $\lVert f \rVert_\infty \le 10^{-6}$ ODE residual bound is always the binding constraint — both paths converge to residuals of order $10^{-8}$ to $10^{-7}$.

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
| 7 — prolonged aerobic | long aerobic | **80–90 min** | 1500–2500 |
| 8 — anaerobic (EXPERIMENTAL) | resistance/HIIT | 30–60 min | 6000–9000 |
| 9 — exercise + missed bolus | moderate aerobic | 30–75 min | 1000–1800 |

Reference thresholds: $a_{AC}=1000$ counts → $f_{AC}=0.5$ (aerobic onset); $a_h=5600$ counts → $f_{HI}=0.5$ (anaerobic onset).

**ML distinguishability note:** The AC ranges for scenarios 2, 7, and 9 all fall in the aerobic band. Because $n_2=100$ (very steep Hill coefficient), $f_{AC}$ saturates to $\approx 1$ for any $AC \gtrsim 1100$ — the ETH model is functionally identical for $AC=1500$ and $AC=2000$, so AC range differences between these scenarios do not produce distinguishable glucose signatures. The primary differentiator between scenario 2 and scenario 7 is **duration**: scenario 7 is guaranteed to be $\ge 80$ min (above the scenario 2 maximum of 75 min), ensuring that every scenario 7 session always produces greater $Z$, $rGU$, and $rGP$ accumulation than any scenario 2 session. Scenario 8 is cleanly separated in both duration and AC (above $a_h=5600$ anaerobic threshold).

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

`src/sensor.py` implements four modes. The **lagged** mode is the one active in the main simulation and library generation:

- **none**: $G_{meas}=G_{true}$
- **gaussian**: $G_{meas}=G_{true}+\epsilon$, $\epsilon\sim\mathcal N(0,\sigma^2)$
- **bias_gaussian**: $G_{meas}=G_{true}+b+\epsilon$
- **lagged** *(active)*: two-stage model combining a physiological CGM lag with AR(1) correlated error

## 8) Simulation Noise Process (Active: Lagged CGM Model)

The simulation loop uses the **lagged** sensor mode applied point-by-point across the state trajectory. The sensor state (previous display value and previous AR(1) error) is carried across day boundaries, making the noise process continuous over the full multi-day patient horizon.

**What is AR(1)?** AR(1) stands for *first-order autoregressive process*: each noise sample is a weighted sum of the previous sample and a fresh Gaussian innovation. This produces temporally correlated noise — consecutive CGM errors tend to be in the same direction rather than jumping independently each minute. The parameter $\phi \in [0,1)$ controls the correlation strength: $\phi=0$ is white (uncorrelated) noise, $\phi \to 1$ is a near-random-walk with very slow drift. Real CGM sensors have correlated errors due to sensor drift and interstitial fluid dynamics, making AR(1) a standard model in the glucose monitoring literature.

**Stage 1 — Physiological CGM lag:**

A real CGM measures interstitial fluid glucose, which lags behind plasma glucose by roughly 5–15 minutes. The first-order lag model blends the current true glucose toward the previous displayed value:

$$
G_{lag}(t)=G_{disp}(t-1)+\alpha_{lag}\bigl(G_{true}(t)-G_{disp}(t-1)\bigr)
$$

where $\alpha_{lag} \in (0,1]$ is the blend factor. Smaller $\alpha_{lag}$ means more lag (slower tracking). Default $\alpha_{lag}=0.25$ corresponds to a time constant of $\tau = 1/\alpha_{lag} - 1 = 3$ sample periods (≈3 min).

**Stage 2 — AR(1) correlated sensor error:**

$$
e_t = \phi\, e_{t-1}+\eta_t,
\quad
\eta_t\sim\mathcal N\!\left(0,\,\sigma^2(1-\phi^2)\right)
$$

The innovation variance $\sigma^2(1-\phi^2)$ is chosen so that the stationary variance of $e_t$ equals $\sigma^2$ regardless of $\phi$.

**Combined measurement:**

$$
G_{meas}(t)=G_{lag}(t)+e_t
$$

(bias $b=0$ by default).

**Parameters** (all config-driven from `SimulationConfig`):

| Parameter | Symbol | Default | Meaning |
| --- | --- | --- | --- |
| `noise_std` | $\sigma$ | 0.10 mmol/L | Stationary std of correlated error |
| `noise_autocorr` | $\phi$ | 0.70 | AR(1) autocorrelation coefficient |
| `cgm_lag_alpha` | $\alpha_{lag}$ | 0.25 | CGM lag blend factor |

The physio (noiseless) trajectory is computed separately and used exclusively for the rejection pipeline — rejection decisions are never influenced by sensor noise.

## 9) Control Layer Equations

## 9.1 Hypo guard latch

Guard activates when:

- $G \le G_{guard}$
- not in existing guard window
- cooldown elapsed

Then basal is forced to zero during guard window.

## 9.2 IOB bolus attenuation

When insulin-on-board (IOB) is high, the next meal bolus is reduced by inflating the effective insulin-to-carb ratio (ICR). A higher ICR means fewer units of insulin are dosed per gram of carbohydrate, so the computed bolus dose decreases. This prevents stacking of meal boluses on top of already-active insulin.

Using thresholds $IOB_{guard}$ and $IOB_{full}$:

$$
frac = \text{clip}\left(\frac{IOB-IOB_{guard}}{IOB_{full}-IOB_{guard}},0,1\right)
$$

$$
ICR_{eff} = ICR\cdot\left(1 + frac\cdot(m_{ICR,max}-1)\right)
$$

- When $IOB \le IOB_{guard}$: $frac=0$, $ICR_{eff}=ICR$ — no attenuation, full bolus dose.
- When $IOB \ge IOB_{full}$: $frac=1$, $ICR_{eff}=m_{ICR,max}\cdot ICR$ — maximum attenuation, bolus dose scaled down to $1/m_{ICR,max}$ of the unadjusted amount.
- $m_{ICR,max} > 1$ **increases** ICR (and therefore **decreases** the bolus dose). Default: $m_{ICR,max}=1.6$ → minimum bolus is 62.5% of the unadjusted dose.
- Default thresholds: $IOB_{guard}=4$ U, $IOB_{full}=8$ U. All values are config-driven from `SimulationConfig`.

## 9.3 ISF correction dose

When glucose is persistently above target and a cooldown window has elapsed, a small correction bolus is computed and delivered spread over a short window. This acts as a lightweight secondary controller to bring glucose back toward the target range.

If $G>G_{target}^{corr}$ and cooldown allows:

$$
U_{raw} = \frac{G-G_{target}^{corr}}{ISF}
$$

This is the naive correction dose based on the current glucose gap and the patient's insulin-sensitivity factor.

$$
U_{IOBcredit} = \max(0, IOB-IOB_{free})
$$

The IOB credit represents insulin already on board that is already working toward correction. Only the excess above the "free zone" ($IOB_{free}$) is credited — a small residual IOB (e.g. $\le 0.5$ U) is not subtracted, to allow independent small corrections when glucose is persistently high.

$$
U_{net} = \max(0, U_{raw}-U_{IOBcredit})
$$

The credit is subtracted to prevent double-dosing (bolus stacking): if IOB is already sufficient to close the gap, no additional correction fires.

$$
U_{dose}=\min(U_{max},U_{net}),\quad U_{dose}\ge U_{min}\ \text{to trigger}
$$

$U_{dose}$ is capped at $U_{max}$ to prevent excessively large automated corrections, and the correction only fires if the net dose is at least $U_{min}$ (prevents spurious micro-boluses).

Delivered as rate over correction duration $T_{corr}$:

$$
rate_{corr,mU/min} = \frac{U_{dose}\cdot 1000}{T_{corr}}
$$

and added to effective basal while active:

$$
U_{basal,eff}^{(U/hr)} \leftarrow U_{basal,eff}^{(U/hr)} + rate_{corr,mU/min}\cdot\frac{60}{1000}
$$

Default values: $G_{target}^{corr}=6.5$ mmol/L, $IOB_{free}=0.5$ U, $U_{max}=2$ U, $U_{min}=0.05$ U, $T_{corr}=5$ min, cooldown=60 min. All values are config-driven from `SimulationConfig`.

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

**Stage 2 — Instability (evaluated over the full concatenated trajectory):**

$$
\max(G) \le G_{max,instability}, \quad \%Hyper \le \theta_{hyper,instability}, \quad \%Hypo \le \theta_{hypo,instability}
$$

Stage 2 is evaluated on the **full multi-day trajectory as a single concatenated sequence** (all days merged). Its purpose is to quickly discard numerically diverging or physiologically runaway simulations before the more expensive per-day quality checks. For example, a single day with 100% hyper-time in a 7-day run contributes only $\sim 14\%$ to the global hyper%, so it passes Stage 2 ($\theta_{hyper,instability}=60\%$) — this is intentional: Stage 2 catches explosive instability (e.g. glucose rocketing to 30+ mmol/L for hours), not moderate per-day excursions. The hypo instability check ($\theta_{hypo,instability}=8\%$) analogously catches sustained global hypoglycaemia across the full horizon.

**Stage 3 — Quality (evaluated per recorded day independently):**

Each day $d$ with scenario $s_d$ gets a base hypo threshold, then a spillover bonus is added if the **previous recorded day** was an exercise scenario:

$$
\theta_{hypo,base}(d) =
\begin{cases}
\theta_{hypo,exercise} & \text{if } s_d \in \{2,7,8,9\} \\
\theta_{hypo,quality}  & \text{otherwise}
\end{cases}
$$

$$
\theta_{hypo}(d) = \theta_{hypo,base}(d) + \delta_{spillover} \cdot \mathbf{1}[s_{d-1} \in \{2,7,8,9\}]
$$

The spillover bonus $\delta_{spillover}$ accounts for the $Z$-state (post-exercise insulin sensitivity, $\tau_Z \approx 600$ min) remaining partially active the next morning, increasing hypo risk even on days with a non-exercise scenario label. The bonus stacks correctly for back-to-back exercise days:

| Day $d-1$ | Day $d$ | Effective $\theta_{hypo}(d)$ |
| --- | --- | --- |
| non-exercise | non-exercise | $10\%$ |
| exercise | non-exercise | $10\% + 2\% = 12\%$ |
| non-exercise | exercise | $15\%$ |
| exercise | exercise | $15\% + 2\% = 17\%$ |

$$
\%Hyper_d \le \theta_{hyper,quality} \quad \forall\, d
$$

Stage 3 is evaluated **per recorded day independently**. A patient is rejected if **any single day** violates its threshold — good days cannot compensate for one bad day. This means a patient with consistently 76% hyper per day (which passes Stage 2 globally at 76% > 60% would still be caught there, but lower consistent values pass Stage 2) is rejected here because each day individually exceeds $\theta_{hyper,quality}=75\%$.

The distinction between Stage 2 and Stage 3 hyper thresholds (60% vs 75%) is intentional: Stage 2 catches explosive global instability across the full horizon; Stage 3 enforces per-day clinical quality — a single catastrophic day cannot be diluted by good days.

**Hard glucose floor (any day, any scenario):**

$$
\min_t G_{physio}(t) \ge G_{floor}
$$

A patient is rejected if any single day violates its threshold, or if the **physiological** (noiseless) glucose drops below $G_{floor}$ at any minute. This prevents deeply hypoglycaemic states from entering the accepted cohort even when the day-average hypo% would pass.

Default values: $\theta_{hypo,quality}=10\%$, $\theta_{hypo,exercise}=15\%$, $\delta_{spillover}=2\%$, $\theta_{hyper,quality}=75\%$, $\theta_{hyper,instability}=60\%$, $\theta_{hypo,instability}=8\%$, $G_{max,instability}=30.53$ mmol/L (550 mg/dL), $G_{floor}=1.78$ mmol/L (32 mg/dL). All thresholds are config-driven from `SimulationConfig`.

**Design note — floor applied to physiological glucose only, not to the CGM signal:**

All rejection criteria, including the hard floor, are evaluated exclusively on the noiseless physiological glucose $G_{physio}(t) = Q_1(t)/(V_G \cdot BW)$. The exported `blood_glucose` column is the lagged CGM signal $G_{meas}(t)$ (section 8), which can occasionally fall below $G_{floor}$ due to sensor noise even when the physiological state is above it.

This design choice is deliberate and motivated by the downstream ML use case:

1. **Scenario anomaly detection** (primary task). The ML model learns to identify anomaly patterns — missed bolus, late bolus, prolonged exercise — from the shape of the glucose and insulin trajectory over hours. Borderline low CGM readings do not affect this learning; the anomaly signature is carried by the global trajectory structure, not individual point values.

2. **Hypoglycaemia alarm** (secondary task). CGM readings below 3.0 mmol/L in the exported data represent genuine near-hypoglycaemia events: the physiological glucose was at or near $G_{floor}$, the hypo rescue system was already active, and the sensor correctly reflected a very low value. Filtering these out would remove exactly the signal the model needs to learn alarm-worthy situations. It would produce a sanitised dataset where hypoglycaemia always appears well-resolved and the model never sees the borderline CGM patterns that precede a clinical alarm.

The alternative — applying the floor to the CGM signal as well — would introduce a selection bias: patients whose measurement noise happened to be positive when physio glucose was near $G_{floor}$ would be accepted, while physiologically identical patients with negative noise at the same moment would be rejected. The physio floor avoids this bias while still guaranteeing that no patient with a genuinely dangerous physiological state enters the cohort.

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
