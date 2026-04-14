# Limitations and Scope

This document describes the known limitations of the simulator and the boundaries of its intended scope for the DTU master's thesis. It distinguishes between design choices (intentional simplifications) and genuine gaps (phenomena that are not modelled but could affect downstream ML results).

---

## 1. Population Scope

**Adult T1D only.** Parameter distributions are calibrated to adult patients (18–65 years, mean 35 y, BW 65–95 kg). Paediatric T1D populations — for which the ETH Deichmann exercise model was actually validated — have different insulin sensitivity, growth hormone profiles, and carb-to-insulin ratios. Applying this simulator to paediatric cohort research would require recalibrating the Hovorka parameter priors.

**Male-range body weight.** The BW prior U(65, 95) is centred on a male adult range. Female patients with lower body weight will have different absolute insulin doses and glucose volumes. A sex-stratified BW prior is not implemented.

**No progression.** The simulator generates independent multi-day traces; it does not model disease progression, changes in insulin sensitivity over months/years, or HbA1c drift. Each patient's parameters are fixed for the entire simulation run.

---

## 2. Insulin Delivery Model

**Subcutaneous rapid-acting analogue only.** The model uses a two-compartment subcutaneous depot (S1 → S2 → plasma), matching rapid-acting insulin (lispro/aspart/glulisine). Long-acting basal analogues, pump profiles with variable basal rates, and closed-loop delivery systems are not modelled. The simulated control stack is a rule-based open-loop approximation, not a clinical MPC/AID system.

**No insulin stacking or pen limitations.** Real patients using pens cannot deliver sub-unit doses; the simulator uses continuous dose values. Pen-delivery minimum increments (0.5 U or 1 U) are not enforced.

**Fixed tauI.** The subcutaneous insulin absorption time constant (τ_I) is sampled per patient but does not change with injection site, temperature, or exercise — all of which affect real absorption rate.

---

## 3. Meal and Absorption Model

**Single-compartment gut absorption.** The Hovorka model uses a two-compartment gut model (D1 → D2 → Q1) with a fixed rate and a single bioavailability parameter (A_g). This does not capture:

- Glycaemic index / fat content effects on absorption rate
- Gastric emptying delays from gastroparesis (common in long-duration T1D)
- Alcohol effects on hepatic glucose production

**Carb estimation model is phenomenological.** The bolus underestimation model (biased patient multiplier + meal noise) is calibrated to match published carb-counting error distributions, but does not model the cognitive process of carb counting or patient education level.

---

## 4. Exercise Model

**ETH exercise parameters validated on 5 paediatric patients.** The aerobic ETH parameters (eth_b, eth_q1, eth_q2, eth_q3l, eth_q4l) were fitted to 5 T1D children from the Basel cohort. Adult aerobic exercise physiology may differ, particularly for post-exercise insulin sensitivity (Z-state τ_Z, eth_b drive).

**Anaerobic parameters are EXPERIMENTAL.** The anaerobic overlay (SC8) uses parameters from `params_standard.csv` in the ETH repository, which were not validated on T1D patients. The initial hyperglycaemia spike and post-exercise hypoglycaemia pattern for HIIT/resistance exercise may not match clinical observations.

**No VO2max or fitness stratification.** Exercise intensity is modelled as a single AC value; patient fitness level (VO2max) is not parameterised. Highly trained athletes would show different glucose-exercise coupling than deconditioned patients.

**No on-exercise carbohydrate supplementation.** Real T1D patients often consume carbs during exercise to prevent hypoglycaemia. The simulator's hypo-rescue applies a rule-based two-tier response (15 g at ≤3.9 mmol/L; 30 g at ≤3.0 mmol/L) with fixed cooldowns, which is a blunt approximation of the nuanced decisions real patients make around exercise nutrition.

**Z-state soft ceiling.** The `eth_Z_max=0.2` cap on the post-exercise insulin sensitivity accumulator is an ad hoc stability fix introduced to prevent multi-day Z-buildup from causing runaway hypoglycaemia. This cap is not present in the original ETH model and lacks physiological justification; it is marked EXPERIMENTAL in the codebase.

---

## 5. Circadian and Hormonal Modelling

**Dawn phenomenon only.** The simulator models GH-driven EGP elevation (03:00–07:00) and cortisol morning insulin resistance (06:00–10:00) using a shared `dawn_amp` parameter. Other circadian hormones (glucagon, adrenaline, growth factors) are not modelled.

**No nocturnal hypoglycaemia physiology.** Counterregulatory responses to hypoglycaemia (glucagon release, adrenaline) are not modelled. The hypo-rescue layer applies carbohydrates as a proxy but does not simulate endogenous counterregulation.

**No sleep physiology.** Sleep quality, sleep apnoea, and nocturnal GH pulsatility variation are not captured beyond the fixed dawn window.

---

## 6. CGM Sensor Model

**First-order lag + AR(1) noise only.** The sensor model applies a physiological interstitial lag (first-order, α=0.25) and autocorrelated measurement noise (AR(1), φ=0.70). It does not model:

- Calibration drift over sensor wear time
- Compression artefacts (falsely low readings during sleep)
- Sensor warm-up period (first 2 h after insertion)
- Compression-induced noise asymmetry

**No signal drop-outs.** Real CGM sensors lose signal intermittently (up to 5–10% of readings in clinical data). No missing data is generated.

---

## 7. Safety and Control Stack

**Rule-based, not clinical MPC.** The four-layer control stack (hypo guard, hypo rescue, IOB attenuation, ISF correction) is a simplified rule-based system designed to produce physiologically plausible insulin delivery, not to match any specific commercial AID device. It should not be used to benchmark real controllers.

**Calibrated but not personalised corrections.** The ISF correction target (10.5 mmol/L) and hypo rescue threshold (3.9 mmol/L) are fixed constants. Real patients use personalised targets, which vary significantly.

---

## 8. Validation Status

**Not validated against real T1D patient data.** This simulator has not been compared against real CGM traces, clinical trial data, or a validated virtual patient population (e.g., the FDA-accepted UVA/Padova simulator). All parameter distributions are derived from published literature and the ETH patient files, not from direct model fitting to a held-out clinical dataset.

**No clinical trial use.** This simulator is intended for ML dataset generation and algorithm pre-screening only. It is not validated for clinical decision support, regulatory submissions, or direct patient care.

---

## 9. Out of Scope (Thesis)

The following topics are explicitly out of scope for the current thesis work:

- Type 2 diabetes or pre-diabetes physiology
- Closed-loop / AID system simulation
- Multi-hormone models (glucagon, GLP-1, amylin)
- Intravenous insulin delivery
- Paediatric population simulation
- Prospective clinical validation against real patient data
- FDA/CE regulatory pathway for virtual patient platforms
