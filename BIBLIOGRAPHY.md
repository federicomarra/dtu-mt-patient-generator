# Bibliography

All references cited in the simulator codebase, equations documentation, and thesis.

---

## Core Glucose-Insulin Model

**Hovorka 2004** — primary ODE model
> Hovorka, R., Canonico, V., Chassin, L. J., Haueter, U., Massi-Benedetti, M., Orsini Federici, M., Pieber, T. R., Schaller, H. C., Schaupp, L., Sendlhofer, G., & Wilinska, M. E. (2004). Nonlinear model predictive control of glucose concentration in subjects with type 1 diabetes. *Physiological Measurement*, 25(4), 905–920. <https://doi.org/10.1088/0967-3334/25/4/010>

**Boiroux 2012** — Hovorka parameter distributions and MPC framework
> Boiroux, D. (2012). *Model Predictive Control for Type 1 Diabetes* (PhD thesis). Technical University of Denmark. [Table 2.1 used for parameter priors]

---

## Exercise Extension

**Deichmann 2023** — ETH Deichmann exercise model (8-state accelerometer-driven extension)
> Deichmann, J., Bachmann, S., Burckhardt, M.-A., Pfister, M., Szinnai, G., Kaltenbach, H.-M., & Mougiakakou, S. (2023). New model of glucose-insulin regulation characterizes effects of physical activity and facilitates personalized treatment evaluation in children and adults with type 1 diabetes. *PLOS Computational Biology*, 19(2), e1010289. <https://doi.org/10.1371/journal.pcbi.1010289>

Source code and patient data: <https://gitlab.com/csb.ethz/t1d-exercise-model>

---

## Patient Parameter Distributions

**Dalla Man 2007** — meal simulation model and insulin sensitivity distributions
> Dalla Man, C., Rizza, R. A., & Cobelli, C. (2007). Meal simulation model of the glucose-insulin system. *IEEE Transactions on Biomedical Engineering*, 54(10), 1740–1749. <https://doi.org/10.1109/TBME.2007.893506>

**Wilinska 2010** — virtual patient cohort for closed-loop evaluation
> Wilinska, M. E., Chassin, L. J., Acerini, C. L., Allen, J. M., Dunger, D. B., & Hovorka, R. (2010). Simulation environment to evaluate closed-loop insulin delivery systems in type 1 diabetes. *Journal of Diabetes Science and Technology*, 4(1), 132–144. <https://doi.org/10.1177/193229681000400117>

---

## Dawn Phenomenon and Circadian Physiology

**Perriello 1988** — growth hormone pulses and hepatic glucose production (dawn EGP elevation)
> Perriello, G., De Feo, P., Torlone, E., Calcinaro, F., Ventura, M. M., Basta, G., Santeusanio, F., Brunetti, P., Gerich, J. E., & Bolli, G. B. (1988). The effect of asymptomatic nocturnal hypoglycemia on glycemic control in diabetes mellitus. *New England Journal of Medicine*, 319(19), 1233–1239. <https://doi.org/10.1056/NEJM198811103191901>

**Carroll & Schade 2005** — review of dawn phenomenon mechanisms (GH + cortisol)
> Carroll, M. F., & Schade, D. S. (2005). The dawn phenomenon revisited: implications for diabetes therapy. *Endocrine Practice*, 11(1), 55–64. <https://doi.org/10.4158/EP.11.1.55>

**Boden 1996** — cortisol-mediated morning insulin resistance
> Boden, G., Chen, X., & Urbain, J. L. (1996). Evidence for a circadian rhythm of insulin sensitivity in patients with NIDDM caused by cyclic changes in hepatic glucose production. *Diabetes*, 45(8), 1044–1050. <https://doi.org/10.2337/diab.45.8.1044>

---

## Sensor and Noise Models

**Kovatchev 2009** — CGM error and interstitial glucose lag
> Kovatchev, B. P., Shields, D., & Breton, M. (2009). Graphical and numerical evaluation of continuous glucose sensing time lag. *Diabetes Technology & Therapeutics*, 11(3), 139–143. <https://doi.org/10.1089/dia.2008.0044>

---

## Steady-State Initialization and Numerical Methods

**Nocedal & Wright 2006** — Newton-Raphson with damping/backtracking (standard reference)
> Nocedal, J., & Wright, S. J. (2006). *Numerical Optimization* (2nd ed.). Springer. Chapter 11.

---

## Bolus Estimation and Dietary Assessment

**Brazeau 2013** — carb counting accuracy in T1D and sources of estimation error
> Brazeau, A.-S., Mircescu, H., Desjardins, K., Leroux, C., Strychar, I., Ekoe, J.-M., & Rabasa-Lhoret, R. (2013). Carbohydrate counting accuracy and understanding of insulin-to-carb ratios in people with type 1 diabetes. *Diabetes Research and Clinical Practice*, 99(1), 19–23. <https://doi.org/10.1016/j.diabres.2012.10.024>
