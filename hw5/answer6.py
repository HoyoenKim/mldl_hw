import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter
from lifelines.plotting import add_at_risk_counts
import matplotlib.pyplot as plt

brain_cancer_data = pd.read_csv('./BrainCancer.csv')
brain_cancer_data = brain_cancer_data.dropna(subset=['Diagnosis'])

def a():
    kmf = KaplanMeierFitter()
    kmf.fit(durations=brain_cancer_data['OS'], event_observed=brain_cancer_data['status'])
    kmf.plot_survival_function(ci_show = True)
    plt.title('Kaplan-Meier Survival Curve with ±1 SE')
    plt.xlabel('Time')
    plt.ylabel('Survival Probability')
    plt.legend()
    plt.savefig('./kaplan_meier_curve.png')
    plt.clf()

def b():
    n = 88 
    B = 200
    bootstrap_survival_functions = []
    kmf = KaplanMeierFitter()
    for _ in range(B):
        bootstrap_sample = brain_cancer_data.sample(n = n, replace = True)
        kmf.fit(durations = bootstrap_sample['OS'], event_observed = bootstrap_sample['status'])
        bootstrap_survival_functions.append(kmf.survival_function_)

    kmf.fit(durations=brain_cancer_data['OS'], event_observed=brain_cancer_data['status'])
    time_points = kmf.survival_function_.index
    bootstrap_survival_estimates = np.zeros((B, len(time_points)))
    for i, bootstrap_survival_function in enumerate(bootstrap_survival_functions):
        bootstrap_survival_estimates[i, :] = bootstrap_survival_function['KM_estimate'].reindex(time_points, method='ffill').fillna(method='bfill').values
    bootstrap_standard_errors = np.std(bootstrap_survival_estimates, axis=0)

    kmf.plot_survival_function(ci_show=False)
    plt.fill_between(
        time_points, 
        kmf.survival_function_['KM_estimate'].values.flatten() - bootstrap_standard_errors,
        kmf.survival_function_['KM_estimate'].values.flatten() + bootstrap_standard_errors,
        color='lightblue', alpha=0.5, label='Bootstrap SE'
    )
    plt.title('Kaplan-Meier Survival Curve with Bootstrap SE')
    plt.xlabel('Time')
    plt.ylabel('Survival Probability')
    plt.legend()
    plt.savefig('./kaplan_meier_curve_bootstrap_se.png')
    plt.clf()

def c():
    cph = CoxPHFitter()
    cph.fit(brain_cancer_data, duration_col='OS', event_col='status')
    cph.summary.round(3).to_csv('./coxphf_summary.csv', index=True)

def d():
    brain_cancer_data['KI'] = brain_cancer_data['KI'].replace(40, 60)
    ki_values = sorted(brain_cancer_data['KI'].unique())
    ki_values_label = [f'KI={ki_value}' for ki_value in ki_values]

    cph = CoxPHFitter()
    cph.fit(brain_cancer_data, duration_col='OS', event_col='status', strata=['KI'])
    baseline_survival = cph.baseline_survival_
    plt.step(baseline_survival.index, baseline_survival, where="post", label=ki_values_label)
    plt.title('Kaplan-Meier Survival Curves with Stratified KI values')
    plt.xlabel('Time')
    plt.ylabel('Survival Probability')
    plt.legend()
    plt.savefig('./kaplan_meier_curves_stratified_KI.png')
    plt.clf()

def d2():
    brain_cancer_data['KI'] = brain_cancer_data['KI'].replace(40, 60)
    ki_values = sorted(brain_cancer_data['KI'].unique())

    kmf = KaplanMeierFitter()
    plt.figure(figsize=(10, 6))
    for ki_value in ki_values:
        strata_data = brain_cancer_data[brain_cancer_data['KI'] == ki_value]
        kmf.fit(durations = strata_data['OS'], event_observed = strata_data['status'])
        plt.step(kmf.survival_function_.index, kmf.survival_function_['KM_estimate'].values.flatten(), where="post", label=f'KI={ki_value}')

    plt.title('Kaplan-Meier Survival Curves with Stratified KI values')
    plt.xlabel('Time')
    plt.ylabel('Survival Probability')
    plt.legend()
    plt.savefig('./kaplan_meier_curves_stratified_KI2.png')
    plt.clf()

a()
b()
c()
d()
d2()