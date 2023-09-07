# -------------------------Torch Modules-------------------------
from __future__ import print_function
import numpy as np
import torch
from sklearn.linear_model import BayesianRidge
from matplotlib import pyplot as plt


# ---------------------------Variables---------------------------
# parameters
Iterations = 3
frequency = 377 ## rad/s
voltage = torch.Tensor([11,33,69]) ## KV 11kv 33kv , 69kv
real = 2000 ## KW
pf  = 0.95 # power factor
Apower =  real / pf # aparent power
sizes =1024 # 24 hours day data generation
KVARs = np.sqrt(Apower ** 2 - real ** 2) # kvars
print("Target KVARs: ", KVARs)


# ------------------Commands to Prepare Dataset------------------
def create_powerfactor_dataset(sizes,real,KVARs,voltage,frequency):
    ## generating the power factor with respect to time
    thetas = np.arange(sizes) ## creating list of theta 
    data_pf = (90 + 10*np.cos(thetas/2) - 0.5 + 0.5* (2*np.random.rand(sizes) - 1))/100  # computing power factor dataset

    Apower =  real /data_pf
    new_KVARs = np.sqrt(Apower ** 2 - real ** 2)
    dels  = (new_KVARs - KVARs) ## required Kvar
    voltages = voltage.repeat(sizes)
    
    for k in range (len(dels)):
        if dels[k] <  0:
            dels[k] = 0
        else:
            dels [k] = dels[k] /(frequency * (voltages[k] ** 2)) * 1000 # Capacitance in F, not µF
    
    return torch.Tensor(data_pf).view(sizes,1), torch.Tensor(dels).view(sizes,1)


# -----------------Using BayesianRidge Regression----------------
regressor_11 = BayesianRidge()

regressor_33= BayesianRidge()

regressor_69 = BayesianRidge()


# ------------------------Plotting Function----------------------
def plot_func(data, y, target, v_level):
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams.update({'font.size': 21})
    color1 = 'purple'
    color2 = 'dodgerblue'
    color3 = 'orange'

    fig, ax1 = plt.subplots(figsize=(12, 6))
    line1 = ax1.plot(y[-24:], color=color1, linewidth='1')
    line2 = ax1.plot(target[-24:], color=color3, linewidth='1')
    ax1.set_ylabel('Capacitor Bank values, F')
    ax1.tick_params(axis='y')
    ax2 = ax1.twinx()
    line3 = ax2.plot(data[-24:], color=color2, linewidth='1')
    ax2.set_ylabel('Power Factor')
    ax2.tick_params(axis='y')

    lines = line1 + line2 + line3
    labels = ['Capacitor Bank Predicted', 'Capacitor Bank Ideal', 'Power Factor for '+str(v_level)+'kV']
    ax2.legend(lines, labels);
    ax1.set_xlabel('Time (Datapoints)')

    ax1.plot(0, 19 if v_level==11 else (2.2 if v_level==33 else 0.54), marker="^", ms=12, color="k", transform=ax1.get_yaxis_transform(), clip_on=False)
    ax2.plot(1, 1.01, marker="^", ms=12, color="k", transform=ax2.get_yaxis_transform(), clip_on=False)
    ax1.set_ylim(-0.01, 19 if v_level == 11 else (2.2 if v_level == 33 else 0.54))
    ax2.set_ylim(0.79, 1.01)

    ax2.spines['top'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    fig.tight_layout()
    plt.savefig('./results/' + str(v_level) + 'control.png', dpi=300)
    plt.close()


# ----------------------Training and Testing---------------------
for i in range(Iterations):
    #Training
    ## 11KV case
    if i == 0:
        data_11, target_11 = create_powerfactor_dataset(sizes,real,KVARs,voltage[i],frequency)  ## creating the dataset
        regressor_11.fit(data_11[:1000], target_11[:1000].ravel())  ## training
        y_11 = regressor_11.predict(data_11)   ## testing
        y_11[y_11<0] = 0

        plot_func(data_11, y_11, target_11, 11)

    if i == 1: ## 33kv case
        data_33, target_33 = create_powerfactor_dataset(sizes,real,KVARs,voltage[i],frequency)  ## creating the dataset

        regressor_33.fit(data_33[:1000], target_33[:1000].ravel())  ## training
        y_33 = regressor_33.predict(data_33)   ## testing
        y_33[y_33<0] = 0

        plot_func(data_33, y_33, target_33, 33)

    if i == 2: # 69 Kv case
        data_69, target_69 = create_powerfactor_dataset(sizes,real,KVARs,voltage[i],frequency)  ## creating the dataset
        
        regressor_69.fit(data_69[:1000], target_69[:1000].ravel())  ## training

        y_69 = regressor_69.predict(data_69)   ## testing
        y_69[y_69<0] = 0

        plot_func(data_69, y_69, target_69, 69)