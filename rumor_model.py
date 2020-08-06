# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from scipy.integrate import odeint
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
sns.set()


def pend_ini(y, t, b, k):
    S, I, R = y
    dsdt = -b * S * I
    didt =  b * S * I - k * I 
    drdt = k * I
    dydt = [dsdt, didt, drdt]
    return dydt

def pend_v1(y, t, b, k_i, k_s):
    S, I, R = y
    dsdt = -b * S * I
    didt =  b * S * I - k_i * I  - k_s * R * I
    drdt = k_i * I  + k_s * R * I
    dydt = [dsdt, didt, drdt]
    return dydt

def pend_v2(y, t, b, k_i, k_s):
    S1, S2, I1, I2, R = y
    ds1dt = -b * S1 * (I1 + I2)
    ds2dt = -b * S2 * (I1 + I2)
    di1dt =  b * S1 * (I1 + I2) -  k_i * I1 - k_s * R * I1
    di2dt =  b * S2 * (I1 + I2) - k_s * R * I2
    drdt = k_i * I1 + k_s * R * I1 + k_s * R * I2
    dydt = [ds1dt, ds2dt, di1dt, di2dt, drdt]
    return dydt


def plot_curve(sol, sol_c=None):
    df = pd.DataFrame(sol, columns=['S',  'I', 'R']).astype(int)
    df.index = df.index + 1
    ax = sns.lineplot(data=df) #plot the curves
    ax.set_xlabel('Day')
    ax.set_ylabel('Number of Students')   

def plot_table(sol, sol_five_var=1):
    if type(sol_five_var) != int:
        df = pd.DataFrame(sol_five_var, columns=['S1', 'S2', 'I1','I2', 'R']).astype(int)
        df.index = df.index + 1
        df.index.name = 'Day'
        keep = np.arange(0, 29, 2)
        df= df.iloc[keep].reset_index().T
        
        fig, ax = plt.subplots()
        
        # hide axes
        fig.patch.set_visible(False)
        ax.axis('off')
        ax.axis('tight')
        ax.table(cellText=df.values, rowLabels=df.index, loc='center')
        fig.tight_layout()
        plt.show()
        
    df = pd.DataFrame(sol, columns=['S',  'I', 'R']).astype(int)
    df.index = df.index + 1
    df.index.name = 'Day'
    keep = np.arange(0, 29, 2)
    df= df.iloc[keep].reset_index().T
    
    fig, ax = plt.subplots()
    
    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    ax.table(cellText=df.values, rowLabels=df.index, loc='center')
    fig.tight_layout()
    plt.show()
    
def plot(b=2/163, k_i=0.2, k_s=2/163, y0=[163, 1, 4], t=np.linspace(1, 30, 30), table=False, model='initial'):
    if model == 'initial':
        sol_c = odeint(pend_ini, y0, t, args=(b, k_i))
    elif model == 'v1':
        sol_c = odeint(pend_v1, y0, t, args=(b, k_i, k_s))
    elif model == 'v2':
        y0=[163 * 0.9, 163 * 0.1, 1, 0, 4]
        sol = odeint(pend_v2, y0, t, args=(b, k_i, k_s))
        sol = np.array(sol).T
        sol_c = np.zeros((3, sol.shape[1]))
        sol_c[0] = sol[0] + sol[1]
        sol_c[1] = sol[2] + sol[3]
        sol_c[2] = sol[4]
        sol_c = sol_c.T
        sol = sol.T
    if table:
        if model != 'v2':
            plot_table(sol_c)
        else:
            plot_table(sol_c, sol_five_var=sol)
    plot_curve(sol_c)
    
    
# plot the model under the initial assumption
plot(k_i=0.2, table=True) 

# plot mode variation 1
plot(k_i=0.2, model='v1', table=True) 

# plot mode variation 2
plot(model='v2', table=True)

