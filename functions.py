# The functions in this notebook were partially taken from coursera course "Introduction to Portfolio Construction 
# and Analysis with Python" (https://www.coursera.org/learn/introduction-portfolio-construction-python) and modified. 
# Other funtions were created by the author of the thesis.

import pandas as pd
import numpy as np
import numpy_financial as npf
import math
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Next 3 functions are from the course
def annualized_returns(r, periods_per_year):
    compounded_growth = (1+r).prod()
    n_periods = r.shape[0]
    return compounded_growth**(periods_per_year/n_periods)-1

def compound(r):
    return np.expm1(np.log1p(r).sum())

def gbm(n_years=10, n_scenarios=1000, mu=0.07, sigma=0.15, steps_per_year=12, s_0=100.0):
    dt = 1/steps_per_year
    n_steps = int(n_years*steps_per_year) + 1
    rets_plus_1 = np.random.normal(loc=(1+mu*dt), scale=sigma*np.sqrt(dt), size=(n_steps, n_scenarios))
    rets_plus_1[0] = 1                              
    prices = s_0*pd.DataFrame(rets_plus_1).cumprod()
    return prices

# Next 4 functions are own
def drawdowns(acc_values):
    previous_peaks = acc_values.cummax()
    drawdowns = (acc_values - previous_peaks)/previous_peaks
    mean_drawdown = -(drawdowns.min().mean())
    worst_drawdown = -(drawdowns.min().min())
    return mean_drawdown, worst_drawdown

def irr(acc_values, c):
    n_scen = acc_values.shape[1]
    neg_ones = np.ones(40)*-c
    index = np.arange(0, n_scen, 1).tolist()
    neg_contrib = pd.DataFrame(data=neg_ones)
    neg_contrib = neg_contrib[np.repeat(neg_contrib.columns.values, n_scen)].set_axis(index, axis='columns')
    last_row = pd.DataFrame(data=acc_values.iloc[-1]).T
    rows_index = range(0, 41)
    new_df = pd.concat([neg_contrib, last_row]).set_axis(rows_index, axis='rows')
    irr = new_df.aggregate(npf.irr)
    return irr.mean()

def imp_goal(acc_values, ess_goal, prob=0.8):
    n_scen = acc_values.shape[1]
    index = np.arange(0, n_scen, 1).tolist()
    final_value = (acc_values.iloc[-1] - ess_goal.iloc[-1]).sort_values()
    final_value.index = index
    quantile = int(n_scen*(1-prob))
    return final_value[quantile]

def prob_imp_goal(acc_values, ess_goal, important_goal):
    final_value = (acc_values.iloc[-1] - ess_goal.iloc[-1]).sort_values()
    n_scen = acc_values.shape[1]
    final_value.index = np.arange(0, n_scen, 1).tolist()
    
    if (final_value.max() >= important_goal):
        dist = abs(final_value - important_goal)
        min_dist = dist.min()
        ind = dist[dist==min_dist].index.tolist()[0]
        prob = 1 - ind/int(acc_values.shape[1])
    else:
        prob = 0.0
    return prob

# Next 4 functions were taken from the course and modified
def allocation_1(psp_r, ghp_r, income, goal, m=1, psp_max=1):
    n_steps, n_scenarios = psp_r.shape        
    account_value = income.iloc[0]                               
    w_history = pd.DataFrame(index=psp_r.index, columns=psp_r.columns)  
    acc_value = pd.DataFrame(index=psp_r.index, columns=psp_r.columns)   
    for step in range(n_steps):                 
        goal_value = goal.iloc[step]        
        cushion = (account_value - goal_value)/account_value  
        psp_w = (m*cushion).clip(0, psp_max) 
        ghp_w = 1-psp_w
        psp_alloc = account_value*psp_w
        ghp_alloc = account_value*ghp_w
        account_value = psp_alloc*(1+psp_r.iloc[step]) + ghp_alloc*(1+ghp_r.iloc[step])
        w_history.iloc[step] = psp_w
        acc_value.iloc[step] = account_value
        account_value = account_value + income.iloc[step+1]
    return acc_value, w_history

def allocation_2(psp_r, ghp_r, income, goal, d_factors, m=1, psp_max=1):
    if d_factors.shape != psp_r.shape:        
        raise ValueError("PSP and ZC Prices must have the same shape")
    
    n_steps, n_scenarios = psp_r.shape        
    account_value = income.iloc[0]            
    w_history = pd.DataFrame(index=psp_r.index, columns=psp_r.columns)  
    acc_value = pd.DataFrame(index=psp_r.index, columns=psp_r.columns)   
    g_value = pd.DataFrame(index=psp_r.index, columns=psp_r.columns)
    
    for step in range(n_steps):                 
        goal_value = goal.iloc[step]*d_factors.iloc[step] 
        cushion = (account_value - goal_value)/account_value  
        psp_w = (m*cushion).clip(0, psp_max)
        ghp_w = 1-psp_w
        psp_alloc = account_value*psp_w
        ghp_alloc = account_value*ghp_w
        account_value = psp_alloc*(1+psp_r.iloc[step]) + ghp_alloc*(1+ghp_r.iloc[step])
        w_history.iloc[step] = psp_w
        acc_value.iloc[step] = account_value
        g_value.iloc[step] = goal_value
        account_value = account_value + income.iloc[step+1]
    
    return acc_value, w_history, g_value

def drawdown_allocation(psp_r, ghp_r, income, maxdd, m=1, psp_max=1):
    n_steps, n_scenarios = psp_r.shape
    account_value = income.iloc[0]
    peak_value = income.iloc[0] 
    w_history = pd.DataFrame(index=psp_r.index, columns=psp_r.columns)
    acc_value = pd.DataFrame(index=psp_r.index, columns=psp_r.columns)
    fl_value = pd.DataFrame(index=psp_r.index, columns=psp_r.columns)
    peak_val = pd.DataFrame(index=psp_r.index, columns=psp_r.columns)
    
    for step in range(n_steps):
        floor_value = (1-maxdd)*peak_value   
        cushion = (account_value - floor_value)/account_value
        psp_w = (m*cushion).clip(0, psp_max)       
        ghp_w = 1-psp_w
        psp_alloc = account_value*psp_w
        ghp_alloc = account_value*ghp_w
        account_value = psp_alloc*(1+psp_r.iloc[step]) + ghp_alloc*(1+ghp_r.iloc[step])
        w_history.iloc[step] = psp_w
        acc_value.iloc[step] = account_value
        fl_value.iloc[step] = floor_value
        peak_val.iloc[step] = peak_value
        
        account_value = account_value + income.iloc[step+1]
        peak_value = np.maximum(peak_value, account_value)
        
    return acc_value, w_history, fl_value, peak_val

def fixedmix_allocation(psp_r, ghp_r, psp_w, income):
    n_steps, n_scenarios = psp_r.shape
    account_value = income.iloc[0]                                  
    acc_value = pd.DataFrame(index=psp_r.index, columns=psp_r.columns) 
    
    for step in range(n_steps):
        ghp_w = 1-psp_w
        psp_alloc = account_value*psp_w
        ghp_alloc = account_value*ghp_w
        account_value = psp_alloc*(1+psp_r.iloc[step]) + ghp_alloc*(1+ghp_r.iloc[step])
        acc_value.iloc[step] = account_value
        account_value = account_value + income.iloc[step+1]
    
    return acc_value

# All next functions are own
def statistics_allocation_income(acc_values, goal, name='Statistics'):
    terminal_wealth = acc_values.iloc[-1]
    goal_value = goal.iloc[-1]
    breach = terminal_wealth < goal_value   
    p_breach = breach.mean() if breach.sum() > 0 else np.nan
    e_short = (goal_value-terminal_wealth[breach]).mean() if breach.sum() > 0 else np.nan
    worst_short = (goal_value-terminal_wealth[breach]).max() if breach.sum() > 0 else np.nan
    mean_drawdown, worst_drawdown = drawdowns(acc_values)
    
    sum_stats = pd.DataFrame.from_dict({
        'Min. final account value': terminal_wealth.min(),
        'Mean final account value': terminal_wealth.mean(),
        'Max. final account value': terminal_wealth.max(),
        'Final account value volatility': terminal_wealth.std(),
        'Probability of breach': p_breach,
        'Expected shortfall':e_short,
        'Worst shortfall': worst_short,
        'Expected drawdown': mean_drawdown, 
        'Worst drawdown': worst_drawdown
    }, orient="index", columns=[name]).round(3)
    
    return sum_stats

def acc_value_10multipliers(rets_eq, rets_cash, income, ess_goal, multipliers, psp_max=1):
    account_value_list, w_list = [], []
    
    for m in multipliers:
        account_value, w = allocation_1(rets_eq, rets_cash, income, ess_goal, m=m, psp_max=psp_max)
        account_value_list.append(account_value)
        w_list.append(w)
        
    return account_value_list, w_list

def acc_value_10multipliers_2(rets_eq, rets_cash, income, ess_goal, df, multipliers, psp_max=1):
    account_value_list, w_list = [], []
    for m in multipliers:
        account_value, w, g = allocation_2(rets_eq, rets_cash, income, ess_goal, df, m=m, psp_max=psp_max)
        account_value_list.append(account_value)
        w_list.append(w)
        
    return account_value_list, w_list

def acc_value_10multipliers_dd(rets_eq, rets_cash, income, maxdd, multipliers, psp_max=1):
    account_value_list, w_list = [], []
    for m in multipliers:
        account_value, w, fl_v, peak  = drawdown_allocation(rets_eq, rets_cash, income, maxdd, m=m, psp_max=psp_max)
        account_value_list.append(account_value)
        w_list.append(w)
        
    return account_value_list, w_list

def statistics_table(acc_values, ess_goal, import_goal, prob_1, prob_2, c):
    multipliers = np.arange(1, 11, 1).tolist()
    stats = []
    irr_stats = []
    import_goal_prob = []
    imp_goal_stats_1 = []
    imp_goal_stats_2 = []

    for m in multipliers:
        st = statistics_allocation_income(acc_values[m-1], ess_goal)
        stats.append(st)
        irr_st = irr(acc_values[m-1], c)
        irr_stats.append(irr_st)
        imp_goal_prob = prob_imp_goal(acc_values[m-1], ess_goal, import_goal)
        import_goal_prob.append(imp_goal_prob)
        imp_goal_st_1 = imp_goal(acc_values[m-1], ess_goal, prob=prob_1)
        imp_goal_stats_1.append(imp_goal_st_1)
        imp_goal_st_2 = imp_goal(acc_values[m-1], ess_goal, prob=prob_2)
        imp_goal_stats_2.append(imp_goal_st_2)
        
    columns = np.arange(0, 10, 1).tolist()
    stats_1 = pd.concat([stats[0], stats[1], stats[2], stats[3], stats[4], stats[5], stats[6], stats[7], stats[8], stats[9]], axis=1).set_axis(columns, axis=1)
    stats_2 = pd.DataFrame(data=irr_stats).round(4).T
    stats_3 = pd.DataFrame(data=import_goal_prob).round(4).T
    stats_4 = pd.DataFrame(data=imp_goal_stats_1).round(4).T
    stats_5 = pd.DataFrame(data=imp_goal_stats_2).round(4).T
    column_names = ['GBI m=1', 'GBI m=2', 'GBI m=3', 'GBI m=4', 'GBI m=5', 'GBI m=6', 'GBI m=7', 'GBI m=8', 'GBI m=9', 'GBI m=10']
    row_names = ['IRR', 'Min. final account value', 'Mean final account value', 'Max. final account value', 
                 'Final account value volatility', 'Probability of breach', 'Expected shortfall', 'Worst shortfall', 
                 'Expected drawdown', 'Worst drawdown', 'Probability of a set goal', 'Goal with a set probability 1', 
                 'Goal with a set probability 2']
    stats_6 = pd.concat([stats_2, stats_1, stats_3, stats_4, stats_5], axis=0).round(4).set_axis(column_names, axis=1).set_axis(row_names, axis=0)
    stats_6 = stats_6.fillna(0)
    
    return stats_6.round(4)

def plot(st, st_c, st_ct, color, title, legend, status):
    plt.title(title)
    plt.xlabel('Multiplier')
    plt.grid(ls='--')
    ls = ['-', '--', '-.']
    multipliers = np.arange(1, 11, 1).tolist()
    if status == 1:
        plt.plot(multipliers, st.iloc[-1], color=color, ls=ls[0])
        plt.plot(multipliers, st_c.iloc[-1], color=color, ls=ls[1])
        plt.plot(multipliers, st_ct.iloc[-1], color=color, ls=ls[2])
        ylabel = 'Amount'
        # plt.ylim(0, 160)
    elif status == 2:
        plt.plot(multipliers, st.iloc[-2], color=color, ls=ls[0])
        plt.plot(multipliers, st_c.iloc[-2], color=color, ls=ls[1])
        plt.plot(multipliers, st_ct.iloc[-2], color=color, ls=ls[2])
        ylabel = 'Amount'
        # plt.ylim(0, 160)
    else:
        plt.plot(multipliers, st.iloc[-3], color=color, ls=ls[0])
        plt.plot(multipliers, st_c.iloc[-3], color=color, ls=ls[1])
        plt.plot(multipliers, st_ct.iloc[-3], color=color, ls=ls[2])
        ylabel = 'Probability'
        plt.ylim(0, 1.1)
    plt.legend(legend, loc="lower center", bbox_to_anchor=(0.5, -0.3), ncol=3)
    plt.xticks(multipliers)
    
    return plt.show()

def plot_distribution(acc_value, stats_table, m, color, ess_goal, import_goal, title, init_cap, ess=True):
    if ess == True:
        acc_value[m-1].iloc[-1].plot.hist(bins=70, ec='w', color=color[0], title=title, figsize=(8, 5))
        plt.axvline(init_cap, ls='-', lw=2, color=color[1])
        plt.axvline(ess_goal.iloc[-1][0], ls='--', lw=2, color=color[1])
        plt.legend(['Invested capital', 'Essential goal'],  loc="lower center", bbox_to_anchor=(0.5, -0.15), ncol=2)
        plt.annotate(f'Essen. goal: {round(ess_goal.iloc[-1][0], 1)} fr. (\'000)', xy=(0.6, 0.95), xycoords='axes fraction', fontsize=12)
        plt.annotate(f'Probability of breach: {stats_table.iloc[-8][m-1]*100}%', xy=(0.6, 0.9), xycoords='axes fraction', fontsize=12)
        plt.annotate(f'Exp. shortfall: {int(stats_table.iloc[-7][m-1])} fr. (\'000)', xy=(0.6, 0.85), xycoords='axes fraction', fontsize=12)
        plt.annotate(f'Worst shortfall: {int(stats_table.iloc[-6][m-1])} fr. (\'000)', xy=(0.6, 0.8), xycoords='axes fraction', fontsize=12)
        
    else:
        (acc_value[m-1].iloc[-1]-ess_goal.iloc[-1]).plot.hist(bins=70, ec='w', color=color[0], title=title, figsize=(8, 5))
        plt.axvline(import_goal, ls='--', lw=2, color=color[1])
        plt.annotate(f'Import. goal: {int(import_goal)} fr. (\'000)', xy=(0.6, 0.95), xycoords='axes fraction', fontsize=12)
        plt.annotate(f'Probability: {round(stats_table.iloc[-3][m-1]*100, 2)}%', xy=(0.6, 0.9), xycoords='axes fraction', fontsize=12)
        plt.xlabel('Final account value minus essential goal')
    
    return plt.show()

def plot_evolution(acc_value, m, title):
    acc_value[m-1].plot(legend=False, grid=True, title=title, figsize=(12, 5))
    plt.xlim(left=0, right=480)
    plt.ylim(bottom=0)
    ticks = np.arange(0, 481, 24).tolist()
    plt.xticks(ticks)
    plt.xlabel('Months')
    plt.ylabel('Amount')
    return plt.show()