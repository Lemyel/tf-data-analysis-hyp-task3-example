import pandas as pd
import numpy as np

chat_id = 280785885 # Ваш chat ID, не меняйте название переменной

def solution(control: pd.Series, test: pd.Series) -> bool:
    n_control = len(control)
    n_test = len(test)
    
    mean_control = np.mean(control)
    mean_test = np.mean(test)
    
    var_control = np.var(control, ddof=1)
    var_test = np.var(test, ddof=1)
    
    se = np.sqrt(var_control / n_control + var_test / n_test)
    
    t_stat = (mean_control - mean_test) / se
    
    df = ((var_control / n_control + var_test / n_test) ** 2) / \
         ((var_control / n_control) ** 2 / (n_control - 1) + (var_test / n_test) ** 2 / (n_test - 1))
    
    alpha = 0.05
    
    t_crit = np.abs(np.percentile(np.random.standard_t(df, size=1000000), 100 * (1 - alpha / 2)))
    
    return np.abs(t_stat) > t_crit
