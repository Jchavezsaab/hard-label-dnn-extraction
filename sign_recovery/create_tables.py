import pandas as pd
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
import numpy as np
import os
import sys

def get_alpha(votes_p, votes_m, n): 
    if  n==0: 
        return 0.9999999999999999999999999999999999999999
    p_observed = max(votes_m, votes_p)/n
    epsilon    = p_observed-0.5
    alpha      = np.exp(-2*epsilon**2*n)
    return alpha 

def analyze_df(df): 

    df['votes+'] = df['Vote dOFF>dON'].cumsum()
    df['votes-'] = (~df['Vote dOFF>dON']).cumsum()
    df['alpha']  = df.apply(lambda x: get_alpha(x['votes+'], x['votes-'], x['nExp']), axis=1)
    df['CL']     = 100*(1-df['alpha'])

    t_median = np.median(df.subpoint_time_seconds)

    _results = {
        'nDual'           : df.dual_point_id.values[-1],
        'nExp'            : len(df),
        'votes+'          : df['votes+'].values[-1],
        'votes-'          : df['votes-'].values[-1],
        'finalCL'         : df['CL'].values[-1],
        '||vON||/||vOFF||': np.mean(df['||vON||/||vOFF||']),
        'sON>sOFF': 100.*(np.sum(df['speed_ON_mean1'].values>df['speed_OFF_mean1'])/len(df)), 
        'sONdir>sOFFdir': 100.*(np.sum(df['speed_ON_dirT1'].values>df['speed_OFF_dirT1'])/len(df)), 
        'timePointMedian' : t_median,
        'timeTotal'       : df.total_execution_time.values[-1],
        'success'         : df['votes+'].values[-1]>df['votes-'].values[-1],
        }

    return _results

if __name__=='__main__':

    for lID in range(1, 5):
        results = []
        N_NEURONS = 64 if lID==4 else 256
        for nID in range(N_NEURONS):
            path = f"results/model_cifar10_3x256_64_10_float64/layerID_{lID}/neuronID_{nID}/df.pkl" 

            if not os.path.exists(path): 
                continue 
            dfnID = pd.read_pickle(path)

            _results = {'nID': nID} | analyze_df(dfnID)
            results.append(_results)

        if len(results)==0: 
            print(f"No results found for layer {lID}")
            continue
        df = pd.DataFrame(results)

        correct_nIDs = df[df.success==True].nID.values
        wrong_nIDs   = df[df.success==False].nID.values

        print(f"========= RECOVERY RESULTS: HIDDEN LAYER {lID} ================")
        print(f"""
        correctly recovered  : \t {len(correct_nIDs)}/{len(df)} \t
                  CL         : \t min: {df['finalCL'].min():.2f}% \t mean: {df['finalCL'].mean():.2f}% \t max: {df['finalCL'].max():.2f}%
        mean      Ndual      : \t ({df['nDual'].mean():.0f}+-{df['nDual'].std():.0f})
        mean      timePerDual: \t ({np.mean(df.timePointMedian):.2f}+-{np.std(df.timePointMedian):.2f}) seconds
        mean      timeTotal  : \t ({np.mean(df.timeTotal):.0f}+-{np.std(df.timeTotal):.0f}) seconds
        Missing   neurons    : \t {np.setdiff1d(np.arange(N_NEURONS), df.nID.values)}
        """)

        pd.options.display.precision = 2
        df                           = df.sort_values(by='finalCL', ascending=False).reset_index(drop=True)
        df['finalCL']                   = df['finalCL'].apply(lambda x: f"{x:.1f}%")
        df['sON>sOFF']                  = df['sON>sOFF'].apply(lambda x: f"{x:.1f}%")
        df['sONdir>sOFFdir']            = df['sONdir>sOFFdir'].apply(lambda x: f"{x:.1f}%")
        df['tperP']                     = df['timePointMedian'].apply(lambda x: f"{x*1e3:.0f}ms")
        df['tTotal']                    = df['timeTotal'].apply(lambda x: f"{x:.0f}s")
        df                              = df[['nID'
                                            , 'nDual'
                                            , 'nExp'
                                            , 'votes+'
                                            , 'votes-' 
                                            , '||vON||/||vOFF||'
                                            , 'sON>sOFF'
                                            , 'sONdir>sOFFdir'
                                            , 'finalCL'  
                                            , 'tperP'
                                            , 'tTotal'
                                            ]]
        print(df.to_markdown())
        df['finalCL'] = df['finalCL'].apply(lambda x: x.replace('%', '\%'))
        df['tperP']   = df['tperP'].apply(lambda x: x.replace('ms', '\,ms'))
        df['tTotal']  = df['tTotal'].apply(lambda x: x.replace('s', '\,s'))
        # print(df.to_latex())
        #df.to_pickle('/results_exp_create_tables/{model_short}_{layerID}.pkl')
        #df.to_markdown('./results_exp_create_tables/{model_short}_{layerID}.md')