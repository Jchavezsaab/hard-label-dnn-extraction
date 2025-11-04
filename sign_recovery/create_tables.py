import pandas as pd
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
import numpy as np
import os
import sys
from common import getSavePath
from sign_recovery_blackbox import parseArguments as parseArguments_blackbox
from sign_recovery_whitebox import parseArguments as parseArguments_whitebox

def get_alpha(votes_p, votes_m, n): 
    if  n==0: 
        return 0.9999999999999999999999999999999999999999
    p_observed = max(votes_m, votes_p)/n
    epsilon    = p_observed-0.5
    alpha      = np.exp(-2*epsilon**2*n)
    return alpha 

def analyze_df(df, whitebox=True): 

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
        'timePointMedian' : t_median,
        'timeTotal'       : df.total_execution_time.values[-1],
        'success'         : df['votes+'].values[-1]>df['votes-'].values[-1],
    }
    if '||vON||/||vOFF||' in df: _results['||vON||/||vOFF||'] = np.mean(df['||vON||/||vOFF||'])
    if 'speed_ON_mean1' in df: _results['sON>sOFF'] = 100.*(np.sum(df['speed_ON_mean1'].values>df['speed_OFF_mean1'])/len(df))
    if 'speed_ON_dirT1' in df: _results['sONdir>sOFFdir'] = 100.*(np.sum(df['speed_ON_dirT1'].values>df['speed_OFF_dirT1'])/len(df))
    if not whitebox:
        _results['sign'] = (df['Real Sign'][0])
        _results['correct'] = '✔' if (_results['sign']*_results['votes+'] > _results['sign']*_results['votes-'])  else "WRONG"
    return _results

if __name__=='__main__':

    # Whitebox Results
    print("================================")
    print("          WHITEBOX")
    print("================================")
    print()
    args = parseArguments_whitebox(sys.argv[1:])
    for lID in range(1, 5):
        results = []
        N_NEURONS = 64 if lID==4 else 256
        for nID in range(N_NEURONS):
            path = f"{getSavePath(args.model, lID, nID, args.runID, mkdir=False)}df.pkl"

            if not os.path.exists(path): 
                continue 
            dfnID = pd.read_pickle(path)

            _results = {'nID': nID} | analyze_df(dfnID)
            results.append(_results)

        if len(results)==0: 
            print(f"========= No results found for layer {lID} =========")
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
        print()

    # Blackbox Results
    print()
    print()
    print("================================")
    print("          BLACKBOX")
    print("================================")
    print()
    args = parseArguments_blackbox(sys.argv[1:])
    for lID in range(1, 5):
        results = []
        N_NEURONS = 8
        for nID in range(N_NEURONS):
            path = f"{getSavePath(args.model, lID, nID, args.runID, mkdir=False, whitebox=False)}df.pkl"

            if not os.path.exists(path): 
                continue 
            dfnID = pd.read_pickle(path)

            _results = {'nID': nID} | analyze_df(dfnID, whitebox=False)
            results.append(_results)

        if len(results)==0: 
            print(f"========= No results found for layer {lID} =========")
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
        df                           = df.sort_values(by='nID', ascending=True).reset_index(drop=True)
        df['finalCL']                   = df['finalCL'].apply(lambda x: f"{x:.1f}%")
        df['tperP']                     = df['timePointMedian'].apply(lambda x: f"{x*1e3:.0f}ms")
        df['tTotal']                    = df['timeTotal'].apply(lambda x: f"{x:.0f}s")
        df                              = df[['nID'
                                            , 'nDual'
                                            , 'nExp'
                                            , 'sign'
                                            , 'votes+'
                                            , 'votes-' 
                                            , 'finalCL'  
                                            , 'tperP'
                                            , 'tTotal'
                                            , 'correct'
                                            ]]
        print(df.to_markdown())
        df['finalCL'] = df['finalCL'].apply(lambda x: x.replace('%', '\%'))
        df['tperP']   = df['tperP'].apply(lambda x: x.replace('ms', '\,ms'))
        df['tTotal']  = df['tTotal'].apply(lambda x: x.replace('s', '\,s'))
        print()
   