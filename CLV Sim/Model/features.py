import numpy as np
from scipy.interpolate import interp1d, Akima1DInterpolator
import pandas as pd
from datetime import timedelta

#convert to decimal odds
def american_to_decimal(sequence):
    dec_seq = sequence.copy()
    dec_seq[dec_seq > 0] = (dec_seq[dec_seq > 0]/100.0)+1.0
    dec_seq[dec_seq < 0] = (100.0/abs(dec_seq[dec_seq<0])) + 1.0
    return dec_seq

#convert to break even percentage
def implied_probability(sequence):
    imp_seq = sequence.copy()
    imp_seq[imp_seq > 0] = 100.0/(100+(imp_seq[imp_seq > 0]))
    imp_seq[imp_seq < 0] = abs(imp_seq[imp_seq <0 ])/(abs(imp_seq[imp_seq < 0])+100.0)
    return imp_seq

#interpolate for a defined length
def fixed_length(seq, output_len, input_kind):
    x_old = np.arange(len(seq))
    x_new = np.linspace(0, len(seq)-1, output_len)

    if input_kind == 'akima':
        f = Akima1DInterpolator(x_old, seq)
    else:
        f = interp1d(x_old, seq, kind=f'{input_kind}')
    return f(x_new)



def fixed_length_timelin(seq, output_len, input_kind, timestamps):
    x_old = np.array(timestamps)
    #clean dups
    if len(x_old) != len(np.unique(x_old)):
        for i in range(len(x_old)):
            if i == 0:
                continue
            elif x_old[i] == x_old[i-1]:
                x_old[i] = x_old[i]+30.0
                
    x_new = np.linspace(x_old.min(), x_old.max(), output_len)

    if input_kind == 'akima':
        f = Akima1DInterpolator(x_old, seq)
    else:
        f = interp1d(x_old, seq, kind=f'{input_kind}')
    return f(x_new)


#split values into two rows to hold date
def split_for_datetime(frame):
    dfC = frame.dropna(axis=1)
    dfC = pd.concat([dfC, dfC], ignore_index=True)
    dfC.loc[1] = dfC.loc[1].apply(
    lambda x: x.split(' - price: ')[1].strip()
    if isinstance(x, str) and 'price: ' in x else x
    )
    dfC.loc[0] = dfC.loc[0].apply(
    lambda x: x.split(' - price: ')[0].strip() 
    if isinstance(x, str) and 'price: ' in x else x
    )
    dfC.loc[0] = dfC.loc[0].apply(
    lambda x: pd.to_datetime(x)
    )

    return dfC


#get the column name of the timestep at 48 hrs out based on time
def time_for_75(frame):
    end_time = pd.to_datetime(frame.iloc[0, -1].split(' - price:')[0])

    target_time = end_time-pd.Timedelta(hours=48)
    row_times = frame.iloc[0, :-1]

    time_diffs = row_times.apply(lambda x: abs(pd.to_datetime(str(x).split(' - price:')[0]) - target_time))
    closest_col = time_diffs.idxmin()

    return closest_col


#pull time since open and time until event
def timeInput(frame):
    eventDate = pd.to_datetime(frame.iloc[0, 2]) + timedelta(days=1)
    frame = frame.iloc[:, 4:]
    dfC = frame.dropna(axis=1)
    dfC = pd.concat([dfC, dfC], ignore_index=True)
    dfC.loc[1] = dfC.loc[1].apply(
    lambda x: x.split(' - price: ')[1].strip()
    if isinstance(x, str) and 'price: ' in x else x
    )
    dfC.loc[0] = dfC.loc[0].apply(
    lambda x: x.split(' - price: ')[0].strip() 
    if isinstance(x, str) and 'price: ' in x else x
    )
    dfC.loc[0] = dfC.loc[0].apply(
    lambda x: pd.to_datetime(x)
    )

    empty_row = pd.Series([np.nan] * dfC.shape[1], index=dfC.columns)
    dfC = pd.concat([dfC.iloc[:1], empty_row.to_frame().T, dfC.iloc[1:]]).reset_index(drop=True)
    dfC = pd.concat([dfC.iloc[:1], empty_row.to_frame().T, dfC.iloc[1:]]).reset_index(drop=True)

    #populate index 1 with time since open, 2 with time until event
    timeOpen = dfC.iloc[0,0]
    dfC.loc[1] = dfC.loc[0].apply(
        lambda x: (x- timeOpen).total_seconds()
    )
    dfC.loc[2] = dfC.loc[0].apply(
        lambda x: (eventDate-x).total_seconds()
    )

    dfC = dfC.drop(index=0).reset_index(drop=True)

    return dfC




def linspace_stamps(frame):
    #frame = frame.iloc[:, 4:]
    dfC = frame.dropna(axis=1)
    dfC = pd.concat([dfC, dfC], ignore_index=True)
    dfC.loc[1] = dfC.loc[1].apply(
    lambda x: x.split(' - price: ')[1].strip()
    if isinstance(x, str) and 'price: ' in x else x
    )
    dfC.loc[0] = dfC.loc[0].apply(
    lambda x: x.split(' - price: ')[0].strip() 
    if isinstance(x, str) and 'price: ' in x else x
    )
    dfC.loc[0] = dfC.loc[0].apply(
    lambda x: pd.to_datetime(x)
    )

    empty_row = pd.Series([np.nan] * dfC.shape[1], index=dfC.columns)
    dfC = pd.concat([dfC.iloc[:1], empty_row.to_frame().T, dfC.iloc[1:]]).reset_index(drop=True)

    #populate index 1 with time since open, 2 with time until event
    timeOpen = dfC.iloc[0,0]
    dfC.loc[1] = dfC.loc[0].apply(
        lambda x: (x- timeOpen).total_seconds()
    )

    dfC = dfC.drop(index=0).reset_index(drop=True)
    dfC = dfC.drop(index=1).reset_index(drop=True)

    return dfC