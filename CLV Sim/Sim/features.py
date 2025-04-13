import numpy as np
from scipy.interpolate import interp1d
import pandas as pd

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
    row_times = frame.iloc[0]

    time_diffs = row_times.apply(lambda x: abs(pd.to_datetime(str(x).split(' - price:')[0]) - target_time))
    closest_col = time_diffs.idxmin()

    return closest_col


def extract_sequence(frame):
    frame = frame.iloc[:, 4:]
    frame = frame.dropna(axis=1)
    split_name = time_for_75(frame)
    split_index = frame.columns.get_loc(split_name)

    frame = frame.applymap(lambda x: x.split(' - price: ')[1].strip()
            if isinstance(x, str) and 'price: ' in x else x
        )
    
    if(split_index == 0):
        split_index+=1

    sequence = frame.iloc[0, :split_index].values
    sequence = np.array(sequence, dtype=float)

    return sequence

