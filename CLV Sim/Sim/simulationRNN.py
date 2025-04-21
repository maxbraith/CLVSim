import pandas as pd
import tensorflow as tf 
import gc
import joblib
from features import extract_sequence, implied_probability, american_to_decimal, fixed_length, split_for_datetime, time_for_75
from tensorflow.keras.models import load_model
import numpy as np


dftp = pd.read_csv("timestampsDatetimeA.csv", header=None, names=range(231))

dftp.columns = dftp.iloc[0]
dftp = dftp[1:]
dftp.reset_index(drop=True, inplace=True)


df = pd.read_csv('simData48Clean.csv')


df = df.dropna().reset_index(drop=True)


def calculate_profit(wager, price):
    if price > 0:
        profit = (price/100)*wager
    else:
        profit = (100/abs(price))*wager
    return profit



import random


def perform_sim(wager, model, size):
    bankroll = wager * len(df)

    profit_dict = {
    'favorite_at_open': bankroll,
    'favorite_at_close': bankroll,
    'favorite_at_48h': bankroll,

    'favorite_ML': bankroll,

    'underdog_at_open': bankroll,
    'underdog_at_close': bankroll,
    'underdog_at_48h': bankroll,

    'underdog_ML': bankroll,

    'favorite_random': bankroll,
    'underdog_random': bankroll,
    
    'true_random': bankroll
    }   

    for index, row in df.iterrows():
        print(index)
        fighter1 = row['fighter1']
        fighter2 = row['fighter2']
        winner = row['winner']

        model = load_model('simpleRNN.keras')


        #get open line
        open1 = int(row['fighter1 - open'].split('price: ')[1])
        open2 = int(row['fighter2 - open'].split('price: ')[1])

        close1 = int(row['fighter1 - close'].split('price: ')[1])
        close2 = int(row['fighter2 - close'].split('price: ')[1])

        forty81 = int(row['fighter1 - 48h'].split('price: ')[1])
        forty82 = int(row['fighter2 - 48h'].split('price: ')[1])

        lines = {
            fighter1: [open1, forty81, close1],
            fighter2: [open2, forty82, close2]
        }

        fighter1_avg = sum(lines[fighter1])/len(lines[fighter1])
        fighter2_avg = sum(lines[fighter2])/len(lines[fighter2])

        #get indice for fav and dog
        if fighter1_avg < fighter2_avg:
            favorite_mean = 1
            dog_mean = 2
        else:
            favorite_mean = 2
            dog_mean = 1
            
        #find the sequence
        fav_frame = dftp[(dftp['fighter'] == row[f'fighter{favorite_mean}']) 
                    & (dftp['opponent'] == row[f'fighter{dog_mean}'])
                    & (dftp['date'] == row['date'])]
        
        dog_frame = dftp[(dftp['fighter'] == row[f'fighter{dog_mean}']) 
                    & (dftp['opponent'] == row[f'fighter{favorite_mean}'])
                    & (dftp['date'] == row['date'])]
        
        #extract sequences
        fav_sequence = extract_sequence(fav_frame)
        dog_sequence = extract_sequence(dog_frame)

        #generate input data
        fav_seq = fixed_length(fav_sequence, size, 'linear')
        dog_seq = fixed_length(dog_sequence, size, 'linear')

        fav_action = model.predict(np.array(fav_seq, dtype=np.float32).reshape(1, 20, 1))[0]
        dog_action = model.predict(np.array(dog_seq, dtype=np.float32).reshape(1, 20, 1))[0]

        #Test changed to decimal
        if(fav_action >= 0.5):
            fav_ML = lines[row[f'fighter{favorite_mean}']][2]
        else:
            fav_ML = lines[row[f'fighter{favorite_mean}']][1]

        if(dog_action >= 0.5):
            dog_ML = lines[row[f'fighter{dog_mean}']][2]
        else:
            dog_ML = lines[row[f'fighter{dog_mean}']][1]

        line_list = lines[winner]

        if fav_ML in line_list:
            profit_dict['favorite_ML']+=calculate_profit(wager, fav_ML)
        else:
            profit_dict['favorite_ML']-=wager
        
        if dog_ML in line_list:
            profit_dict['underdog_ML']+=calculate_profit(wager, dog_ML)
        else:
            profit_dict['underdog_ML']-=wager
            

        for_random = [open1, open2, forty81, close1, forty82, close2]

        neg_lines = []
        pos_lines = []
        for line in for_random:
            if(line > 0):
                pos_lines.append(line)
            else:
                neg_lines.append(line)

        if(len(pos_lines) >0):
            #dog random blindly takes a line
            random_dog = random.choice(pos_lines)
        else:
            #no pos line default to closest to pos
            random_dog = max(for_random)

        if(len(neg_lines)> 0):
            #fav random blindly takes a line
            random_fav = random.choice(neg_lines)
        else:
            random_fav = min(for_random)


        #if random fav picked a winning line
        if random_fav in line_list:
            profit_dict['favorite_random']+=calculate_profit(wager, random_fav)
        else:
            profit_dict['favorite_random']-=wager

        #if random dog picked a winning line
        if random_dog in line_list:
            profit_dict['underdog_random']+=calculate_profit(wager, random_dog)
        else:
            profit_dict['underdog_random']-=wager



        #true random
        random_fighter = row[f'fighter{random.randint(1,2)}']
        random_line = random.randint(0,2)
        true_random = lines[random_fighter][random_line]
        

        #if true random picked a winning line
        if true_random in line_list:
            profit_dict['true_random']+=calculate_profit(wager, true_random)
        else:
            profit_dict['true_random']-=wager

        #open fav line wins
        if(line_list[0] < 0):
            #favorite open bettors win
            profit_dict['favorite_at_open']+=calculate_profit(wager, line_list[0])
            #dog open bettor loses
            profit_dict['underdog_at_open']-=wager
        #open fav line loses
        else:
            profit_dict['favorite_at_open']-=wager
            profit_dict['underdog_at_open']+=calculate_profit(wager, line_list[0])

        #48h fav wins
        if(line_list[1] < 0):
            #favorite 48h bettors win
            profit_dict['favorite_at_48h']+=calculate_profit(wager, line_list[1])
            #dog 48h bettor loses
            profit_dict['underdog_at_48h']-=wager
        #48h fav line loses
        else:
            profit_dict['favorite_at_48h']-=wager
            profit_dict['underdog_at_48h']+=calculate_profit(wager, line_list[1])


        if(line_list[2] < 0):
            #favorite open bettors win
            profit_dict['favorite_at_close']+=calculate_profit(wager, line_list[2])
            #dog open bettor loses
            profit_dict['underdog_at_close']-=wager
        #open fav line loses
        else:
            profit_dict['favorite_at_close']-=wager
            profit_dict['underdog_at_close']+=calculate_profit(wager, line_list[2])

        del model
        tf.keras.backend.clear_session()
        gc.collect()


    for k in profit_dict:
        profit_dict[k] = (profit_dict[k] - bankroll) / bankroll

    return(profit_dict)




simpleRNN = 'simpleRNN.keras'


simpleRNNSim = perform_sim(100, simpleRNN, 20)


print(simpleRNNSim)