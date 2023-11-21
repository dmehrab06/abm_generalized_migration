import pandas as pd
import numpy as np
import sys
import random
import time
import warnings
from file_paths_and_consts import *
import math
import s2sphere
import resource
import datetime
import multiprocessing as mp
import gc
##################################3

def haversine(lon1, lat1, lon2, lat2):
    KM = 6372.8 #Radius of earth in km instead of miles
    lat1, lon1, lat2, lon2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    total_km = KM * c
    return total_km

## rule about how different types of demographic group decides to move
def get_move_prob(DEMO_NUMS): ##this function can be played with
    tot_size = 0
    for v in DEMO_NUMS:
        tot_size = tot_size+v
    move_prob = 0.0
    for i in range(0,len(DEMO_NUMS)):
        move_prob = move_prob + DEMO_NUMS[i]*MOVE_PROB[i]
    return move_prob/tot_size

def bernoulli(val,p):
    if (val<=p):
        return 1
    else:
        return 0
    
def bernoulli_border(val,moves,current_phase,multiply_lo=0.8,multiply_hi=1.5):
    if current_phase == 0:
        multiply = multiply_lo
    elif current_phase==1:
        multiply = multiply_hi
    else:
        multiply = multiply_hi
    if moves==0:
        return 0
    else:
        if val<=BORDER_CROSS_PROB*multiply:
            return 2
        else:
            return 1

def get_omega_k(cur_time,SHI_DATE,SHI):
    if cur_time>=SHI_DATE:
        return SHI
    else:
        return 1.0

    
def safety_remaining(dis_expo,omega_k,C):
    return 1.0 - C*omega_k*(float(np.exp(dis_expo)))
    
def calc_attitude_db(cur_impact_data,cur_household_data,SHI,SHI_DATE,C,SIGMA):
    cur_impact_data = cur_impact_data.rename(columns={'latitude':'impact_lat','longitude':'impact_lng'})
    cur_impact_data['OMEGA_K'] = cur_impact_data['time'].apply(lambda x: get_omega_k(x,SHI_DATE,SHI))
    
    impact_in_homes = cur_impact_data.merge(cur_household_data,on='matching_place_id',how='inner')
    impact_in_homes['dis_conflict_home'] = haversine(impact_in_homes['h_lng'],impact_in_homes['h_lat'],impact_in_homes['impact_lng'],impact_in_homes['impact_lat'])
    impact_in_homes['dis_conflict_home_exponent'] =(impact_in_homes['dis_conflict_home']**2.0)/(-2.0*SIGMA*SIGMA)
    impact_in_homes['safety_felt'] = impact_in_homes.apply(lambda x: safety_remaining(x['dis_conflict_home_exponent'],x['OMEGA_K'],C),axis=1)
    
    home_conflict_df = impact_in_homes.groupby(['hid'])['safety_felt'].prod().reset_index()
    home_conflict_df['P(violence)'] = 1 - home_conflict_df['safety_felt']
    home_conflict_df = home_conflict_df.merge(cur_household_data,on='hid',how='inner')
    return home_conflict_df

def calc_attitude_parallel_db(args):
    cur_household_data,cur_impact_data,SHI,SHI_DATE,C,SIGMA = args
    
    cur_impact_data = cur_impact_data.rename(columns={'latitude':'impact_lat','longitude':'impact_lng'})
    cur_impact_data['OMEGA_K'] = cur_impact_data['time'].apply(lambda x: get_omega_k(x,SHI_DATE,SHI))
    
    impact_in_homes = cur_impact_data.merge(cur_household_data,on='matching_place_id',how='inner')
    impact_in_homes['dis_conflict_home'] = haversine(impact_in_homes['h_lng'],impact_in_homes['h_lat'],impact_in_homes['impact_lng'],impact_in_homes['impact_lat'])
    impact_in_homes['dis_conflict_home_exponent'] =(impact_in_homes['dis_conflict_home']**2.0)/(-2.0*SIGMA*SIGMA)
    impact_in_homes['safety_felt'] = impact_in_homes.apply(lambda x: safety_remaining(x['dis_conflict_home_exponent'],x['OMEGA_K'],C),axis=1)
    
    home_conflict_df = impact_in_homes.groupby(['hid'])['safety_felt'].prod().reset_index()
    home_conflict_df['P(violence)'] = 1 - home_conflict_df['safety_felt']
    home_conflict_df = home_conflict_df.merge(cur_household_data,on='hid',how='inner')
    return home_conflict_df

def multiproc_attitude(cur_household_data, impact_data, lookahead_date_1, lookahead_date_2, SHI, SHI_DATE, C, SIGMA):
    cpus = USE_CORE#mp.cpu_count()
    hh_splits = np.array_split(cur_household_data, cpus) #--this a list with multiple dataframe.. each dataframe is used by one core
    
    cur_impact_data = impact_data[(impact_data.time>=lookahead_date_1) & (impact_data.time<=lookahead_date_2)]
    
    pool_args = [(h_chunk,cur_impact_data, SHI, SHI_DATE, C, SIGMA) for h_idx,h_chunk in enumerate(hh_splits)]
    #print('total time taken to split',time.time()-st_time)
    pool = mp.Pool(processes = cpus)
    results = pool.map(calc_attitude_parallel_db, pool_args)
    pool.close()
    pool.join()
    return results
    
if __name__ == "__main__":
    print('starting',flush=True)
    #mp.set_start_method('forkserver')
    warnings.filterwarnings('ignore')

    random.seed(time.time())

    start_time = time.time()
    print(datetime.datetime.now(),flush=True)

    #############################2
    APPLY_PEER = 1
    EPS = 0.0001 #0.0001 ## calibrated
    CONFLICT_DATA_PREFIX = 'ukraine_conflict_data_ADM2_HDX_buffer_'
    HOUSEHOLD_DATA_PREFIX = 'ukraine_household_data_ADM2_HDX.csv'
    START_DATE = '2022-02-24'
    END_DATE = '2022-10-15'

    PLACE_NAME = str(sys.argv[1])
    hyper_comb = int(sys.argv[2])
    KERNEL_DISPERSION_PARAMETER = float(sys.argv[3]) ##SIGMA
    CONFLICT_WINDOW = int(sys.argv[4])
    KERNEL_SCALING_PARAMETER = float(random.uniform(0.5,0.7)) ##C
    T_S = pd.to_datetime('2022-03-23') ##SHI_DATE
    BORDER_CROSS_PROB = float(np.random.normal(0.32,0.02,1)[0])
    CONFLICT_SATURATION = float(np.random.normal(0.7,0.3,1)[0]) ##SHI
    USE_CORE = int(sys.argv[5])
    
    print(PLACE_NAME,flush=True)
    
    multiply_lo = 1
    multiply_hi = 1.8
    multiply_very_hi = 2.0
    
    MOVE_PROB = [1,1,0,1]
    PHASE_SHIFT = 10000
    
    total_impact_data = pd.read_csv(IMPACT_DIR+CONFLICT_DATA_PREFIX+str(5)+'_km.csv')
    total_household_data = pd.read_csv(HOUSEHOLD_DIR+HOUSEHOLD_DATA_PREFIX)
    
    impact_data = total_impact_data[total_impact_data.matching_place_id==PLACE_NAME]
    impact_data['time'] = pd.to_datetime(impact_data['time'])
    impact_data['event_weight'] = 1

    cur_household_data = total_household_data[total_household_data.matching_place_id==PLACE_NAME]    
    print('data loaded until garbage collector',flush=True)
    
    cur_household_data['hh_size'] = cur_household_data[DEMO_TYPES].sum(axis = 1, skipna = True)
    cur_household_data['P(move|violence)'] = cur_household_data.apply(lambda x: get_move_prob([x['OLD_PERSON'],x['CHILD'],x['ADULT_MALE'],x['ADULT_FEMALE']]),axis=1)
    cur_household_data['prob_conflict'] = 0
    cur_household_data['moves'] = 0
    cur_household_data['move_type'] = 0 # 0 means did not move, 1 means IDP, 2 means outside

    if 'h_lat' not in cur_household_data.columns.tolist():
        cur_household_data = cur_household_data.rename(columns={'latitude':'h_lat','longitude':'h_lng'})
    temp_prefix = ''
    
    f = 0
    start = time.time()
    cur_checkpoint = 1000
    #print('combination_no',hyper_comb)

    prev_temp_checkpoint = 0
    last_saved_checkpoint = -1

    peer_used = 0

    DEL_COLUMNS = ['P(violence)','P(move)','random']
    min_date = pd.to_datetime(START_DATE)
    end_date = pd.to_datetime(END_DATE)
    simulated_refugee_df = pd.DataFrame(columns=['id','time','refugee','old_people','child','male','female'])
    simulated_leaving_df = pd.DataFrame(columns=['id','time','leaving','old_people','child','male','female'])
    timing_log_df = pd.DataFrame(columns=['step','remaining_household_agent','remaining_person_agent','conflict_events_now','network_nodes',
                                          'network_edges','attitude_time','pcb_time','subjective_norm_time','pre_time','post_time'])
    who_went_where = []
    hid_displacement_df = []
    print('simulation_starting')
    ATT_FLAG = 1
    PBC_FLAG = 1
    SN_FLAG = 0 #0-0 1-1 2-0 3-1
    #########################################5
    for i in range(0,100):

        print(min_date,flush=True)
        preprocess_start = time.time()
        prev_temp_checkpoint = prev_temp_checkpoint + 1

        max_date = min_date + pd.DateOffset(days=1)
        lookahead_date_1 = min_date - pd.DateOffset(days=CONFLICT_WINDOW)
        lookahead_date_2 = min_date - pd.DateOffset(days=0)

        if(f==1 and min_date > end_date):
            break

        if(f!=0):
            cur_household_data = pd.read_csv(TEMPORARY_DIR+'last_saved_household_data_'+str(temp_prefix)+str(PLACE_NAME)+'_'+str(hyper_comb)+'.csv')
            cur_household_data = cur_household_data[cur_household_data.moves==0]

        if(cur_household_data.shape[0]==0):
            new_row = {'id':PLACE_NAME,'time':min_date,'refugee':0,'old_people':0,'child':0,'male':0,'female':0}
            min_date = max_date
            simulated_refugee_df = simulated_refugee_df.append(new_row,ignore_index=True)
            continue

        cur_impact_data = impact_data[(impact_data.time>=lookahead_date_1) & (impact_data.time<=lookahead_date_2)]

        if(cur_impact_data.shape[0]==0):
            new_row = {'id':PLACE_NAME,'time':min_date,'refugee':0,'old_people':0,'child':0,'male':0,'female':0}
            min_date = max_date
            simulated_refugee_df = simulated_refugee_df.append(new_row,ignore_index=True)
            continue
        preprocess_end = time.time()
        ############# Social theory and main agent decision start ###########
        ### attitude
        print(preprocess_end-preprocess_start,'time to preprocess',flush=True)
        
        attitude_start = time.time()
        if USE_CORE==1:
            home_conflict_df = calc_attitude_db(cur_impact_data,cur_household_data,CONFLICT_SATURATION,T_S,KERNEL_SCALING_PARAMETER,KERNEL_DISPERSION_PARAMETER)
        else:
            home_conflict_df = multiproc_attitude(cur_household_data, impact_data, lookahead_date_1, lookahead_date_2, CONFLICT_SATURATION, T_S, KERNEL_SCALING_PARAMETER, KERNEL_DISPERSION_PARAMETER)
            home_conflict_df = pd.concat(home_conflict_df)
        attitude_end = time.time()
        print(attitude_end-attitude_start,'time to attitue',flush=True)
        ## attitude

        ##pcb
        pcb_start = time.time()
        if PBC_FLAG==1:
            home_conflict_df['P(move)'] = home_conflict_df['P(violence)']*home_conflict_df['P(move|violence)']
        else:
            home_conflict_df['P(move)'] = home_conflict_df['P(violence)']
        home_conflict_df['random'] = np.random.random(home_conflict_df.shape[0])
        home_conflict_df['moves'] = home_conflict_df.apply(lambda x: bernoulli(x['random'],x['P(move)']),axis=1)
        home_conflict_df = home_conflict_df.drop(columns=DEL_COLUMNS)
        pcb_end = time.time()
        print(pcb_end-pcb_start,'time to pbc',flush=True)
        ##pcb

        subjective_norm_start = time.time()
        temp_households = home_conflict_df
        nodes = temp_households.shape[0]
        phase = 0 if (peer_used < PHASE_SHIFT) else 1

        if APPLY_PEER==1 and SN_FLAG==1:
            if USE_CORE==1:
                temp_households = refine_through_peer_effect(temp_households,phase)
            else:
                #temp_households = refine_through_peer_effect(temp_households,phase)
                print('peer effect household size',temp_households.shape[0],flush=True)
                temp_households = multiproc_peer_effect(temp_households)
                temp_households = pd.concat(temp_households)
            peer_used = peer_used + 1

        temp_households['move_type_random'] = np.random.random(temp_households.shape[0])
        temp_households['move_type'] = temp_households.apply(lambda x: bernoulli_border(x['move_type_random'],x['moves'],phase,multiply_lo,multiply_hi),axis=1)
        temp_households = temp_households.drop(columns=['move_type_random'])
        subjective_norm_end = time.time()
        print(subjective_norm_end-subjective_norm_start,'time to peer effect',flush=True)
        ############# Social theory and main agent decision end ########### 
        post_process_and_save_start = time.time()

        new_row = {'id':PLACE_NAME,'time':min_date,'refugee':temp_households[temp_households.move_type==2]['hh_size'].sum(),
                   'old_people':temp_households[temp_households.move_type==2]['OLD_PERSON'].sum(),
                   'child':temp_households[temp_households.move_type==2]['CHILD'].sum(),
                   'male':temp_households[temp_households.move_type==2]['ADULT_MALE'].sum(),
                   'female':temp_households[temp_households.move_type==2]['ADULT_FEMALE'].sum()}
        simulated_refugee_df = simulated_refugee_df.append(new_row,ignore_index=True)

        curtime = time.time()
        if((curtime-start)>=cur_checkpoint):
            #print('checkpoint for ',str(PLACE_NAME))
            simulated_refugee_df.to_csv(OUTPUT_DIR+'mim_result_'+str(PLACE_NAME)+'_'+str(hyper_comb).zfill(5)+'.csv',index=False)
            start = curtime

        temp_households.to_csv(TEMPORARY_DIR+'last_saved_household_data_'+str(temp_prefix)+str(PLACE_NAME)+'_'+str(hyper_comb)+'.csv',index=False)

        last_saved_checkpoint = prev_temp_checkpoint
        min_date = max_date
        f = 1

        post_process_and_save_end = time.time()
        print(post_process_and_save_end-post_process_and_save_start,'time to post process',flush=True)



    ##################################6
    
    simulated_refugee_df.to_csv(OUTPUT_DIR+'mim_result_completed_'+str(PLACE_NAME)+'_'+str(hyper_comb).zfill(5)+'.csv',index=False)

    end = time.time()
    
    print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,flush=True)
    print(datetime.datetime.now(),flush=True)
