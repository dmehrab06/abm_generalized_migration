import pandas as pd
import random
import time
import sys

random.seed(time.time())

import json

samples = []
with open('../ukraine_next_step_calibration_peer_effect_int/logs/bayes_examples_all_log_revised.json', 'r') as file:
    for line in file:
        samples.append(json.loads(line))

good = []
max_err = -40000

param_dicts = []
good_hyper_combs = []
for i in range(0,len(samples)):
    tt = samples[i]['target']
    if tt>max_err:
        #print()
        param_dicts.append(samples[i]['params'])
        good_hyper_combs.append(6000+i)
print(len(param_dicts))
#print(good_hyper_combs)

hyper_comb_idx = int(sys.argv[1])
comment = str(sys.argv[2])
incomplete = []
#print('aschi')
param_dict = param_dicts[hyper_comb_idx]
#print(param_dict)
hyper_comb = good_hyper_combs[hyper_comb_idx]

D = param_dict['D']#3.817987416766939
A = param_dict['A']#52.31338027281235#random.uniform(1.0,10.0) #1.5/2.0 ## kind of tuned
T = param_dict['T']#1.2793500570167806##random.uniform(0.5, 3.0) #0.8/1.0 ## calibrated
S = param_dict['S']#98.67 #98.67 ## from paper

t_l = int(param_dict['t_r'])#random.sample([10,12,14,18,21],1)[0]
t_r = t_l

ps = param_dict['ps']#0.8000680516154739#0.21#random.uniform(0.4,1.0) ##make this a parameter
ews = param_dict['ews']#0.9974003434192641
pactive = 40#random.randint(15,30)
peer_thresh_hi = 15#int(param_dict['p_hi'])
peer_thresh_lo = 2#int(param_dict['p_lo'])
network_struct = 13
dynamic_network = 1
phase_shift = 40
use_neighbor = 5#random.sample([2,5,8,10],1)[0]#random.unifrom(0.0,1.0)
border_cross_prob = param_dict['b_prob']#random.uniform(0.25,0.28)
multiply_lo = 1.0#0.8-0.9
multiply_hi = 1.8

trap_start = '2022-01-01'
trap_end = '2022-01-01'

# D = random.uniform(6.0,7.0)#random.uniform(1.0,10.0) #1.5/2.0 ## kind of tuned
# A = random.uniform(52.0,55.0)#random.uniform(50.0,100.0) #70.0 ## calibrated
# T = random.uniform(1.9,2.0)##random.uniform(0.5, 3.0) #0.8/1.0 ## calibrated
# S = 98.67 #98.67 ## from paper

# t_l = 7#random.sample([10,12,14,18,21],1)[0]
# t_r = t_l

# ps = random.uniform(0.5,0.7)#0.21#random.uniform(0.4,1.0) ##make this a parameter
# ews = random.uniform(0.65,0.75)
# pactive = 40#random.randint(15,30)
# peer_thresh_hi = random.randint(10,15)
# peer_thresh_lo = random.randint(1,2)
# network_struct = 13
# dynamic_network = 1
# phase_shift = 40
# use_neighbor = 5#random.sample([2,5,8,10],1)[0]#random.unifrom(0.0,1.0)
# border_cross_prob = 0.3#random.uniform(0.25,0.28)
# multiply_lo = random.uniform(0.28,0.32)#0.8-0.9
# multiply_hi = 1.8




#first: order the raions
def get_memory(hh):
    if hh<=100000:
        return 16000
    if hh<=300000:
        return 64000
    return 256000

def get_core(hh):
    if hh<=100000:
        return 1
    if hh<=300000:
        return 4
    return 8


necessary_mem = pd.read_csv('/home/zm8bh/radiation_model/migration_shock/scripts/analysis_notebooks/memory_req_raion.csv')
raion_df = pd.read_csv('hh_cnts.csv')

raion_df = raion_df.sort_values(by='hh',ascending=False)
raion_df['memory'] = raion_df['hh'].apply(lambda x: get_memory(x))
raion_df['core'] = raion_df['hh'].apply(lambda x: get_core(x))
raion_df = raion_df.merge(necessary_mem[['Raion','mem_need']],left_on='raion',right_on='Raion',how='inner')
raion_df['memory'] = raion_df[["memory", "mem_need"]].max(axis=1)
raion_df = raion_df.sort_values(by='hh',ascending=False)

raion = raion_df['raion'].tolist()
mem = raion_df['memory'].tolist()
cc = raion_df['core'].tolist()

partition = int(sys.argv[3])

set1_raion = raion[0:partition]
set1_mem = mem[0:partition]
set1_cc = cc[0:partition]

set2_raion = raion[partition:]
set2_mem = mem[partition:]
set2_cc = cc[partition:]
set2_raion.reverse()
set2_mem.reverse()
set2_cc.reverse()

entrapped_raions = pd.read_csv('entrapped.csv')

for i in range(len(set1_raion)):
    name = set1_raion[i]
    mem_req = set1_mem[i]
    core_use = set1_cc[i]
    if name in entrapped_raions['Raion'].tolist():
        #trap_start = entrapped_raions[entrapped_raions['Raion']==name].iloc[0]['date_start']
        #trap_end = entrapped_raions[entrapped_raions['Raion']==name].iloc[0]['date_end']
        print('sbatch --mem='+str(mem_req)+' --cpus-per-task='+str(core_use)+' abm.sbatch',end=' ')
        if name.startswith('Chornobyl'):
            print('"'+name+'"',hyper_comb,D,A,T,S,t_l,t_r,ps,ews,pactive,peer_thresh_lo,peer_thresh_hi,
                  network_struct,dynamic_network,use_neighbor,border_cross_prob,phase_shift,multiply_lo,multiply_hi,core_use,trap_start,trap_end)
        else:
            print(name,hyper_comb,D,A,T,S,t_l,t_r,ps,ews,pactive,peer_thresh_lo,peer_thresh_hi,
                  network_struct,dynamic_network,use_neighbor,border_cross_prob,phase_shift,multiply_lo,multiply_hi,core_use,trap_start,trap_end)


for i in range(len(set2_raion)):
    name = set2_raion[i]
    mem_req = set2_mem[i]
    core_use = set2_cc[i]
    if name in entrapped_raions['Raion'].tolist():
        #trap_start = entrapped_raions[entrapped_raions['Raion']==name].iloc[0]['date_start']
        #trap_end = entrapped_raions[entrapped_raions['Raion']==name].iloc[0]['date_end']
        print('sbatch --mem='+str(mem_req)+' --cpus-per-task='+str(core_use)+' abm.sbatch',end=' ')
        if name.startswith('Chornobyl'):
            print('"'+name+'"',hyper_comb,D,A,T,S,t_l,t_r,ps,ews,pactive,peer_thresh_lo,peer_thresh_hi,
                  network_struct,dynamic_network,use_neighbor,border_cross_prob,phase_shift,multiply_lo,multiply_hi,core_use,trap_start,trap_end)
        else:
            print(name,hyper_comb,D,A,T,S,t_l,t_r,ps,ews,pactive,peer_thresh_lo,peer_thresh_hi,
                  network_struct,dynamic_network,use_neighbor,border_cross_prob,phase_shift,multiply_lo,multiply_hi,core_use,trap_start,trap_end)

df = pd.DataFrame(columns=['hyper_comb','DISTANCE_DECAY','SIGMOID_SCALAR','SIGMOID_EXPONENT','MEMORY_DECAY',
                           'TIME_LEFT','TIME_RIGHT','BIAS_SCALE','EVENT_WEIGHT_SCALE','PEER_AFFECT_ACTIVE_DAY','THRESH_HI','USE_NEIGHBOR','BORDER_CROSS_PROB','THRESH_LO','NETWORK_STRUCTURE','COMMENT',
                          'FIRST_PHASE_BORDER_CROSS_SCALE','SECOND_PHASE_BORDER_CROSS_SCALE'])

new_row = {'hyper_comb':hyper_comb,'DISTANCE_DECAY':D,'SIGMOID_SCALAR':A,'SIGMOID_EXPONENT':T,
           'MEMORY_DECAY':S,'TIME_LEFT':t_l,'TIME_RIGHT':t_r,
           'BIAS_SCALE':ps,'EVENT_WEIGHT_SCALE':ews,'PEER_AFFECT_ACTIVE_DAY':pactive,'THRESH_HI':peer_thresh_hi,'THRESH_LO':peer_thresh_lo,'NETWORK_STRUCTURE':network_struct,
           'USE_NEIGHBOR':use_neighbor,'BORDER_CROSS_PROB':border_cross_prob,
           'FIRST_PHASE_BORDER_CROSS_SCALE':multiply_lo,'SECOND_PHASE_BORDER_CROSS_SCALE':multiply_hi,
          'COMMENT':comment}

df = df.append(new_row,ignore_index=True)

df.to_csv('../runtime_log/parameter_comb_comp_paper.csv', mode='a', index=False, header=False)
                
