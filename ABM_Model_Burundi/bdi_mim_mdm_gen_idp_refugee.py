all_ids = ['Musongati', 'Rutana', 'Gitanga', 'Mabanda', 'Mpanda', 'Bubanza', 'Rumonge', 'Mwakiro', 'Itaba', 'Kayogoro', 'Bururi', 'Ntega', 'Bisoro', 'Buhinyuza', 'Rugombo', 'Bukinanyan', 'Rusaka', 'Mugongoman', 'Cankuzo', 'Muruta', 'Matana', 'Muhuta', 'Butezi', 'Kiremba', 'Mugamba', 'Buyengero', 'Gisuru', 'Nyabitsind', 'Gashoho', 'Mwumba', 'Bukeye', 'Burambi', 'Matongo', 'Rutovu', 'Mukike', 'Vumbi', 'Muramvya', 'Kigamba', 'Butaganzwa', 'Butihinda', 'Gisagara', 'Kinyinya', 'Giteranyi', 'Ngozi', 'Gasorwe', 'Mpinga-Kay', 'Giheta', 'Mugina', 'Mabayi', 'Gitobe', 'Gisozi', 'Gishubi', 'Vugizo', 'Nyabihanga', 'Ruyigi', 'Kibago', 'Murwi', 'Mbuye', 'Nyabikere', 'Nyanrusang', 'Cendajuru', 'Mishiha', 'Buraza', 'Bwambarang', 'Gihanga', 'Busoni', 'Shombo', 'Kirundo', 'Bugenyuzi', 'Musigati', 'Makamba', 'Buhiga', 'Nyanza-Lac', 'Bweru', 'Bugarama', 'Mutumba', 'Gahombo', 'Nyamurenza', 'Kabarore', 'Isale', 'Ndava', 'Muyinga', 'Giharo', 'Makebuko', 'Marangara', 'Gitega', 'Songa', 'Gitaramuka', 'Bukemba', 'Kabezi', 'Mutimbuzi', 'Buganda', 'Rugazi', 'Gihogazi', 'Ruhororo', 'Bugendana', 'Mutaho', 'Kayanza', 'Bugabira', 'Kanyosha1', 'Muha', 'Nyabiraba', 'Tangara', 'Mubimbi', 'Vyanda', 'Rutegama', 'Gatara', 'Muhanga', 'Rango', 'Kiganda', 'Bukirasazi', 'Gashikanwa', 'Mukaza', 'Mutambu', 'Ntahangwa', 'Busiga', 'Kayokwe', 'Ryansoro']

big_4 = ['Kharkivskyi','Odeskyi','Donetskyi','Kyiv']

sub_event_type = ['None','Shelling_or_artillery_or_missile_attack', 'Air_or_drone_strike', 'Peaceful_protest', 'Remote_explosive_or_landmine_or_IED', 'Armed_clash', 'Grenade', 'Abduction_or_forced_disappearance', 'Attack', 'Sexual_violence', 'Excessive_force_against_protesters', 'Looting_or_property_destruction', 'Protest_with_intervention', 'Non-state_actor_overtakes_territory', 'Change_to_group_or_activity', 'Disrupted_weapons_use', 'Agreement', 'Government_regains_territory', 'Non-violent_transfer_of_territory', 'Violent_demonstration', 'Mob_violence', 'Arrests', 'Headquarters_or_base_established']

import random
import time
import sys

random.seed(time.time())

# D = random.sample([1.0,1.5,2.0,4.0,5.0,8.0,10.0,20.0],1)[0] #1.5/2.0 ## kind of tuned
# A = random.sample([50.0,70.0,80.0,100.0,200.0,500.0],1)[0] #70.0 ## calibrated
# T = random.sample([0.1,0.5,0.8,1.0,1.5,2.0],1)[0] #0.8/1.0 ## calibrated
# S = random.sample([50.0,80.0,98.67,100.0],1)[0] #98.67 ## from paper

# time_choice = random.sample([[4,1],[7,1],[7,3],[14,1],[14,5],[14,7],[21,1],[21,7],[21,14],[21,18]],1)[0]
# t_l = time_choice[0]
# t_r = time_choice[1]

# ps = random.sample([0.2,0.5,0.8,1.0],1)[0] ##make this a parameter
# ews = random.sample([0.01,0.1,0.2,0.5,1,2,5,10],1)[0]
# pactive = random.sample([7,15,30,45,100],1)[0]

D = 2.212#random.uniform(1.0,10.0) #1.5/2.0 ## kind of tuned
A = 55.0#random.uniform(50.0,100.0) #70.0 ## calibrated
T = 0.253##random.uniform(0.5, 3.0) #0.8/1.0 ## calibrated
S = 98.67 #98.67 ## from paper

#time_choice = random.sample([[4,1],[7,1],[7,3],[14,1],[14,5],[14,7],[21,1],[21,7],[21,14],[21,18]],1)[0]
t_l = 18#random.sample([10,12,14,18,21],1)[0]
t_r = 7

ps = random.uniform(0.34,0.36)#0.21#random.uniform(0.4,1.0) ##make this a parameter
ews = random.uniform(0.32,0.36)
pactive = 1000#random.randint(15,30)
peer_thresh_hi = 30#zrandom.randint(0,1)
peer_thresh_lo = 1
network_struct = 13
dynamic_network = 1
phase_shift = 1000
use_neighbor = 5#random.sample([2,5,8,10],1)[0]#random.unifrom(0.0,1.0)
border_cross_prob = 0.25676486858191583#random.uniform(0.25,0.28)

#all_val_err =  [33, 29, 423, 367, 199, 8, 41, 69, 64, 15, 55, 38, 60, 1, 82, 28, 71, 30, 10, 4, 59, 13, 87, 47, 115, 36, 21, 49, 84, 9, 26, 171, 76, 27, 31, 56, 65, 20, 61, 48, 77, 3, 479, 43, 6, 451, 83, 37, 283, 227, 14, 395, 34, 32, 62, 57, 58, 5, 255, 75, 2, 54, 311, 339, 19, 66, 143]



hyper_comb = int(sys.argv[1])
comment = str(sys.argv[2])
incomplete = []

##80000-89999

for name in all_ids:
    if name.startswith('Chornobyl'):
        print('sbatch bdi_mim_mdm_sample.sbatch','"'+name+'"',hyper_comb,D,A,T,S,t_l,t_r,ps,ews,pactive,peer_thresh_lo,peer_thresh_hi,network_struct,dynamic_network,use_neighbor,border_cross_prob,phase_shift)
    else:
        print('sbatch bdi_mim_mdm_sample.sbatch',name,hyper_comb,D,A,T,S,t_l,t_r,ps,ews,pactive,peer_thresh_lo,peer_thresh_hi,network_struct,dynamic_network,use_neighbor,border_cross_prob,phase_shift)
    #print('sbatch ukr_mim_mdm_sample.sbatch',name,hyper_comb+900,D,A,T,S,t_l,t_r,ps,ews,pactive,use_civil_data,'1')
    
import pandas as pd

df = pd.DataFrame(columns=['hyper_comb','DISTANCE_DECAY','SIGMOID_SCALAR','SIGMOID_EXPONENT','MEMORY_DECAY',
                           'TIME_LEFT','TIME_RIGHT','BIAS_SCALE','EVENT_WEIGHT_SCALE','PEER_AFFECT_ACTIVE_DAY','THRESH_HI','USE_NEIGHBOR','BORDER_CROSS_PROB','THRESH_LO','NETWORK_STRUCTURE','COMMENT'])

new_row = {'hyper_comb':hyper_comb,'DISTANCE_DECAY':D,'SIGMOID_SCALAR':A,'SIGMOID_EXPONENT':T,
           'MEMORY_DECAY':S,'TIME_LEFT':t_l,'TIME_RIGHT':t_r,
           'BIAS_SCALE':ps,'EVENT_WEIGHT_SCALE':ews,'PEER_AFFECT_ACTIVE_DAY':pactive,'THRESH_HI':peer_thresh_hi,'THRESH_LO':peer_thresh_lo,'NETWORK_STRUCTURE':network_struct,
           'USE_NEIGHBOR':use_neighbor,'BORDER_CROSS_PROB':border_cross_prob,
          'COMMENT':comment}

df = df.append(new_row,ignore_index=True)

df.to_csv('../runtime_log/parameter_comb_peer_effect.csv', mode='a', index=False, header=False)
                
