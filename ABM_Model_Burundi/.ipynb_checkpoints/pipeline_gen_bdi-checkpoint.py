import pandas as pd
import random
import time
import sys

random.seed(time.time())

all_ids = ['Musongati', 'Rutana', 'Gitanga', 'Mabanda', 'Mpanda', 'Bubanza', 'Rumonge', 'Mwakiro', 'Itaba', 'Kayogoro', 'Bururi', 'Ntega', 'Bisoro', 'Buhinyuza', 'Rugombo', 'Bukinanyan', 'Rusaka', 'Mugongoman', 'Cankuzo', 'Muruta', 'Matana', 'Muhuta', 'Butezi', 'Kiremba', 'Mugamba', 'Buyengero', 'Gisuru', 'Nyabitsind', 'Gashoho', 'Mwumba', 'Bukeye', 'Burambi', 'Matongo', 'Rutovu', 'Mukike', 'Vumbi', 'Muramvya', 'Kigamba', 'Butaganzwa', 'Butihinda', 'Gisagara', 'Kinyinya', 'Giteranyi', 'Ngozi', 'Gasorwe', 'Mpinga-Kay', 'Giheta', 'Mugina', 'Mabayi', 'Gitobe', 'Gisozi', 'Gishubi', 'Vugizo', 'Nyabihanga', 'Ruyigi', 'Kibago', 'Murwi', 'Mbuye', 'Nyabikere', 'Nyanrusang', 'Cendajuru', 'Mishiha', 'Buraza', 'Bwambarang', 'Gihanga', 'Busoni', 'Shombo', 'Kirundo', 'Bugenyuzi', 'Musigati', 'Makamba', 'Buhiga', 'Nyanza-Lac', 'Bweru', 'Bugarama', 'Mutumba', 'Gahombo', 'Nyamurenza', 'Kabarore', 'Isale', 'Ndava', 'Muyinga', 'Giharo', 'Makebuko', 'Marangara', 'Gitega', 'Songa', 'Gitaramuka', 'Bukemba', 'Kabezi', 'Mutimbuzi', 'Buganda', 'Rugazi', 'Gihogazi', 'Ruhororo', 'Bugendana', 'Mutaho', 'Kayanza', 'Bugabira', 'Kanyosha1', 'Muha', 'Nyabiraba', 'Tangara', 'Mubimbi', 'Vyanda', 'Rutegama', 'Gatara', 'Muhanga', 'Rango', 'Kiganda', 'Bukirasazi', 'Gashikanwa', 'Mukaza', 'Mutambu', 'Ntahangwa', 'Busiga', 'Kayokwe', 'Ryansoro']

D = 3.67#random.uniform(1.0,10.0) #1.5/2.0 ## kind of tuned
A = 23.33#random.uniform(50.0,100.0) #70.0 ## calibrated
T = 0.78##random.uniform(0.5, 3.0) #0.8/1.0 ## calibrated
S = 98.67 #98.67 ## from paper

#time_choice = random.sample([[4,1],[7,1],[7,3],[14,1],[14,5],[14,7],[21,1],[21,7],[21,14],[21,18]],1)[0]
t_l = 18#random.sample([10,12,14,18,21],1)[0]
t_r = 7

ps = random.uniform(0.34,0.36)#0.21#random.uniform(0.4,1.0) ##make this a parameter
ews = random.uniform(0.32,0.36)
pactive = 1000#random.randint(15,30)
peer_thresh_hi = random.randint(10,20)
peer_thresh_lo = 2
network_struct = 13
dynamic_network = 1
phase_shift = 1000
use_neighbor = 5#random.sample([2,5,8,10],1)[0]#random.unifrom(0.0,1.0)
border_cross_prob = 0.3#random.uniform(0.25,0.28)
multiply_lo = 1.0#0.8-0.9
multiply_hi = 1.8

hyper_comb = int(sys.argv[1])
comment = str(sys.argv[2])
incomplete = []

MEMORY = False

if MEMORY==True:

    raion_df = pd.read_csv('/home/zm8bh/radiation_model/migration_shock/scripts/analysis_notebooks/memory_req_raion_bdi.csv')

    for _,row in raion_df.iterrows():
        name = row['Raion']
        mem_req = row['mem_need']
        print('sbatch --mem='+str(mem_req)+' bdi_mim_mdm_sample.sbatch',end=' ')
        if name.startswith('Chornobyl'):
            print('"'+name+'"',hyper_comb,D,A,T,S,t_l,t_r,ps,ews,pactive,peer_thresh_lo,peer_thresh_hi,network_struct,
                  dynamic_network,use_neighbor,border_cross_prob,phase_shift,multiply_lo,multiply_hi)
        else:
            print(name,hyper_comb,D,A,T,S,t_l,t_r,ps,ews,pactive,peer_thresh_lo,peer_thresh_hi,network_struct,
                  dynamic_network,use_neighbor,border_cross_prob,phase_shift,multiply_lo,multiply_hi)
else:
    for name in all_ids:
        if name.startswith('Chornobyl'):
            print('sbatch bdi_mim_mdm_sample.sbatch','"'+name+'"',hyper_comb,D,A,T,S,t_l,t_r,ps,ews,pactive,peer_thresh_lo,peer_thresh_hi,network_struct,
                  dynamic_network,use_neighbor,border_cross_prob,phase_shift,multiply_lo,multiply_hi)
        else:
            print('sbatch bdi_mim_mdm_sample.sbatch',name,hyper_comb,D,A,T,S,t_l,t_r,ps,ews,pactive,peer_thresh_lo,peer_thresh_hi,network_struct,
                  dynamic_network,use_neighbor,border_cross_prob,phase_shift,multiply_lo,multiply_hi)

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
                
