

all_ids = ['Musongati', 'Rutana', 'Gitanga', 'Mabanda', 'Mpanda', 'Bubanza', 'Rumonge', 'Mwakiro', 'Itaba', 'Kayogoro', 'Bururi', 'Ntega', 'Bisoro', 'Buhinyuza', 'Rugombo', 'Bukinanyan', 'Rusaka', 'Mugongoman', 'Cankuzo', 'Muruta', 'Matana', 'Muhuta', 'Butezi', 'Kiremba', 'Mugamba', 'Buyengero', 'Gisuru', 'Nyabitsind', 'Gashoho', 'Mwumba', 'Bukeye', 'Burambi', 'Matongo', 'Rutovu', 'Mukike', 'Vumbi', 'Muramvya', 'Kigamba', 'Butaganzwa', 'Butihinda', 'Gisagara', 'Kinyinya', 'Giteranyi', 'Ngozi', 'Gasorwe', 'Mpinga-Kay', 'Giheta', 'Mugina', 'Mabayi', 'Gitobe', 'Gisozi', 'Gishubi', 'Vugizo', 'Nyabihanga', 'Ruyigi', 'Kibago', 'Murwi', 'Mbuye', 'Nyabikere', 'Nyanrusang', 'Cendajuru', 'Mishiha', 'Buraza', 'Bwambarang', 'Gihanga', 'Busoni', 'Shombo', 'Kirundo', 'Bugenyuzi', 'Musigati', 'Makamba', 'Buhiga', 'Nyanza-Lac', 'Bweru', 'Bugarama', 'Mutumba', 'Gahombo', 'Nyamurenza', 'Kabarore', 'Isale', 'Ndava', 'Muyinga', 'Giharo', 'Makebuko', 'Marangara', 'Gitega', 'Songa', 'Gitaramuka', 'Bukemba', 'Kabezi', 'Mutimbuzi', 'Buganda', 'Rugazi', 'Gihogazi', 'Ruhororo', 'Bugendana', 'Mutaho', 'Kayanza', 'Bugabira', 'Kanyosha1', 'Muha', 'Nyabiraba', 'Tangara', 'Mubimbi', 'Vyanda', 'Rutegama', 'Gatara', 'Muhanga', 'Rango', 'Kiganda', 'Bukirasazi', 'Gashikanwa', 'Mukaza', 'Mutambu', 'Ntahangwa', 'Busiga', 'Kayokwe', 'Ryansoro']

sub_event_type = ['None','Shelling_or_artillery_or_missile_attack', 'Air_or_drone_strike', 'Peaceful_protest', 'Remote_explosive_or_landmine_or_IED', 'Armed_clash', 'Grenade', 'Abduction_or_forced_disappearance', 'Attack', 'Sexual_violence', 'Excessive_force_against_protesters', 'Looting_or_property_destruction', 'Protest_with_intervention', 'Non-state_actor_overtakes_territory', 'Change_to_group_or_activity', 'Disrupted_weapons_use', 'Agreement', 'Government_regains_territory', 'Non-violent_transfer_of_territory', 'Violent_demonstration', 'Mob_violence', 'Arrests', 'Headquarters_or_base_established']

for name in all_ids:
    if name.startswith('Chornobyl'):
        print('sbatch create_network.sbatch','"'+name+'"')
    else:
        print('sbatch create_network.sbatch',name)
    #print('sbatch ukr_mim_mdm_sample.sbatch',name,hyper_comb+900,D,A,T,S,t_l,t_r,ps,ews,pactive,use_civil_data,'1')
    

