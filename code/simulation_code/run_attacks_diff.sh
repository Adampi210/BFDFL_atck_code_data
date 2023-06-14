#!/bin/bash
init_power=1
init_clients=10
init_adv_perc=1
init_seed=0
init_iid_style='iid'
prev_power=$init_power
prev_clients=$init_clients
prev_adv_perc=$init_adv_perc
prev_seed=$init_seed
prev_iid_style=$init_iid_style

sed -i "s/seed = $prev_seed/seed = $init_seed/g" full_centralized_all.py
sed -i "s/N_CLIENTS  = $prev_clients/N_CLIENTS  = $init_clients/g" full_centralized_all.py
sed -i "s/adv_percent = 0.$prev_adv_perc/adv_percent = 0.$init_adv_perc/g" full_centralized_all.py
sed -i "s/adv_pow = $prev_power/adv_pow = $init_power/g" full_centralized_all.py
sed -i "s/iid_type = '$prev_iid_style'/iid_type = '$init_iid_style'/g" full_centralized_all.py

for iid_style in 'iid' 'non_iid'
do
    for seed in 0
    do 
        for power in 50 70 100
        do
            for adv_prec in 1 3 5
            do
                for clients in 10 30 50
                do
                    sed -i "s/seed = $prev_seed/seed = $seed/g" full_centralized_all.py
                    sed -i "s/iid_type = '$prev_iid_style'/iid_type = '$iid_style'/g" full_centralized_all.py
                    sed -i "s/N_CLIENTS  = $prev_clients/N_CLIENTS  = $clients/g" full_centralized_all.py
                    sed -i "s/adv_percent = 0.$prev_adv_perc/adv_percent = 0.$adv_prec/g" full_centralized_all.py
                    sed -i "s/adv_pow = $prev_power/adv_pow = $power/g" full_centralized_all.py

                    python3 full_centralized_all.py
                    wait
                    prev_clients=$clients
                done
                prev_adv_perc=$adv_prec
            done
            prev_power=$power
        done
        prev_seed=$seed
    done
    prev_iid_style=$iid_style
done
sed -i "s/seed = $prev_seed/seed = $init_seed/g" full_centralized_all.py
sed -i "s/N_CLIENTS  = $prev_clients/N_CLIENTS  = $init_clients/g" full_centralized_all.py
sed -i "s/adv_percent = 0.$prev_adv_perc/adv_percent = 0.$init_adv_perc/g" full_centralized_all.py
sed -i "s/adv_pow = $prev_power/adv_pow = $init_power/g" full_centralized_all.py
sed -i "s/iid_type = '$prev_iid_style'/iid_type = '$init_iid_style'/g" full_centralized_all.py
