#!/bin/bash
init_power=0
init_clients=10
init_adv_perc=0
init_seed=0
init_iid_style='iid'
prev_power=$init_power
prev_clients=$init_clients
prev_adv_perc=$init_adv_perc
prev_seed=$init_seed
prev_iid_style=$init_iid_style

sed -i "s/seed = $prev_seed/seed = $init_seed/g" full_decentralized.py
sed -i "s/N_CLIENTS  = $prev_clients/N_CLIENTS  = $init_clients/g" full_decentralized.py
sed -i "s/adv_percent = 0.$prev_adv_perc/adv_percent = 0.$init_adv_perc/g" full_decentralized.py
sed -i "s/adv_pow = $prev_power/adv_pow = $init_power/g" full_decentralized.py
sed -i "s/iid_type = '$prev_iid_style'/iid_type = '$init_iid_style'/g" full_decentralized.py


for iid_style in 'iid' 'non_iid'
do
    for clients in 10 30 50
    do
        for power in 0
        do
            for adv_prec in 0
            do
                for seed in 0 1 2
                do 
                    sed -i "s/seed = $prev_seed/seed = $seed/g" full_decentralized.py
                    sed -i "s/iid_type = '$prev_iid_style'/iid_type = '$iid_style'/g" full_decentralized.py
                    sed -i "s/N_CLIENTS  = $prev_clients/N_CLIENTS  = $clients/g" full_decentralized.py
                    sed -i "s/adv_percent = 0.$prev_adv_perc/adv_percent = 0.$adv_prec/g" full_decentralized.py
                    sed -i "s/adv_pow = $prev_power/adv_pow = $power/g" full_decentralized.py

                    python3 full_decentralized.py
                    wait
                    prev_seed=$seed
                done
                prev_adv_perc=$adv_prec
            done
            prev_power=$power
        done
        prev_clients=$clients
    done
    prev_iid_style=$iid_style
done


sed -i "s/seed = $prev_seed/seed = $init_seed/g" full_decentralized.py
sed -i "s/iid_type = '$prev_iid_style'/iid_type = '$init_iid_style'/g" full_decentralized.py
sed -i "s/N_CLIENTS  = $prev_clients/N_CLIENTS  = $init_clients/g" full_decentralized.py
sed -i "s/adv_percent = 0.$prev_adv_perc/adv_percent = 0.$init_adv_perc/g" full_decentralized.py
sed -i "s/adv_pow = $prev_power/adv_pow = $init_power/g" full_decentralized.py
