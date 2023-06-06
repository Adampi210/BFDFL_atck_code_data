#!/bin/bash
init_power=1
init_clients=10
init_adv_perc=1
init_seed=0
prev_power=$init_power
prev_clients=$init_clients
prev_adv_perc=$init_adv_perc
prev_seed=$init_seed

sed -i "s/seed = $prev_seed/seed = $init_seed/g" full_centralized_all.py
sed -i "s/N_CLIENTS  = $prev_clients/N_CLIENTS  = $init_clients/g" full_centralized_all.py
sed -i "s/adv_percent = 0.$prev_adv_perc/adv_percent = 0.$init_adv_perc/g" full_centralized_all.py
sed -i "s/adv_pow = $prev_power/adv_pow = $init_power/g" full_centralized_all.py

for seed in 0
do 
    for power in 0
    do
        for adv_prec in 0
        do
            for clients in 10 20 30 40 50
            do
                sed -i "s/seed = $prev_seed/seed = $seed/g" full_centralized_all.py
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

sed -i "s/seed = $prev_seed/seed = $init_seed/g" full_centralized_all.py
sed -i "s/N_CLIENTS  = $prev_clients/N_CLIENTS  = $init_clients/g" full_centralized_all.py
sed -i "s/adv_percent = 0.$prev_adv_perc/adv_percent = 0.$init_adv_perc/g" full_centralized_all.py
sed -i "s/adv_pow = $prev_power/adv_pow = $init_power/g" full_centralized_all.py
