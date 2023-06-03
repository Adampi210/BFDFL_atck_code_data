#!/bin/bash
init_power=1
init_clients=10
init_advs=0
init_seed=0
prev_power=$init_power
prev_clients=$init_clients
prev_advs=$init_advs
prev_seed=$init_seed

sed -i "s/adv_number = $prev_advs/adv_number = $init_advs/g" full_centralized_all.py
sed -i "s/adv_pow = $prev_power/adv_pow = $init_power/g" full_centralized_all.py
sed -i "s/N_CLIENTS  = $prev_clients/N_CLIENTS  = $init_clients/g" full_centralized_all.py
sed -i "s/seed = $prev_seed/seed = $init_seed/g" full_centralized_all.py

for seed in 0 1 2
do 
    for power in 0
    do
        for advs in 0
        do
            for clients in 10 15 20 25 30 35 40 45 50
            do
                sed -i "s/adv_number = $prev_advs/adv_number = $advs/g" full_centralized_all.py
                sed -i "s/adv_pow = $prev_power/adv_pow = $power/g" full_centralized_all.py
                sed -i "s/N_CLIENTS  = $prev_clients/N_CLIENTS  = $clients/g" full_centralized_all.py
                sed -i "s/seed = $prev_seed/seed = $seed/g" full_centralized_all.py

                python3 full_centralized_all.py
                wait
                prev_clients=$clients
            done
            prev_advs=$advs
        done
        prev_power=$power
    done
    prev_seed=$seed
done

sed -i "s/adv_number = $prev_advs/adv_number = $init_advs/g" full_centralized_all.py
sed -i "s/adv_pow = $prev_power/adv_pow = $init_power/g" full_centralized_all.py
sed -i "s/N_CLIENTS  = $prev_clients/N_CLIENTS  = $init_clients/g" full_centralized_all.py
sed -i "s/seed = $prev_seed/seed = $init_seed/g" full_centralized_all.py
