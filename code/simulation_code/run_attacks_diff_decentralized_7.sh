#!/bin/bash
init_power=0
init_adv_perc=0
init_seed=0
init_iid_style='iid'
init_cent_used=0
init_designated_clients=10
prev_power=$init_power
prev_adv_perc=$init_adv_perc
prev_seed=$init_seed
prev_iid_style=$init_iid_style
prev_cent_used=$init_cent_used
prev_designated_clients=$init_designated_clients

sed -i "s/seed = $prev_seed/seed = $init_seed/g" full_decentralized_7.py
sed -i "s/adv_percent = 0.$prev_adv_perc/adv_percent = 0.$init_adv_perc/g" full_decentralized_7.py
sed -i "s/adv_pow = $prev_power/adv_pow = $init_power/g" full_decentralized_7.py
sed -i "s/iid_type = '$prev_iid_style'/iid_type = '$init_iid_style'/g" full_decentralized_7.py
sed -i "s/cent_measure_used = $prev_cent_used/cent_measure_used = $init_cent_used/g" full_decentralized_7.py
sed -i "s/designated_clients = $prev_designated_clients/designated_clients = $init_designated_clients/g" full_decentralized_7.py

for designated_clients in 25
do
    for iid_style in 'iid'
    do
        for power in 50 250 500 1000
        do
            for adv_prec in 2
            do
                for seed in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
                do 
                    for cent_used in 5
                    do
                        sed -i "s/seed = $prev_seed/seed = $seed/g" full_decentralized_7.py
                        sed -i "s/iid_type = '$prev_iid_style'/iid_type = '$iid_style'/g" full_decentralized_7.py
                        sed -i "s/adv_percent = 0.$prev_adv_perc/adv_percent = 0.$adv_prec/g" full_decentralized_7.py
                        sed -i "s/adv_pow = $prev_power/adv_pow = $power/g" full_decentralized_7.py
                        sed -i "s/cent_measure_used = $prev_cent_used/cent_measure_used = $cent_used/g" full_decentralized_7.py
                        sed -i "s/designated_clients = $prev_designated_clients/designated_clients = $designated_clients/g" full_decentralized_7.py

                        python3 full_decentralized_7.py
                        wait
                    prev_cent_used=$cent_used
                    done
                    prev_seed=$seed
                done
                prev_adv_perc=$adv_prec
            done
            prev_power=$power
        done
        prev_iid_style=$iid_style
    done
    prev_designated_clients=$designated_clients
done

sed -i "s/seed = $prev_seed/seed = $init_seed/g" full_decentralized_7.py
sed -i "s/iid_type = '$prev_iid_style'/iid_type = '$init_iid_style'/g" full_decentralized_7.py
sed -i "s/adv_percent = 0.$prev_adv_perc/adv_percent = 0.$init_adv_perc/g" full_decentralized_7.py
sed -i "s/adv_pow = $prev_power/adv_pow = $init_power/g" full_decentralized_7.py
sed -i "s/cent_measure_used = $prev_cent_used/cent_measure_used = $init_cent_used/g" full_decentralized_7.py
sed -i "s/designated_clients = $prev_designated_clients/designated_clients = $init_designated_clients/g" full_decentralized_7.py
