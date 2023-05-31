#!/bin/bash
init_power=1
init_clients=10
init_advs=0
prev_power=$init_power
prev_clients=$init_clients
prev_advs=$init_advs

sed -i "s/adv_number = $prev_advs/adv_number = $init_advs/g" 02_mnist_bacis_FL_net_and_attack.py
sed -i "s/adv_pow = $prev_power/adv_pow = $init_power/g" 02_mnist_bacis_FL_net_and_attack.py
sed -i "s/N_CLIENTS  = $prev_clients/N_CLIENTS  = $init_clients/g" 02_mnist_bacis_FL_net_and_attack.py


for power in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30
do
    for advs in 0 1 2 3 4 5 6 7 8
    do
        for clients in 10
        do
            sed -i "s/adv_number = $prev_advs/adv_number = $advs/g" 02_mnist_bacis_FL_net_and_attack.py
            sed -i "s/adv_pow = $prev_power/adv_pow = $power/g" 02_mnist_bacis_FL_net_and_attack.py
            sed -i "s/N_CLIENTS  = $prev_clients/N_CLIENTS  = $clients/g" 02_mnist_bacis_FL_net_and_attack.py

            python3 02_mnist_bacis_FL_net_and_attack.py
            wait
            prev_clients=$clients
        done
        prev_advs=$advs
    done
    prev_power=$power
done


sed -i "s/adv_number = $prev_advs/adv_number = $init_advs/g" 02_mnist_bacis_FL_net_and_attack.py
sed -i "s/adv_pow = $prev_power/adv_pow = $init_power/g" 02_mnist_bacis_FL_net_and_attack.py
sed -i "s/N_CLIENTS  = $prev_clients/N_CLIENTS  = $init_clients/g" 02_mnist_bacis_FL_net_and_attack.py
