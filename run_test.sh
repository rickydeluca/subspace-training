#!/bin/bash

# Define the input values
tests=("subspace" "baseline")
datasets=("mnist" "cifar10")
networks=("fc" "lenet" "resnet20")
shuffle_pixels=(0 1)
shuffle_labels=(0 1)

# FC properties
depths=(1 2 3 4 5)
widths=(50 100 200 400)

# CNNs properties
n_features=()
for f in $(seq 1 1 30); do
    n_features+=($f)
done

# Subspace dimensions
subspace_dims=()
for d in $(seq 0 100 1600); do
    # Append the current value to the array
    subspace_dims+=($d)
done

epochs=(10)
lr=(3e-3)

# Loop through the input values and run main.py
for test in "${tests[@]}"; do
    
    # SUBSPACE TEST
    if [ "$test" = "subspace" ]; then
        
        for dataset in "${datasets[@]}"; do
            for sp in "${shuffle_pixels[@]}"; do
                for sl in "${shuffle_labels[@]}"; do
                    for network in "${networks[@]}"; do

                        # FCN
                        if [ "$network" = "fc" ]; then
                            
                            for depth in "${depths[@]}"; do
                                for width in "${widths[@]}"; do
                                    for sd in "${subspace_dims[@]}"; do
                                        python main.py --dataset "$dataset" --shuffle_pixels "$sp" --shuffle_labels "$sl" --network "$network" --hidden_depth "$depth" --hidden_width "$width" --subspace_dim "$sd" --test "$test"
                                    done
                                done
                            done

                        # LeNet and ResNet20
                        else

                            for nf in "${n_features[@]}"; do
                                for sd in "${subspace_dims[@]}"; do
                                    python main.py --dataset "$dataset" --shuffle_pixels "$sp" --shuffle_labels "$sl" --network "$network" --n_feature "$nf" --subspace_dim "$sd" --test "$test"
                                done
                            done    

                        fi

                    done
                done
            done
        done

    # BASELINE TEST
    else
        for dataset in "${datasets[@]}"; do
            for sp in "${shuffle_pixels[@]}"; do
                for sl in "${shuffle_labels[@]}"; do
                    for network in "${networks[@]}"; do
                        
                        # FCN
                        if [ "$network" = "fc" ]; then
                            
                            for depth in "${depths[@]}"; do
                                for width in "${widths[@]}"; do
                                    python main.py --dataset "$dataset" --shuffle_pixels "$sp" --shuffle_labels "$sl" --network "$network" --hidden_depth "$depth" --hidden_width "$width" --test "$test"
                                done
                            done

                        # LeNet and ResNet20
                        else

                            for nf in "${n_features[@]}"; do
                                python main.py --dataset "$dataset" --shuffle_pixels "$sp" --shuffle_labels "$sl" --network "$network" --n_feature "$nf" --subspace_dim "$sd" --test "$test"
                            done    

                        fi

                    done
                done
            done
        done
    fi

done