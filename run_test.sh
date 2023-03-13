#!/bin/bash

# Define the input values
# tests=("subspace" "baseline" "time")
tests=("time")
datasets=("mnist" "cifar10")
networks=("fc" "lenet" "resnet20")
shuffle_pixels=(0 1)
shuffle_labels=(0 1)


# FC properties
depths=(1 2 3 4 5)
widths=(50 100 200 400)

# CNNs properties
n_features=()
for f in $(seq 1 1 20); do
    n_features+=($f)
done

# Subspace dimensions
subspace_dims=()
for d in $(seq 100 100 1600); do
    # Append the current value to the array
    subspace_dims+=($d)
done

projs=("dense" "sparse" "fastfood")

epochs=(10)
lr=(3e-3)

# Loop through the input values and run main.py
for test in "${tests[@]}"; do
    
    # =================
    #   SUBSPACE TEST
    # =================
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
                                        python main.py --dataset "$dataset" --shuffle_pixels "$sp" --shuffle_labels "$sl" --network "$network" --hidden_depth "$depth" --hidden_width "$width" --proj sparse --subspace_dim "$sd" --test "$test"
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

    # =================
    #   BASELINE TEST
    # =================
    elif [ "$test" = "baseline" ]; then
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

    # ===================
    #   PROJECTION TIME
    # ===================
    else
        for dataset in "${datasets[@]}"; do

            # ===========
            #   FC 100k
            # ===========

            # Define output filenames
            output_file="results/proj_time/time_${dataset}_fc_100k.csv"

            # Write the header
            echo "subspace_dim,dense (s),sparse (s),fastfood (s)" > $output_file

            # Subspace dimensions
            subspace_dims=(1000 5000 10000 50000 100000)

            for sd in "${subspace_dims[@]}"; do
                # Save the runtimes for the three projection methods
                # If the script fails, set the runtime to inf
                dense_output=$(python main.py --dataset "$dataset" --network fc --hidden_depth 2 --hidden_width 100 --proj dense --subspace_dim "$sd" --epochs 1 --test time)
                if [ $? -ne 0 ]; then
                    dense_output="inf"
                fi

                sparse_output=$(python main.py --dataset "$dataset" --network fc --hidden_depth 2 --hidden_width 100 --proj sparse --subspace_dim "$sd" --epochs 1 --test time)
                if [ $? -ne 0 ]; then
                    sparse_output="inf"
                fi
                
                fastfood_output=$(python main.py --dataset "$dataset" --network fc --hidden_depth 2 --hidden_width 100 --proj fastfood --subspace_dim "$sd" --epochs 1 --test time)
                if [ $? -ne 0 ]; then
                    fastfood_output="inf"
                fi

                # Get only the last line of the output (runtime)
                dense_runtime=$(echo "$dense_output" | tail -n 1)
                sparse_runtime=$(echo "$sparse_output" | tail -n 1)
                fastfood_runtime=$(echo "$fastfood_output" | tail -n 1)

                # Write results in the CSV file
                echo "$sd,$dense_runtime,$sparse_runtime,$fastfood_runtime" >> $output_file
            done


            # =========
            #   FC 1M
            # =========
            
            # Define output filenames
            output_file="results/proj_time/time_${dataset}_fc_1m.csv"

            # Write the header
            echo "subspace_dim,dense (s),sparse (s),fastfood (s)" > $output_file

            # Subspace dimensions
            subspace_dims=(1000 5000 10000 50000 100000 500000 1000000)
            
            for sd in "${subspace_dims[@]}"; do
                # Save the runtimes for the three projection methods
                dense_output=$(python main.py --dataset "$dataset" --network fc --hidden_depth 5 --hidden_width 400 --proj dense --subspace_dim "$sd" --epochs 1 --test time)
                if [ $? -ne 0 ]; then
                    dense_output="inf"
                fi
                
                sparse_output=$(python main.py --dataset "$dataset" --network fc --hidden_depth 5 --hidden_width 400 --proj sparse --subspace_dim "$sd" --epochs 1 --test time)
                if [ $? -ne 0 ]; then
                    sparse_output="inf"
                fi
                
                fastfood_output=$(python main.py --dataset "$dataset" --network fc --hidden_depth 5 --hidden_width 400 --proj fastfood --subspace_dim "$sd" --epochs 1 --test time)
                if [ $? -ne 0 ]; then
                    fastfood_output="inf"
                fi

                # Get only the last line of the output (runtime)
                dense_runtime=$(echo "$dense_output" | tail -n 1)
                sparse_runtime=$(echo "$sparse_output" | tail -n 1)
                fastfood_runtime=$(echo "$fastfood_output" | tail -n 1)

                # Write results in the CSV file
                echo "$sd,$dense_runtime,$sparse_runtime,$fastfood_runtime" >> $output_file
            done

            # ==============
            #   LeNet 60k
            # ==============
            
            # Define output filenames
            output_file="results/proj_time/time_${dataset}_lenet_60k.csv"

            # Write the header
            echo "subspace_dim,dense (s),sparse (s),fastfood (s)" > $output_file
            
            # Subspace dimensions
            subspace_dims=(1000 5000 10000 50000 60000)
            
            for sd in "${subspace_dims[@]}"; do
                # Save the runtimes for the three projection methods
                dense_output=$(python main.py --dataset "$dataset" --network lenet --n_feature 6 --proj dense --subspace_dim "$sd" --epochs 1 --test time)
                if [ $? -ne 0 ]; then
                    dense_output="inf"
                fi
                
                sparse_output=$(python main.py --dataset "$dataset" --network lenet --n_feature 6 --proj sparse --subspace_dim "$sd" --epochs 1 --test time)
                if [ $? -ne 0 ]; then
                    sparse_output="inf"
                fi
                
                fastfood_output=$(python main.py --dataset "$dataset" --network lenet --n_feature 6 --proj fastfood --subspace_dim "$sd" --epochs 1 --test time)
                if [ $? -ne 0 ]; then
                    fastfood_output="inf"
                fi

                # Get only the last line of the output (runtime)
                dense_runtime=$(echo "$dense_output" | tail -n 1)
                sparse_runtime=$(echo "$sparse_output" | tail -n 1)
                fastfood_runtime=$(echo "$fastfood_output" | tail -n 1)

                # Write results in the CSV file
                echo "$sd,$dense_runtime,$sparse_runtime,$fastfood_runtime" >> $output_file
            done
            
            # =================
            #   ResNet20 270k
            # =================
            
            # Define output filenames
            output_file="results/proj_time/time_${dataset}_resnet20_270k.csv"

            # Write the header
            echo "subspace_dim,dense (s),sparse (s),fastfood (s)" > $output_file

            # Subspace dimensions
            subspace_dims=(1000 5000 10000 50000 100000 200000 270000)

            for sd in "${subspace_dims[@]}"; do
                # Save the runtimes for the three projection methods
                dense_output=$(python main.py --dataset "$dataset" --network resnet20 --proj dense --subspace_dim "$sd" --epochs 1 --test time)
                if [ $? -ne 0 ]; then
                    dense_output="inf"
                fi
                
                sparse_output=$(python main.py --dataset "$dataset" --network resnet20 --proj sparse --subspace_dim "$sd" --epochs 1 --test time)
                if [ $? -ne 0 ]; then
                    sparse_output="inf"
                fi
                
                fastfood_output=$(python main.py --dataset "$dataset" --network resnet20 --proj fastfood --subspace_dim "$sd" --epochs 1 --test time)
                if [ $? -ne 0 ]; then
                    fastfood_output="inf"
                fi
                
                # Get only the last line of the output (runtime)
                dense_runtime=$(echo "$dense_output" | tail -n 1)
                sparse_runtime=$(echo "$sparse_output" | tail -n 1)
                fastfood_runtime=$(echo "$fastfood_output" | tail -n 1)

                # Write results in the CSV file
                echo "$sd,$dense_runtime,$sparse_runtime,$fastfood_runtime" >> $output_file
            done
       
        done
    fi
done