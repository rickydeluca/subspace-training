#!/bin/bash

# Define the input values
# tests=("subspace" "baseline" "time" "small-nets")
tests=("subspace")
# datasets=("mnist" "cifar10")
datasets=("mnist")
# networks=("fc" "lenet" "resnet20")
networks=("fc")
# shuffle_pixels=(0 1)
shuffle_pixels=(0)
shuffle_labels=(0)


# FC properties
# depths=(1 2 3 4 5)
# widths=(50 100 200 400)
depths=(2)
widths=(200)

# CNNs properties
n_features=()
for f in $(seq 1 1 20); do
    n_features+=($f)
done

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

                        # Define the subspace dimensions
                        if [ "$dataset" = "mnist" ]; then   # MNIST
                            
                            if [ "$network" = "fc" ]; then  # FC
                                # subspace_dims=(100 200 300 400 500 600 700 800 900 1000)
                                subspace_dims=(2400 3200 4000 5000 5700 6500 7300 8200)
                            elif [ "$network" = "lenet" ]; then  # LeNet
                                if [ "$sp" = "1" ]; then
                                    subspace_dims=(100 200 400 600 800 1000 1200 1400 1600)
                                else
                                    subspace_dims=(100 150 200 250 300 350 400)
                                fi
                            else  # ResNet20
                                if [ "$sp" = "1" ]; then
                                    subspace_dims=(100 200 400 600 800 1000 1200 1400 1600 2000 5000 10000 20000 30000 50000 1000000)
                                else
                                    subspace_dims=(500 1000 2000 3000 5000 10000 20000 30000 50000 100000)
                                fi
                            fi
                            
                        else    # CIFAR-10
                            if [ "$network" = "fc" ]; then
                                subspace_dims=(100 500 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000)
                            elif [ "$network" = "lenet" ]; then  # LeNet
                                if [ "$sp" = "1" ]; then
                                    subspace_dims=(1)
                                else
                                    subspace_dims=(100 500 1000 1500 2000 2500 3000 3500 4000 6000 8000 10000)
                                fi
                            else  # ResNet20
                                if [ "$sp" = "1" ]; then
                                    subspace_dims=(1)
                                else
                                    subspace_dims=(100 500 1000 1500 2000 3500 4000 5000 10000 20000 30000 50000 1000000)
                                fi
                            fi
                        fi

                        # FCN
                        if [ "$network" = "fc" ]; then
                            for depth in "${depths[@]}"; do
                                for width in "${widths[@]}"; do
                                    for sd in "${subspace_dims[@]}"; do
                                        # Define projection type wrt subspace dim
                                        if [ "$sd" -gt 3000 ]; then
                                            proj="fastfood"
                                        else
                                            proj="dense"
                                        fi

                                        python main.py --dataset "$dataset" --shuffle_pixels "$sp" --shuffle_labels "$sl" --network "$network" --hidden_depth "$depth" --hidden_width "$width" --proj "$proj" --subspace_dim "$sd" --test "$test"
                                    done
                                done
                            done

                        # LeNet and ResNet20
                        else
                            for nf in "${n_features[@]}"; do
                                for sd in "${subspace_dims[@]}"; do
                                    # Define projection type wrt subspace dim
                                    if [ "$sd" -gt 3000 ]; then
                                        proj="fastfood"
                                    else
                                        proj="dense"
                                    fi

                                    python main.py --dataset "$dataset" --shuffle_pixels "$sp" --shuffle_labels "$sl" --network "$network" --n_feature "$nf" --subspace_dim "$sd" --proj "$proj" --test "$test"
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
                                python main.py --dataset "$dataset" --shuffle_pixels "$sp" --shuffle_labels "$sl" --network "$network" --n_feature "$nf" --test "$test"
                            done    

                        fi

                    done
                done
            done
        done

    # ===================
    #   SMALL NETS TEST
    # ===================
    elif [ "$test" = "small-nets" ]; then
        for dataset in "${datasets[@]}"; do
            for sp in "${shuffle_pixels[@]}"; do
                for sl in "${shuffle_labels[@]}"; do

                    # FC
                    depths=(2)
                    widths=(6 7 8 9 10 )
                        
                    for depth in "${depths[@]}"; do
                        for width in "${widths[@]}"; do
                            python main.py --dataset "$dataset" --shuffle_pixels "$sp" --shuffle_labels "$sl" --network fc --hidden_depth "$depth" --hidden_width "$width" --test "$test"
                        done
                    done

                done
            done
        done

    # ===================
    #   PROJECTION TIME
    # ===================
    else
        for dataset in "${datasets[@]}"; do
            # Define the maximum runtime in seconds
            MAX_RUNTIME=180

            # ===========
            #   FC 100k
            # ===========

            # Define output filenames
            output_file="results/proj_time/time_${dataset}_fc_100k.csv"

            # Write the header
            echo "subspace_dim,dense (ms),sparse (ms),fastfood (ms),direct (ms)" > $output_file

            # Subspace dimensions
            subspace_dims=(1000 5000 10000 50000 100000)

            for sd in "${subspace_dims[@]}"; do
                # Save the runtimes for the three projection methods
                # If the script fails or runs out of time, set the output to inf
                dense_output=$(timeout ${MAX_RUNTIME} python main.py --dataset "$dataset" --network fc --hidden_depth 2 --hidden_width 100 --proj dense --subspace_dim "$sd" --epochs 1 --test time)
                if [ $? -ne 0 ]; then
                    dense_output="inf"
                fi

                sparse_output=$(timeout ${MAX_RUNTIME} python main.py --dataset "$dataset" --network fc --hidden_depth 2 --hidden_width 100 --proj sparse --subspace_dim "$sd" --epochs 1 --test time)
                if [ $? -ne 0 ]; then
                    sparse_output="inf"
                fi
                
                fastfood_output=$(timeout ${MAX_RUNTIME} python main.py --dataset "$dataset" --network fc --hidden_depth 2 --hidden_width 100 --proj fastfood --subspace_dim "$sd" --epochs 1 --test time)
                if [ $? -ne 0 ]; then
                    fastfood_output="inf"
                fi

                direct_output=$(timeout ${MAX_RUNTIME} python main.py --dataset "$dataset" --network fc --hidden_depth 2 --hidden_width 100 --epochs 1 --test time)
                if [ $? -ne 0 ]; then
                    direct_output="inf"
                fi

                # Get only the last line of the output (runtime)
                dense_runtime=$(echo "$dense_output" | tail -n 1)
                sparse_runtime=$(echo "$sparse_output" | tail -n 1)
                fastfood_runtime=$(echo "$fastfood_output" | tail -n 1)
                direct_runtime=$(echo "$direct_output" | tail -n 1)

                # Write results in the CSV file
                echo "$sd,$dense_runtime,$sparse_runtime,$fastfood_runtime" >> $output_file
            done


            # =========
            #   FC 1M
            # =========
            
            # Define output filenames
            output_file="results/proj_time/time_${dataset}_fc_1m.csv"

            # Write the header
            echo "subspace_dim,dense (ms),sparse (ms),fastfood (ms),direct (ms)" > $output_file

            # Subspace dimensions
            subspace_dims=(1000 5000 10000 50000 100000 500000 1000000)
            
            for sd in "${subspace_dims[@]}"; do
                # Save the runtimes for the three projection methods
                dense_output=$(timeout ${MAX_RUNTIME} python main.py --dataset "$dataset" --network fc --hidden_depth 5 --hidden_width 400 --proj dense --subspace_dim "$sd" --epochs 1 --test time)
                if [ $? -ne 0 ]; then
                    dense_output="inf"
                fi
                
                sparse_output=$(timeout ${MAX_RUNTIME} python main.py --dataset "$dataset" --network fc --hidden_depth 5 --hidden_width 400 --proj sparse --subspace_dim "$sd" --epochs 1 --test time)
                if [ $? -ne 0 ]; then
                    sparse_output="inf"
                fi
                
                fastfood_output=$(timeout ${MAX_RUNTIME} python main.py --dataset "$dataset" --network fc --hidden_depth 5 --hidden_width 400 --proj fastfood --subspace_dim "$sd" --epochs 1 --test time)
                if [ $? -ne 0 ]; then
                    fastfood_output="inf"
                fi

                direct_output=$(timeout ${MAX_RUNTIME} python main.py --dataset "$dataset" --network fc --hidden_depth 5 --hidden_width 400 --epochs 1 --test time)
                if [ $? -ne 0 ]; then
                    direct_output="inf"
                fi

                # Get only the last line of the output (runtime)
                dense_runtime=$(echo "$dense_output" | tail -n 1)
                sparse_runtime=$(echo "$sparse_output" | tail -n 1)
                fastfood_runtime=$(echo "$fastfood_output" | tail -n 1)
                direct_runtime=$(echo "$direct_output" | tail -n 1)

                # Write results in the CSV file
                echo "$sd,$dense_runtime,$sparse_runtime,$fastfood_runtime" >> $output_file
            done
        
            # ==============
            #   LeNet 60k
            # ==============
            
            # Define output filenames
            output_file="results/proj_time/time_${dataset}_lenet_60k.csv"

            # Write the header
            echo "subspace_dim,dense (ms),sparse (ms),fastfood (ms),direct (ms)" > $output_file
            
            # Subspace dimensions
            subspace_dims=(1000 5000 10000 50000 60000)
            
            for sd in "${subspace_dims[@]}"; do
                # Save the runtimes for the three projection methods
                dense_output=$(timeout ${MAX_RUNTIME} python main.py --dataset "$dataset" --network lenet --n_feature 6 --proj dense --subspace_dim "$sd" --epochs 1 --test time)
                if [ $? -ne 0 ]; then
                    dense_output="inf"
                fi
                
                sparse_output=$(timeout ${MAX_RUNTIME} python main.py --dataset "$dataset" --network lenet --n_feature 6 --proj sparse --subspace_dim "$sd" --epochs 1 --test time)
                if [ $? -ne 0 ]; then
                    sparse_output="inf"
                fi
                
                fastfood_output=$(timeout ${MAX_RUNTIME} python main.py --dataset "$dataset" --network lenet --n_feature 6 --proj fastfood --subspace_dim "$sd" --epochs 1 --test time)
                if [ $? -ne 0 ]; then
                    fastfood_output="inf"
                fi

                direct_output=$(timeout ${MAX_RUNTIME} python main.py --dataset "$dataset" --network lenet --n_feature 6 --epochs 1 --test time)
                if [ $? -ne 0 ]; then
                    direct_output="inf"
                fi

                # Get only the last line of the output (runtime)
                dense_runtime=$(echo "$dense_output" | tail -n 1)
                sparse_runtime=$(echo "$sparse_output" | tail -n 1)
                fastfood_runtime=$(echo "$fastfood_output" | tail -n 1)
                direct_runtime=$(echo "$direct_output" | tail -n 1)

                # Write results in the CSV file
                echo "$sd,$dense_runtime,$sparse_runtime,$fastfood_runtime" >> $output_file
            done
            
            # =================
            #   ResNet20 270k
            # =================
            
            # Define output filenames
            output_file="results/proj_time/time_${dataset}_resnet20_270k.csv"

            # Write the header
            echo "subspace_dim,dense (ms),sparse (ms),fastfood (ms),direct (ms)" > $output_file

            # Subspace dimensions
            subspace_dims=(1000 5000 10000 50000 100000 200000 270000)

            for sd in "${subspace_dims[@]}"; do
                # Save the runtimes for the three projection methods
                dense_output=$(timeout ${MAX_RUNTIME} python main.py --dataset "$dataset" --network resnet20 --proj dense --subspace_dim "$sd" --epochs 1 --test time)
                if [ $? -ne 0 ]; then
                    dense_output="inf"
                fi
                
                sparse_output=$(timeout ${MAX_RUNTIME} python main.py --dataset "$dataset" --network resnet20 --proj sparse --subspace_dim "$sd" --epochs 1 --test time)
                if [ $? -ne 0 ]; then
                    sparse_output="inf"
                fi
                
                fastfood_output=$(timeout ${MAX_RUNTIME} python main.py --dataset "$dataset" --network resnet20 --proj fastfood --subspace_dim "$sd" --epochs 1 --test time)
                if [ $? -ne 0 ]; then
                    fastfood_output="inf"
                fi

                direct_output=$(timeout ${MAX_RUNTIME} python main.py --dataset "$dataset" --network resnet20 --epochs 1 --test time)
                if [ $? -ne 0 ]; then
                    direct_output="inf"
                fi
                
                # Get only the last line of the output (runtime)
                dense_runtime=$(echo "$dense_output" | tail -n 1)
                sparse_runtime=$(echo "$sparse_output" | tail -n 1)
                fastfood_runtime=$(echo "$fastfood_output" | tail -n 1)
                direct_runtime=$(echo "$direct_output" | tail -n 1)

                # Write results in the CSV file
                echo "$sd,$dense_runtime,$sparse_runtime,$fastfood_runtime" >> $output_file
            done

        done
    fi
done