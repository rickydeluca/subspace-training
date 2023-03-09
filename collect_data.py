import sys

import pandas as pd
import torch

from main import BATCH_SIZE, PATH_DATASETS, setup_model, setup_trainer

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Choose which test to run
test = [0, 1]    # if test_i == 1, run it, otherwise skip it

# Define hyperparams
hidden_depths = [1, 2, 3, 4, 5]
hidden_widths = [50, 100, 200, 400]
subspace_dims = list(range(0, 1600, 100)) # 0, 100, 200, ..., 1500
n_feature = list(range(1, 31, 1)) # 1, 2, 3, ..., 30
network_types = ["fc", "lenet", "resnet20"]
datasets = ["mnist", "cifar10"]
epochs = 10
lr = 3e-3

# ==================================================================
# 1) TEST ACCURACY VS SUBSPACE DIMENSION
# (with different hidden depths, widths and number of features)
# ==================================================================
if test[0] == 1:

    for d in datasets:
        for n in network_types:

            # If it is a fully connected network, test for different width and depth size
            if n == "fc":
                for hd in hidden_depths:
                    for hw in hidden_widths:

                        # Init dictionary to store the results of learning
                        data = {"subspace_dim": [],
                                "test_acc":     [],
                                "test_loss":    []}

                        for s in subspace_dims:
                            if s == 0:  # No learning is possibile with zero parameters
                                data["subspace_dim"].append(s)
                                data["test_loss"].append(sys.float_info.max)
                                data["test_acc"].append(0)
                                continue

                            hyperparams = {
                                "dataset":          d,
                                "network_type":     n,
                                "subspace_dim":     s,
                                "proj_type":        "sparse",
                                "deterministic":    True,
                                "shuffle_pixels":   False,
                                "shuffle_labels":   False,
                                "lr":               lr,
                                "epochs":           epochs,
                                "logs_dir":         "logs/",
                                "hidden_width":     hw,
                                "hidden_depth":     hd,
                                "n_feature":        0           # Not used in FCN
                            }

                            # Setup model
                            data_module, model = setup_model(hyperparams)

                            # Setup trainer
                            trainer = setup_trainer(hyperparams)

                            # Train and log
                            trainer.fit(model, data_module)

                            # Test and log
                            test_metrics = trainer.test(model, data_module)

                            # Append results to the dictionary
                            data["subspace_dim"].append(s)
                            data["test_loss"].append(test_metrics[0]["test_loss"])
                            data["test_acc"].append(test_metrics[0]["test_acc"])

                            # Free memory to handle future iterations
                            del model
                            torch.cuda.empty_cache()
                            torch.cuda.reset_max_memory_allocated()

                        # Convert the dictionary to a dataframe
                        df = pd.DataFrame.from_dict(data)

                        # Save the dataframe to a csv file
                        df.to_csv(f"results/subspace_{d}_{n}_width_{hw}_depth_{hd}_epochs_{epochs}_lr_{lr}.csv")

            # If it is a convolutional network, test for different number of features
            else:
                for nf in n_feature:

                    # Init dictionary to store the results of learning
                    data = {"subspace_dim": [],
                            "test_acc":     [],
                            "test_loss":    []}

                    for s in subspace_dims:
                        if s == 0:  # No learning is possibile with zero parameters
                            data["subspace_dim"].append(s)
                            data["test_loss"].append(sys.float_info.max)
                            data["test_acc"].append(0)
                            continue
                        
                        hyperparams = {
                            "dataset":          d,
                            "network_type":     n,
                            "subspace_dim":     s,
                            "proj_type":        "sparse",
                            "deterministic":    True,
                            "shuffle_pixels":   False,
                            "shuffle_labels":   False,
                            "lr":               lr,
                            "epochs":           epochs,
                            "logs_dir":         "logs/",
                            "hidden_width":     0,        # Not used in CNN
                            "hidden_depth":     0,        # Not used in CNN
                            "n_feature":        nf
                        }

                        # Setup model
                        data_module, model = setup_model(hyperparams)

                        # Setup trainer
                        trainer = setup_trainer(hyperparams)

                        # Train and log
                        trainer.fit(model, data_module)

                        # Test and log
                        test_metrics = trainer.test(model, data_module)

                        # Append results to the dictionary
                        data["subspace_dim"].append(s)
                        data["test_loss"].append(test_metrics[0]["test_loss"])
                        data["test_acc"].append(test_metrics[0]["test_acc"])

                        # Free memory to handle future iterations
                        del model
                        torch.cuda.empty_cache()
                        torch.cuda.reset_max_memory_allocated()

                    # Convert the dictionary to a dataframe
                    df = pd.DataFrame.from_dict(data)

                    # Save the dataframe to a csv file
                    df.to_csv(f"results/subspace_{d}_{n}_nfeature_{nf}_epochs_{epochs}_lr_{lr}.csv")


# ==================================================================
# 2) BASELINES
# (test accuracy of the direct models)
# ==================================================================
if test[1] == 1:
    for d in datasets:
        for n in network_types:
            
            # FCNs
            if n == "fc":
                # Init dictionary to store the results of learning
                data = {"depth":        [],
                        "width":        [],
                        "total_params": [],
                        "baseline":     []}
                        
                for hd in hidden_depths:
                    for hw in hidden_widths:
                        
                        hyperparams = {
                            "dataset":          d,
                            "network_type":     n,
                            "subspace_dim":     None,       # No subspace learning
                            "proj_type":        "sparse",
                            "deterministic":    True,
                            "shuffle_pixels":   False,
                            "shuffle_labels":   False,
                            "lr":               lr,
                            "epochs":           epochs,
                            "logs_dir":         "logs/",
                            "hidden_width":     hw,
                            "hidden_depth":     hd,
                            "n_feature":        0           # Not used in FCN
                        }

                        # Setup model
                        data_module, model = setup_model(hyperparams)
                        
                        # Setup trainer
                        trainer = setup_trainer(hyperparams)

                        # Train and log
                        trainer.fit(model, data_module)

                        # Test and log
                        test_metrics = trainer.test(model, data_module)

                        # Append results to the dictionary
                        data["depth"].append(hd)
                        data["width"].append(hw)
                        data["total_params"].append(count_params(model))
                        data["baseline"].append(test_metrics[0]["test_acc"])

                        # Free memory to handle future iterations
                        del model
                        torch.cuda.empty_cache()
                        torch.cuda.reset_max_memory_allocated()
                        
                    
                # Convert the dictionary to dataframe and save it to a csv file
                df = pd.DataFrame.from_dict(data)
                df.to_csv(f"results/baselines_{d}_{n}_epochs_{epochs}_lr_{lr}.csv")
            
            # CNNs
            else:
                # Init dictionary to store the results of learning
                data = {"n_feature":    [],
                        "total_params": [],
                        "baseline":     []}
                        
                for nf in n_feature:

                    hyperparams = {
                        "dataset":          d,
                        "network_type":     n,
                        "subspace_dim":     None,       # No subspace learning
                        "proj_type":        "sparse",
                        "deterministic":    True,
                        "shuffle_pixels":   False,
                        "shuffle_labels":   False,
                        "lr":               lr,
                        "epochs":           epochs,
                        "logs_dir":         "logs/",
                        "hidden_width":     0,          # Not used in CNN
                        "hidden_depth":     0,          # Not used in CNN
                        "n_feature":        nf
                    }

                    # Setup model
                    data_module, model = setup_model(hyperparams)
                    
                    # Setup trainer
                    trainer = setup_trainer(hyperparams)

                    # Train and log
                    trainer.fit(model, data_module)

                    # Test and log
                    test_metrics = trainer.test(model, data_module)

                    # Append results to the dictionary
                    data["n_feature"].append(nf)
                    data["total_params"].append(count_params(model))
                    data["baseline"].append(test_metrics[0]["test_acc"])

                    # Free memory to handle future iterations
                    del model
                    torch.cuda.empty_cache()
                    torch.cuda.reset_max_memory_allocated()
                    
                
                # Convert the dictionary to dataframe and save it to a csv file
                df = pd.DataFrame.from_dict(data)
                df.to_csv(f"results/baselines_{d}_{n}_epochs_{epochs}_lr_{lr}.csv")