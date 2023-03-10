import os
import sys
import csv
from pytorch_lightning.loggers import CSVLogger

def count_params(model):
    """
    Return the number of trainable parameters of the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_results(model, test_metrics, hyperparams):
    """
    Store the test metrics in a csv file
    """

    test = hyperparams['test']
    outfile = hyperparams['res_dir']
    total_params = count_params(model)

    # ================================
    #     SUBSPACE_DIM VS TEST_ACC    
    # ================================
    if test == "subspace":
        # Define the outfile path wrt the hyperparams
        outfile += f"/subspace_{hyperparams['dataset']}"

        if hyperparams['shuffle_pixels']:
            outfile += f"_shuffled_pixels"
        
        if hyperparams['shuffle_labels']:
            outfile += f"_shuffled_labels"
        
        outfile += f"_{hyperparams['network_type']}"

        if hyperparams['network_type'] == 'fc':
            outfile += f"_depth_{hyperparams['hidden_depth']}_width_{hyperparams['hidden_width']}"
        else:
            outfile += f"_nfeatures_{hyperparams['n_feature']}"
        
        outfile += f"_epochs_{hyperparams['epochs']}_lr_{hyperparams['lr']}.csv"

        # Check if the outfile exist and if it is empty
        if os.path.isfile(outfile) and os.path.getsize(outfile) > 0:
            # Append data
            with open(outfile, mode='a', newline='') as of:
                writer = csv.writer(of)
                row = [hyperparams['subspace_dim'], test_metrics['test_acc'], test_metrics['test_loss']]
                writer.writerow(row)

        else:
            # Write data
            header = ["subspace_dim", "test_acc", "test_loss"]
            
            with open(outfile, mode='w', newline='') as of:
                writer = csv.writer(of)

                # Write the header
                writer.writerow(header)

                # Write data
                row = [hyperparams['subspace_dim'], test_metrics['test_acc'], test_metrics['test_loss']]
                writer.writerow(row)

    # ============================
    #     N_PARAMS VS BASELINE    
    # ============================
    if test == "baseline":
        # Define the outfile path name wrt the hyperparams
        outfile += f"/baseline_{hyperparams['dataset']}"
        
        if hyperparams['shuffle_pixels']:
            outfile += f"_shuffled_pixels"
        
        if hyperparams['shuffle_labels']:
            outfile += f"_shuffled_labels"

        outfile += f"_{hyperparams['network_type']}_epochs_{hyperparams['epochs']}_lr_{hyperparams['lr']}.csv"

        # Check if the outfile exist and it is empty
        if os.path.isfile(outfile) and os.path.getsize(outfile) > 0:
            # Append data
            with open(outfile, mode='a', newline='') as of:
                writer = csv.writer(of)

                if hyperparams['network_type'] == 'fc':
                    row = [hyperparams['hidden_depth'], hyperparams['hidden_width'], total_params, test_metrics['test_acc']]
                else:
                    row = [hyperparams['n_feature'], total_params, test_metrics['test_acc']]

                writer.writerow(row)

        else:
            # Create file
            if hyperparams['network_type'] == 'fc': 
                header = ["depth", "width", "total_params", "baseline"]
            else:
                header = ["n_feature", "total_params", "baseline"]
            
            with open(outfile, mode='w', newline='') as of:
                writer = csv.writer(of)

                # Write the header
                writer.writerow(header)

                # Write data
                if hyperparams['network_type'] == 'fc':
                    row = [hyperparams['hidden_depth'], hyperparams['hidden_width'], total_params, test_metrics['test_acc']]
                else:
                    row = [hyperparams['n_feature'], total_params, test_metrics['test_acc']]

                writer.writerow(row)


class CustomCSVLogger(CSVLogger):
    """
    Custom logger to save the logs in a csv file with hyperparmeters in the filename.
    """
    
    def __init__(self, hyperparams, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hyperparams = hyperparams
        self._version = self.get_version()

    def get_version(self):
        """Return a dynamic version name based on the hyperparameters."""
        
        hp = self.hyperparams

        # FC
        if hp['network_type'] == 'fc':

            if hp['subspace_dim'] is not None:
                if hp['shuffle_pixels'] == True and hp['shuffle_labels'] == True:
                    filename = f"dataset_{hp['dataset']}_shuffle_pixels_shuffle_labels_network_{hp['network_type']}_width_{hp['hidden_width']}_depth_{hp['hidden_depth']}_subspace_dim_{hp['subspace_dim']}_proj_{hp['proj_type']}.csv"

                elif hp['shuffle_pixels'] == True and hp['shuffle_labels'] == False:
                    filename = f"dataset_{hp['dataset']}_shuffle_labels_network_{hp['network_type']}_width_{hp['hidden_width']}_depth_{hp['hidden_depth']}_subspace_dim_{hp['subspace_dim']}_proj_{hp['proj_type']}.csv"

                elif hp['shuffle_pixels'] == False and hp['shuffle_labels'] == True:
                    filename = f"dataset_{hp['dataset']}_shuffle_labels_network_{hp['network_type']}_width_{hp['hidden_width']}_depth_{hp['hidden_depth']}_subspace_dim_{hp['subspace_dim']}_proj_{hp['proj_type']}.csv"

                else:
                    filename = f"dataset_{hp['dataset']}_network_{hp['network_type']}_width_{hp['hidden_width']}_depth_{hp['hidden_depth']}_subspace_dim_{hp['subspace_dim']}_proj_{hp['proj_type']}.csv"
        
            else:
                if hp['shuffle_pixels'] == True and hp['shuffle_labels'] == True:
                    filename = f"dataset_{hp['dataset']}_shuffle_pixels_shuffle_labels_network_{hp['network_type']}_direct_width_{hp['hidden_width']}_depth_{hp['hidden_depth']}.csv"

                elif hp['shuffle_pixels'] == True and hp['shuffle_labels'] == False:
                    filename = f"dataset_{hp['dataset']}_shuffle_pixels_network_{hp['network_type']}_direct_width_{hp['hidden_width']}_depth_{hp['hidden_depth']}.csv"

                elif hp['shuffle_pixels'] == False and hp['shuffle_labels'] == True:
                    filename = f"dataset_{hp['dataset']}_shuffle_labels_network_{hp['network_type']}_direct_width_{hp['hidden_width']}_depth_{hp['hidden_depth']}.csv"

                else:
                    filename = f"dataset_{hp['dataset']}_network_{hp['network_type']}_direct_width_{hp['hidden_width']}_depth_{hp['hidden_depth']}.csv"
        

            # Return the full path to the log file
            return filename
        
        # CNN, LeNet and ResNet20
        else:
            if hp['subspace_dim'] is not None:
                if hp['shuffle_pixels'] == True and hp['shuffle_labels'] == True:
                    filename = f"dataset_{hp['dataset']}_shuffle_pixels_shuffle_labels_network_{hp['network_type']}_n_feature_{hp['n_feature']}_subspace_dim_{hp['subspace_dim']}_proj_{hp['proj_type']}.csv"

                elif hp['shuffle_pixels'] == True and hp['shuffle_labels'] == False:
                    filename = f"dataset_{hp['dataset']}_shuffle_labels_network_{hp['network_type']}_n_feature_{hp['n_feature']}_subspace_dim_{hp['subspace_dim']}_proj_{hp['proj_type']}.csv"

                elif hp['shuffle_pixels'] == False and hp['shuffle_labels'] == True:
                    filename = f"dataset_{hp['dataset']}_shuffle_labels_network_{hp['network_type']}_n_feature_{hp['n_feature']}_subspace_dim_{hp['subspace_dim']}_proj_{hp['proj_type']}.csv"

                else:
                    filename = f"dataset_{hp['dataset']}_network_{hp['network_type']}_n_feature_{hp['n_feature']}_subspace_dim_{hp['subspace_dim']}_proj_{hp['proj_type']}.csv"
        
            else:
                if hp['shuffle_pixels'] == True and hp['shuffle_labels'] == True:
                    filename = f"dataset_{hp['dataset']}_shuffle_pixels_shuffle_labels_network_{hp['network_type']}_direct_n_feature_{hp['n_feature']}.csv"

                elif hp['shuffle_pixels'] == True and hp['shuffle_labels'] == False:
                    filename = f"dataset_{hp['dataset']}_shuffle_pixels_network_{hp['network_type']}_direct_n_feature_{hp['n_feature']}.csv"

                elif hp['shuffle_pixels'] == False and hp['shuffle_labels'] == True:
                    filename = f"dataset_{hp['dataset']}_shuffle_labels_network_{hp['network_type']}_direct_n_feature_{hp['n_feature']}.csv"

                else:
                    filename = f"dataset_{hp['dataset']}_network_{hp['network_type']}_direct_n_feature_{hp['n_feature']}.csv"
        

            # Return the full path to the log file
            return filename