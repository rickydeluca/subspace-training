import os
import sys
from pytorch_lightning.loggers import CSVLogger

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