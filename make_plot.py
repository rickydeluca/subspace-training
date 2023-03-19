import csv
import re

import matplotlib.pyplot as plt
import pandas as pd


def get_metadata_from(filename):
    """
    Given the name of a CSV file, return a dictionary containing the metadata
    of the newtwork.
    """
    # Extract metadata from the filename
    filename = filename.split('/')[-1]
    filename = filename.split('.csv')[0]
    filename = filename.split('_')
    
    test = filename[0]
    dataset = None
    network_type = None
    depth = None
    width = None
    n_feature = None    
    n_params = None
    epochs = None
    lr = None
    
    if test == 'subspace':
        dataset = filename[1]
        network_type = filename[2]
        
        if network_type == 'fc':
            depth = filename[4]
            width = filename[6]
            epochs = filename[8]
            lr = filename[10]

        else:
            n_feature = filename[4]
            epochs = filename[6]
            lr = filename[8]

    elif test == 'baseline' or test == 'small-nets':
        dataset = filename[1]
        network_type = filename[2]
        epochs = filename[4]
        lr = filename[6]
    
    elif test == 'time':
        dataset = filename[1]
        network_type = filename[2]
        n_params = filename[3]

    else:
        raise ValueError(f'File with name {filename} not found!')

    # Return a dictionary containing the metadata
    metadata = {
        'dataset': dataset,
        'network_type': network_type,
        'depth': int(depth) if network_type == 'fc' and test == 'subspace' else None,
        'width': int(width) if network_type == 'fc' and test == 'subspace' else None,
        'n_feature': int(n_feature) if network_type != 'fc' and test == 'subspace' else None,
        'epochs': int(epochs) if test == 'subspace' or test == 'baselines' or test == 'small-nets' else None,
        'lr': float(lr) if test == 'subspace' or test == 'baselines' or test == 'small-nets' else None,
        'n_params': n_params if test == 'time' else None
    }
    
    return metadata


def get_baseline(filename, network, depth=None, width=None, n_feature=None):
    """
    Get the baseline test accuracy for the given network with 
    the given parameters.
    """

    if network == 'fc':
        # Read the CSV file as a pandas dataframe
        data = pd.read_csv(filename, header=0, names=["depth", "width", "total_params", "baseline"])

        # Return the baseline value associated with the given depth and width
        return data[(data['depth'] == depth) & (data['width'] == width)]['baseline'].values[0]

    
    if network == 'lenet' or network == 'resnet20':
        # Read the CSV file as a pandas dataframe
        data = pd.read_csv(filename, header=0, names=["n_feature", "total_params", "baseline"])

        # Return the baseline value associated with the given number of features
        return data[data['n_feature'] == n_feature]['baseline'].values[0]
   
    
def get_intrinsic_dim(filename, network, baseline_90, depth=None, width=None, n_feature=None):
    """
    Get the intrinsic dimension (the subspace dimension that produces a 
    test accouracy equal to 90% of the baseline) for the given network
    with the given parameters.
    """

def proj_time(filename, metadata):
    """
    Plot the execution time for each projection method.
    """
    # Read CSV file into a pandas dataframe
    data = pd.read_csv(filename)

    # Extract data from each column
    subspace_dim = data['subspace_dim']
    time_dense= data['dense (ms)']
    time_sparse = data['sparse (ms)']
    time_fastfood = data['fastfood (ms)']
    time_direct = data['direct (ms)']

    # Close all previously opened plots
    plt.close('all')

    # Create a dotted plot with connected lines
    plt.plot(subspace_dim, time_dense, 'o-', label='dense')
    plt.plot(subspace_dim, time_sparse, 'o-', label='sparse')
    plt.plot(subspace_dim, time_fastfood, 'o-', label='fastfood')
    plt.plot(subspace_dim, time_direct, 'o--', color='gray', alpha=0.5, label='direct')

    # Set plot title and axis labels
    plt.suptitle("Time vs. Intrinsic Dimension")
    plt.title(f"{metadata['network_type'].upper()} | {metadata['dataset'].upper()} | {metadata['n_params'].upper()}")
    plt.xlabel("intrinsic dimension")
    plt.ylabel("time (ms)")

    # Refine plot
    plt.xticks(range(0, max(subspace_dim)+1, int(max(subspace_dim)/10)))
    plt.gcf().set_size_inches(10, 5)
    
    # Add legend
    plt.legend()

    # Save the plot as a PNG file
    outfile = f"plots/time/time_{metadata['dataset']}_{metadata['network_type']}_{metadata['n_params']}.png"
    plt.savefig(outfile,  bbox_inches='tight')


def subspace_dim_vs_accuracy(filename, outfile, baseline):
    """
    Plot the test accuracy for each subspace dimension wrt the baseline.
    """
    # Read CSV file into a pandas dataframe
    data = pd.read_csv(filename, header=0, names=["subspace_dim", "test_accuracy", "test_loss"])
    
    # Extract data from each column
    subspace_dim = data['subspace_dim']
    test_accuracy = data['test_accuracy']

    # Close all previously opened plots
    plt.close('all')

    # Plot test accuracy
    plt.plot(subspace_dim, test_accuracy, '-o', alpha=1.0)

    # Plot baseline
    plt.axhline(y=baseline, linestyle='-', color='gray', alpha=1.0)
    
    # Plot 90% baseline
    plt.axhline(y=baseline*0.9, linestyle='--', color='gray', alpha=1.0)

    # Plot legend for baseline and 90% baseline
    plt.legend(['test accuracy', 'baseline', '90% baseline'])

    # Set plot title and axis labels
    plt.title('test accuracy vs. subspace dimension')
    plt.xlabel('subspace dimension')
    plt.ylabel('test accuracy')

    # Save the plot as a PNG file
    plt.savefig(outfile, bbox_inches='tight')


def subspace_vs_small_networks_accuracy(subspace_file, small_net_file, subspace_metadata, small_net_metadata, outfile):
    """
    Plot the test accuracy of a standard size network but trained with
    subspace training vs. the test accuracy of naturally smaller networks.
    """

    # Read the test accuracy of the subspace trained network
    subspace_data = pd.read_csv(subspace_file, header=0, names=["subspace_dim", "test_accuracy", "test_loss"])
    subspace_dims = subspace_data['subspace_dim']
    subspace_accuracies = subspace_data['test_accuracy']

    # Read the test accuracy of the naturally smaller networks
    if small_net_metadata['network_type'] == 'fc':  # Fully connected network
        small_net_data = pd.read_csv(small_net_file, header=0, names=["depth", "width", "total_params", "baseline"])
        small_net_dims = small_net_data['total_params']
        small_net_accuracies = small_net_data['baseline']
    else: # LeNet or ResNet20
        small_net_data = pd.read_csv(small_net_file, header=0, names=["n_feature", "total_params", "baseline"])
        small_net_dims = small_net_data['total_params']
        small_net_accuracies = small_net_data['baseline']

    # Close all previously opened plots
    plt.close('all')

    # Plot test accuracy
    plt.plot(subspace_dims, subspace_accuracies, '-o', alpha=1.0)
    plt.plot(small_net_dims, small_net_accuracies, '-o', alpha=1.0)

    # Add legend
    plt.legend(['subspace network', 'small networks'])

    # Set plot title and axis labels
    plt.title('subspace vs. small network accuracy')

    # Save the plot as a PNG file
    plt.savefig(outfile, bbox_inches='tight')


def main(test):

    # ==============================================
    #     projection time vs. intrisic dimension
    # ==============================================
    if test == "proj" or test == "all":
        filename1 = "results/proj_time/time_mnist_fc_100k.csv"
        filename2 = "results/proj_time/time_mnist_fc_1m.csv"
        proj_time(filename1, get_metadata_from(filename1))
        proj_time(filename2, get_metadata_from(filename2))
    
    # ==============================================
    #     subspace dimension vs. test accuracy
    # ==============================================
    if test == "subspace" or test == "all":
        for dataset in ['mnist', 'cifar10' 'mnist_shuffled_pixels', 'mnist_shuffled_labels', 'mnist_shuffled_pixels_shuffled_labels', 'cifar10_shuffled_pixels', 'cifar10_shuffled_labels', 'cifar10_shuffled_pixels_shuffled_labels']:
            for network in ['fc', 'lenet', 'resnet20']:
                if network == 'fc':  # Fully connected network
                    for depth in [1,2,3,4,5]:
                        for width in [50, 100, 200, 400]:
                            try:
                                filename=f"results/subspace/subspace_{dataset}_{network}_depth_{depth}_width_{width}_epochs_10_lr_0.003.csv"
                                outfile=f"plots/subspace/subspace_{dataset}_{network}_depth_{depth}_width_{width}_epochs_10_lr_0.003.png"
                                baseline=get_baseline(f"results/baseline/baseline_{dataset}_{network}_epochs_10_lr_0.003.csv", network, depth=depth, width=width)
                                subspace_dim_vs_accuracy(filename, outfile, baseline)
                            except:
                                print(f"Could not find file {filename} or {baseline}!")

                elif network == 'lenet':  # LeNet
                    for n_feature in range(1, 21, 1):
                        try:
                            filename=f"results/subspace/subspace_{dataset}_{network}_nfeatures_{n_feature}_epochs_10_lr_0.003.csv"
                            outfile=f"plots/subspace/subspace_{dataset}_{network}_nfeatures_{n_feature}_epochs_10_lr_0.003.png"
                            baseline=get_baseline(f"results/baseline/baseline_{dataset}_{network}_epochs_10_lr_0.003.csv", network, n_feature=n_feature)
                            subspace_dim_vs_accuracy(filename, outfile, baseline)
                        except:
                            print(f"Could not find file {filename} or {baseline}!")

                else:  # ResNet20
                    try:
                        filename=f"results/subspace/subspace_{dataset}_{network}_nfeatures_16_epochs_10_lr_0.003.csv"
                        outfile=f"plots/subspace/subspace_{dataset}_{network}_nfeatures_16_epochs_10_lr_0.003.png"
                        baseline=get_baseline(f"results/baseline/baseline_{dataset}_{network}_epochs_10_lr_0.003.csv", network, n_feature=16)
                        subspace_dim_vs_accuracy(filename, outfile, baseline)
                    except:
                        print(f"Could not find file {filename} or {baseline}!")

    # ==============================================
    #     subspace networks vs. small networks
    # ============================================== 
    if test == "small-nets" or test == "all":
        # Comparison between the accuracy of smaller FC networs
        # and the accuracy of a 784-200-200-10 network with different intrinsic dimensions
  
        subspace_file = f"results/subspace/subspace_mnist_fc_depth_2_width_200_epochs_10_lr_0.003.csv"
        small_net_file = f"results/small-nets/small-nets_mnist_fc_epochs_10_lr_0.003.csv"
        outfile = f"plots/small-nets/subspace_vs_small-nets_mnist_fc_depth_2_width_200_epochs_10_lr_0.003.png"

        subspace_vs_small_networks_accuracy(subspace_file, small_net_file, get_metadata_from(subspace_file), get_metadata_from(small_net_file), outfile)



if __name__ == '__main__':
    main("small-nets")