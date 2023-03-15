import csv
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

    elif test == 'baselines':
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
        'epochs': int(epochs),
        'lr': float(lr),
        'n_params': int(n_params) if test == 'time' else None
    }
    
    return metadata


def get_baseline(filename, network, depth=None, width=None, n_feature=None):
    """
    Get the baseline test accuracy for the given network with 
    the given parameters.
    """

    if network == 'fc':
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0] == depth and row[1] == width:
                    return row[3]
        return None
    
    if network == 'lenet' or network == 'resnet20':
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0] == n_feature:
                    return row[2]
        return None
    
    
def get_intrinsic_dim(filename, network, baseline_90, depth=None, width=None, n_feature=None):
    """
    Get the intrinsic dimension (the subspace dimension that produces a 
    test accouracy equal to 90% of the baseline) for the given network
    with the given parameters.
    """
    

        

def proj_time(filename):
    """
    Plot the execution time for each projection method.
    """
    # Read CSV file into a pandas dataframe
    data = pd.read_csv(filename)

    # Remove any data points with "inf" values
    # data = data.replace([float('inf'), float('-inf')], pd.NA).dropna()

    # Extract data from each column
    subspace_dim = data['subspace_dim']
    time_dense= data['dense (s)']
    time_sparse = data['sparse (s)']
    time_fastfood = data['fastfood (s)']

    # Close all previously opened plots
    plt.close('all')

    # Create a dotted plot with connected lines
    plt.plot(subspace_dim, time_dense, 'o-', label='dense')
    plt.plot(subspace_dim, time_sparse, 'o-', label='sparse')
    plt.plot(subspace_dim, time_fastfood, 'o-', label='fastfood')

    # Add an "x" marker at the last data point of each algorithm
    line_color = plt.gca().get_lines()[0].get_color()
    plt.scatter(subspace_dim.iloc[-1], time_dense.iloc[-1], marker='x', color=line_color)
    line_color = plt.gca().get_lines()[1].get_color()
    plt.scatter(subspace_dim.iloc[-1], time_sparse.iloc[-1], marker='x', color=line_color)
    line_color = plt.gca().get_lines()[2].get_color()
    plt.scatter(subspace_dim.iloc[-1], time_fastfood.iloc[-1], color=line_color)

    # Set plot title and axis labels
    plt.title('time vs. intrinsic dimension')
    plt.xlabel('intrinsic dimension')
    plt.ylabel('time (s)')

    # Add legend to the plot
    plt.legend()

    # Save the plot as a PNG file
    plt.savefig(f'test.png')


def subspace_dim_vs_accuracy(filename, baseline):
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

    # Create a line plot with markers
    plt.plot(subspace_dim, test_accuracy, '-o', alpha=0.5)

    # Add a horizontal dotted line for the baseline test loss value
    plt.axhline(y=baseline, linestyle='--', color='gray', alpha=0.5)

    # Set plot title and axis labels
    plt.title('test accuracy vs. subspace dimension')
    plt.xlabel('subspace dimension')
    plt.ylabel('test accuracy')

    # Save the plot as a PNG file
    plt.savefig(f'test2.png', bbox_inches='tight')


def intrinsic_dim_vs_num_of_params(subspace_file, baseline_file):
    """
    Plot the intrinsic dimension (computed as 90% of the baseline) vs the number
    of parameters of the network.
    """

    # Extract metadata from the filename
    subspace_metadata = get_metadata_from(subspace_file)
    subspace_baseline = get_metadata_from(baseline_file)

    network = subspace_metadata['network_type']
    depth = subspace_metadata['depth']
    width = subspace_metadata['width']
    n_feature = subspace_metadata['n_feature']

    # Get the baseline
    baseline = get_baseline(baseline_file, network, depth=depth, width=width, n_feature=n_feature)
    baseline_90 = 0.9 * float(baseline)

    # Get the relative intrinsic dimension


    exit(0)


def main():
    # proj_time("results/proj_time/time_mnist_fc_100k.csv")
    # subspace_dim_vs_accuracy("results/subspace/subspace_mnist_fc_depth_1_width_400_epochs_10_lr_0.003.csv", 0.9)
    intrinsic_dim_vs_num_of_params("results/subspace/subspace_mnist_fc_depth_1_width_400_epochs_10_lr_0.003.csv", "results/baseline/baseline_mnist_fc_epochs_10_lr_0.003.csv")


if __name__ == '__main__':
    main()