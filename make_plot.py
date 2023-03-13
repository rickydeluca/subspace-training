import csv
import os
import matplotlib.pyplot as plt
import pandas as pd

def get_baseline_fc(filename, depth, width):
    """
    Get the baseline test accuracy for the fully-connected network
    with the given depth and width.
    """
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] == depth and row[1] == width:
                return row[3]
    return None

def get_baseline_cnn(filename, n_feauture):
    """
    Get the baseline test accuracy for the CNN with the given number of features.
    """
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] == n_feauture:
                return row[2]
    return None


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
    time_dense= data['dense (ms)']
    time_sparse = data['sparse (ms)']
    time_fastfood = data['fastfood (ms)']

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

def main():
    # # Iterate over all the CSV files in the results directory
    # for filename in os.listdir('results'):
    #     # Skip the baseline CSV files
    #     if filename.startswith('baseline'):
    #         continue

    #     # Get the baseline test accuracy for the current experiment
    #     if filename.startswith('fc'):
    #         depth, width = filename.split('_')[1:3]
    #         baseline = get_baseline_fc('results/baseline_fc.csv', depth, width)
    #     elif filename.startswith('cnn'):
    #         n_feature = filename.split('_')[1]
    #         baseline = get_baseline_cnn('results/baseline_cnn.csv', n_feature)
    #     else:
    #         raise ValueError(f'Invalid filename: {filename}')

    #     # Plot the test accuracy for each subspace dimension wrt the baseline
    #     subspace_dim_vs_accuracy(filename, baseline)

    #     # Plot the execution time for each projection method
    #     proj_time(filename)
    proj_time("results/proj_time/time_mnist_fc_100k.csv")
    subspace_dim_vs_accuracy("results/subspace/subspace_mnist_fc_depth_1_width_400_epochs_10_lr_0.003.csv", 0.9)
if __name__ == '__main__':
    main()