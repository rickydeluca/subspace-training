# Subspace Training
PyTorch implementation of a NN able to reduce its hypothesis space by projecting the training parameters in a lower dimensional space, as described by Li et al. (2018) (https://openreview.net/forum?id=ryup8-WCW). This is a project for the course of Deep Learning and Applied Artficial Intelligence (2021-22), Sapienza University of Rome.

## Prerequisites
Please see ```requirements.txt``` to see the list of needed packages.

If you are using a __conda__ you can create the environment with all the required packages to run the experiments with:
```
conda env create -f environment.yaml
```

Or, if you prefer to use __pip__:
```
pip install -r requirements.txt
```

## How to run
Activate the _virtual environment_ if you are using one.

You __do not__ need to manually download the dataset beacuse the script will automatically check if they are already present in your data folder and if not, it will download them.

To run one experiment you can use:

```
python3 main.py --database --network --lr --epochs --subspace_dim --proj --deterministic --shuffle_pixels --shuffle_labels --hidden_width --hidden_depth --n_feature --logs_dir --res_dir --test
```
Where:

| Argument 	| What it does 	|
|---	|---	|
| database  | The dataset to use. You can choose between: "mnist" or "cifar10". (default: mnist) |
| network | Which neural network to use. You can choose between: "fc", "lenet", "resnet20". (default: "fc") |
| lr  | Learning rate. (default: 3e-3)  |
| epochs  | Number of epochs for training. (default: 10)  |
| subspace_dim  | The number of parameters to use with subspace training. If None do not use subspace trainig. (default: None) |
| proj  | The type of projection you want to use for subspace training. You can choose between: "dense", "sparse" or "fastfood". If subspace dimension is None this parameter will be ignored. (default: "dense")') |
| deterministic | 1 if you want we want the training to act deterministically, 0 otherwise. (default: 1) |
| shuffle_pixels  | 1 if you shuffle the pixels of the input images. 0 otherwise. (default: 0) |
| shuffle_labels  | 1 if you want to shuffle the labels in the training dataset, 0 otherwise. (default: 0)  |
| hidden_width  | The size of the hidden layers in the FC network. This parameter is ignored if other networks were chosen. (default: 100)  |
| hidden_depth  | How many hidden layer in the FC network. This parameter is ignored if other networks were chosen. (default: 1)  |
| n_feature  | Number of initial features for LeNet. This parameter is ignored if other networks were chosen. (default: 6)  |
| logs_dir  | Path to the directory in which store the training logs. (default: "logs/")  |
| res_dir  | Path to the directory in which store the test metrics. (default: "results/")  |
| test  | Which type of data we are collecting: "subspace" for subspace dim vs test accuracy; "baseline" for num of params VS baseline value; "small-nets" to compute and store the accuracy for natural small networks; "time" for forward+backword pass time. It influences the structure of the output file, its filename and in which sub-directory it will be stored. If None then do not store the output results. (default: None)  |

For example: 
```
python3 main.py --database cifar10 --network lenet --subspace_dim 1000 --proj sparse --shuffle_pixels 1 --test subspace
```
Will run one experiment of CIFAR-10 dataset using LeNet and the subspace training technique. The dimension of the subspace parameter space will be 1000 and the projection method will be the Fastfood transform. Also the pixels of the input images will be shuffles. If not exists yet, a new CSV file called ```subspace_cifar10_lenet_nfeatures_6_epochs_10_lr_0.003.csv``` will be created and stored in ```results/subspace```. Otherwise the output will be appended to the already existing file.
All other parameters will have the default value.

### Run multiple experiments at once
```run_test.sh``` is a script the run multiple experiments to collect the data for different metrics. The metrics collected with this script was used to create the plots that you can see in ```plots/```folder.

To use it you need to give it the execution permissions with:
```
chmod +x run_tests.sh
```

and then run it with:
```
./run_tests.sh
```

## Make plots
Once you collected the data you can create plots from them using ```make_plot.py```.

__NB:__ This script was made to make the plots for the report and it is not very customizable. Please read carefully the code before using it and make sure that you have all the required data if you do not want to incurr in errors.

The script generate different plots for different type of data.
To run it use:
```python3 make_plot.py --plot plot_name```

Where, the test name defines the type of plot you want to generate. It can be:
| plot_name 	| plot 	|
|---	|---	|
| subspace  | Using the files in ```results/subspace``` and in ```results/baseling`` plots the test accuracy wrt the subspace dimension for all the network tested on different datasets showing the relative baseline (and 90% baseline) for that dataset with that network |
| small-nets | Test accuracy of standars NN but trained with subspace training vs the test accuracy of naturally smaller networks |
| time  | Plots the average time of one backward + one forward pass of the different projection methods using 2 types of fully connected networks (100k and 1 million parameters) |
