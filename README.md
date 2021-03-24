# dl_time_series_class

***Deep Learning for time series data: A survey and experimental study***

## Results

### UCR Archive

* [Raw performance metrics](results/results.csv)

### [Requirements](requirements.txt)

* Python;
* Matplotlib
* Numba;
* NumPy;
* Pandas;
* scikit-learn;
* sktime;
* scipy;
* TensorFlow-GPU;
* tqdm.

## Usage

### [`main.py`](main.py)

```
Arguments:
-d --dataset_names          : dataset names (optional, default=all)
-c --classifier_names       : classifier (optional, default=all)
-o --output_path            : path to results (optional, default=root_dir)
-i --iterations             : number of runs (optional, default=3)
-g --generate_results_csv   : make results.csv (optional, default=False)

Examples:
> python main.py
> python main.py -d Adiac Coffee -c rocket_tf mlp -i 1
> python main.py -g True
```
The framework expects data from the UCR archive in the .ts format.

The folder structure for the datasets is as follows: <root>/UCRArchive_2018/dataset_name/
  
For example, the train/test of Adiac should be saved under /UCRArchive_2018/Adiac/


Calling main.py without any arguments trains every model on every dataset.


Results are saved in <root>/results.
  

To generate a results.csv for the tested models, main.py -g True is called.
