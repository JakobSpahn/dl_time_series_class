# dl_time_series_class

***Deep Learning for time series data: A survey and experimental study***

## Results

### UCR Archive

* [Raw performance metrics](results/results_ucr.csv)

### [Requirements](requirements.txt)

* Python;
* Matplotlib
* Numba;
* NumPy;
* Pandas
* scikit-learn (or equivalent).
* sktime
* scipy
* TensorFlow-GPU
* tqdm

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

<div align="center">:rocket:</div>