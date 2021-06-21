## DCASE2021 Challenge - Monitoring (Task 2)
This is our submission to task 2 of the 2021's DCASE challenge on *Unsupervised Anomalous Sound Detection for Machine Condition Monitoring under Domain Shifted Conditions*.
For more information go to the [official DCASE website](http://dcase.community/challenge2021/task-unsupervised-detection-of-anomalous-sounds).


### Setup


To reproduce our experiments we recommend a GPU with at least 11GB of VRAM (e.g. NVIDIA GTX 1080Ti) and at least 32 Gb of main memory.


1. Install [miniconda](https://docs.conda.io/en/latest/miniconda.html).
2. Store unzipped data into folder `~/shared/DCASE2021/task2` (or set `--data_root` parameter accordingly)
3. change dir to root of this project
3. run `conda env create -f environment.yaml` to install the conda environment
5. activate the conda environment with `conda activate dcase2021_task2`

**Hint:** Run your experiments with the `--debug` flag to check whether your environment is set up properly. 

### Run Auxiliary Classification Experimentns

To train a classifiers based on auxiliary classification (in this example for machine type fan), simply run:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python -m experiments.train multi_section --version auxiliary_classification --proxy_outliers other_machines --proxy_outlier_lambda 0.5 --machine_type fan
```
Options for proxy_outliers are {*none*, *other_machines*}. 

After training, fine-tuneing can be done with:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python -m experiments.train multi_section --version fine_tune --run_id 97fcd5a6e7d24a8c83f77e286384f5aa --da_task ccsa --da_lambda 1.0 --margin 0.5 --learning_rate 1e-5 --rampdown_length 0 --rampdown_start 3 --max_epochs 3 --proxy_outliers other_machines --machine_type fan
```

where the `run_id` parameter has to be replaced with the run_id of the pre-trained model from the previous step.


### Run Density Estimation & Reconstruction Error Experiments

In a similar vein, density estimation/ reconstruction error models for machine type fan can be trained with the following command:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python -m experiments.train density --version made --architecture made --n_gaussians 1 --proxy_outliers other_sections --proxy_outlier_lambda 1.0 --margin 0.5 --consistent_with_librosa --machine_type fan
```
You can choose the model type by setting the `--architecture` parameter {*AE*, *MADE*, *MAF*}.
Options for proxy_outliers are {*none*, *other_sections*, *other_sections_and_machines*}. 
The `--consistent_with_librosa` flag ensures torchaudio returns the same results as librosa.

### User interface

To view the training progress/ results, change directory to the log directory (`cd logs`) and start the mlflow dashboard with `mlflow ui`.
By default, the dashboard will be served at `http://127.0.0.1:5000`.
