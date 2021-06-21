# DCASE2021 Challenge - Task 2 - Machine Condition Monitoring

This is our submission to task 2 of the 2021's DCASE challenge on *Unsupervised Anomalous Sound Detection for Machine Condition Monitoring under Domain Shifted Conditions*.
For more information visit the [official DCASE website](http://dcase.community/challenge2021/task-unsupervised-detection-of-anomalous-sounds).


## Setup

To reproduce our experiments we recommend a GPU with at least 11GB of VRAM (e.g. NVIDIA GTX 1080Ti) and at least 32 Gb of main memory.


1. install [miniconda](https://docs.conda.io/en/latest/miniconda.html).
2. download the [data](http://dcase.community/challenge2021/task-unsupervised-detection-of-anomalous-sounds#download)
3. store unzipped data into folder `~/shared/DCASE2021/task2` (or set `--data_root` parameter accordingly)
4. change dir to root of this project
5. run `conda env create -f environment.yaml` to install the conda environment
6. activate the conda environment with `conda activate dcase2021_task2`

**Hint:** Run your experiments with the `--debug` flag to check whether your environment is set up properly. 

## Run Auxiliary Classification Experimentns

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


## Run Density Estimation & Reconstruction Error Experiments

In a similar vein, density estimation/ reconstruction error models for machine type fan can be trained with the following command:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python -m experiments.train density --version made --architecture made --n_gaussians 1 --proxy_outliers other_sections --proxy_outlier_lambda 1.0 --margin 0.5 --consistent_with_librosa --machine_type fan
```
You can choose the model type by setting the `--architecture` parameter {*AE*, *MADE*, *MAF*}.
Options for proxy_outliers are {*none*, *other_sections*, *other_sections_and_machines*}. 
The `--consistent_with_librosa` flag ensures torchaudio returns the same results as librosa.

## Dashboard

To view the training progress/ results, change directory to the log directory (`cd logs`) and start the mlflow dashboard with `mlflow ui`.
By default, the dashboard will be served at `http://127.0.0.1:5000`.


## References

- Yohei Kawaguchi, Keisuke Imoto, Yuma Koizumi, Noboru Harada, Daisuke Niizumi, Kota Dohi, Ryo Tanabe, Harsh Purohit, and Takashi Endo. [*Description and discussion on DCASE 2021 challenge task 2: unsupervised anomalous sound detection for machine condition monitoring under domain shifted conditions*](https://arxiv.org/pdf/2106.04492.pdf). In arXiv e-prints: 2106.04492, 1–5, 2021. 
- Ryo Tanabe, Harsh Purohit, Kota Dohi, Takashi Endo, Yuki Nikaido, Toshiki Nakamura, and Yohei Kawaguchi. [*MIMII DUE: sound dataset for malfunctioning industrial machine investigation and inspection with domain shifts due to changes in operational and environmental conditions*](https://arxiv.org/pdf/2105.02702.pdf). In arXiv e-prints: 2006.05822, 1–4, 2021.
- Noboru Harada, Daisuke Niizumi, Daiki Takeuchi, Yasunori Ohishi, Masahiro Yasuda, and Shoichiro Saito. [*ToyADMOS2: another dataset of miniature-machine operating sounds for anomalous sound detection under domain shift conditions*](https://arxiv.org/pdf/2106.02369.pdf). arXiv preprint arXiv:2106.02369, 2021.

## Citation
If you use any parts of the implementation please cite our report:
```
@techreport{Primus2021DCASEChallenge,
    Author      =   {Primus, Paul and Zwifl, Martin and Widmer, Gerhard},
    institution =   {{DCASE2021 Challenge}},
    title       =   {CP-JKU Submission to DCASE'21: Improving Out-of-Distribution Detectors for Machine Condition Monitoring with Proxy Outliers \& \\ Domain Adaptation via Semantic Alignment},
    month       =   {June},
    year        =   2021
}
```

If you use the ResNet model or the model implementation please cite the following paper:
```
@inproceedings{Koutini2019Receptive,
    author      =   {Koutini, Khaled and Eghbal-zadeh, Hamid and Dorfer, Matthias and Widmer, Gerhard},
    title       =   {{The Receptive Field as a Regularizer in Deep Convolutional Neural Networks for Acoustic Scene Classification}},
    booktitle   =   {Proceedings of the European Signal Processing Conference (EUSIPCO)},
    address     =   {A Coru\~{n}a, Spain},
    year        =   2019
}
```

## Links
- [DCASE Community](http://dcase.community/)
- [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning)
- [mlflow](https://mlflow.org/)
