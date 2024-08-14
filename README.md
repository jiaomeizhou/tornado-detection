# tornardo-detection
Adopt CNNs on the tornado detection task by using Tornet dataset

## Dataset
TorNet: the TorNet dataset as described in the paper [*A Benchmark Dataset for Tornado Detection and Prediction using Full-Resolution Polarimetric Weather Radar Data*](https://arxiv.org/abs/2401.16437)

## Downloading the Data

The TorNet dataset can be downloaded from the following location:

#### Zenodo

TorNet is split across 10 files, each containing 1 year of data. There is also a catalog CSV file that is used by some functions in this repository.    

* Tornet 2013 (3 GB) and catalog: [https://doi.org/10.5281/zenodo.12636522](https://doi.org/10.5281/zenodo.12636522)
* Tornet 2014 (15 GB): [https://doi.org/10.5281/zenodo.12637032](https://doi.org/10.5281/zenodo.12637032)
* Tornet 2015 (17 GB): [https://doi.org/10.5281/zenodo.12655151](https://doi.org/10.5281/zenodo.12655151)
* Tornet 2016 (16 GB): [https://doi.org/10.5281/zenodo.12655179](https://doi.org/10.5281/zenodo.12655179)
* Tornet 2017 (15 GB): [https://doi.org/10.5281/zenodo.12655183](https://doi.org/10.5281/zenodo.12655183)
* Tornet 2018 (12 GB): [https://doi.org/10.5281/zenodo.12655187](https://doi.org/10.5281/zenodo.12655187)
* Tornet 2019 (18 GB): [https://doi.org/10.5281/zenodo.12655716](https://doi.org/10.5281/zenodo.12655716)
* Tornet 2020 (17 GB): [https://doi.org/10.5281/zenodo.12655717](https://doi.org/10.5281/zenodo.12655717)
* Tornet 2021 (18 GB): [https://doi.org/10.5281/zenodo.12655718](https://doi.org/10.5281/zenodo.12655718)
* Tornet 2022 (19 GB): [https://doi.org/10.5281/zenodo.12655719](https://doi.org/10.5281/zenodo.12655719)

If downloading through your browser is slow, we recommend downloading these using `zenodo_get` (https://gitlab.com/dvolgyes/zenodo_get).

After downloading, there should be 11 files, `catalog.csv`, and 10 files named as `tornet_YYYY.tar.gz`.   Move and untar these into a target directory, which will be referenced using the `TORNET_ROOT` environment variable in the code.  After untarring the 10 files, this directory should contain `catalog.csv` along with sub-directories `train/` and `test/` filled with `.nc` files for each year in the dataset.

For this project, we only use the subset of data: 2013, 2021.

## Setup

Basic python requirements are listed in `requirements/basic.txt`.

The `tornet` package can then installed into your environment by running

`pip install .`

In this repo.  To do ML with TorNet, additional installs may be necessary depending on library of choice.  `requirements/tensorflow.txt`.

### Conda

If using conda

```
conda create -n tornet-{backend} python=3.10
conda activate tornet-{backend}
pip install -r requirements/{backend}.txt
```

Replace {backend} with tensorflow.

## Notebooks
The following Jupyter notebooks are included in this project:

1. baselineCNN_training_evaluation.ipynb: Contains the training and evaluation of a baseline CNN model.
2. MobileNetV2_evaluate_analysis.ipynb: Analysis and evaluation of the MobileNetV2 model for tornado detection on different types of tornado data.
3. MobileNetV2.ipynb: Details the implementation and training of the MobileNetV2 model.
4. ModelTraining_keras_features.ipynb: Demonstrates model training using Keras with additional feature engineering.

## Tornet Directory
The ```tornet/``` directory contains all the critical code and configurations for the model implementation:

- Model Implementation: Code files detailing the construction of the CNN models, including MobileNetV2 and baseline models.
- Metrics: Custom metric functions used to evaluate model performance, such as precision, recall, F1-score, and others tailored for tornado detection.
- Data Preprocessing: Scripts for preprocessing the TorNet dataset.


## Pretrained MobileNetV2 model
A pretrained MobileNetV2 model on the TorNet dataset is provided here```saved_models/mobilenet_model.keras```:

This model has been trained and evaluated on the 2021 and 2013 subsets of the TorNet dataset. 

## References and Acknowledgements

Our code is based on the following repository, and we thank the authors for their excellent contributions:

- [MIT TorNet](https://github.com/mit-ll/tornet)
