# Projects in Data Science (2025)
> Final Assignment

#### 1. Introduction

This project implements a complete modular pipeline for lesion image analysis, from raw data loading through preprocessing, feature extraction, model training, and evaluation.

#### 2. The program:
- Loads RGB images and corresponding lesion masks (if `dataset.csv` does not exist, it will create one using `metdata.csv`)
- Applies pre-processing methods on the images
- Extracts up to six features (features A–F and exports them to `dataset.csv`)
- Trains and evaluates classifiers (based on `dataset.csv`) with cross-validation and hyperparameter tuning
- Exports detailed probability and metric files along with saved model objects

#### 3. Run the program
The code requires several libraries. Therefore, it uses a virtual environment to manage dependencies.\
The following steps will help you set up the environment and run the code.

> Run the provided code (after the '>') in a terminal or command prompt.
- Clone the repository: > git clone https://github.com/Oliverdron/2025-FYP-Orca
- Change directory to the project folder: > cd (absolute path to the cloned repository)
- Activate the virtual environment: > python -m venv venv
- Activate the virtual environment: > venv\Scripts\activate
- Install the required libraries: > pip install -r requirements.txt
- Run the scripts: > python main_baseline.py

> If updates were made to the libraries, you can export by `pip freeze > requirements.txt`.

#### 4. File Hierarchy

**The program was designed and tested to run in the following structure.**
*Modifications to the file structure may lead to errors.*

```
2025-FYP/
├── data/                        # unzip the dataset and put it here (uploading data is ignored by git)
│   ├── images/                  # all images
│   └── lesion_masks/            # all lesion masks
│ 
├── result/
│   ├── result_baseline.csv      # results on the baseline setup
│   ├── result_extended.csv      # results on the extended setup
│   └── report.pdf               # the report in PDF
│ 
├── util/
│   ├── __init__.py              # package initialization file
│   ├── inpaint.py               # image inpainting function
│   ├── classifier.py            # code for training, validating, and testing the classifier
│   ├── feature_A.py             # feature A extraction
│   ├── feature_B.py             # feature B extraction
│   ├── feature_C.py             # feature C extraction
│   ├── feature_D.py             # feature D extraction
│   ├── feature_E.py             # feature E extraction
│   ├── feature_F.py             # feature F extraction
│   ├── img_preprocess.py        # image preprocessing functions
│   └── img_util.py              # basic image read and write functions
│ 
├── dataset.csv                  # all image file names, mask names, ground-truth labels and the extracted feature values (A to F)
├── main_baseline.py             # complete script (baseline setup)
├── main_extended.py             # complete script (extended setup)
├── metadata.csv                 # all image file names, mask names and ground-truth labels
└── README.md
```

#### 5. Datasets

The `metadata.csv` file contains metadata for each image in the dataset, structured as follows:
*It is used to load the images and masks by the img_util.py to **extract the feature values***
```
patient_id: unique identifier (e.g. PAT_1516)
image_path: filename under data/images/
mask_path: filename under data/lesion_masks/ (*blank if unavailable*)
label: categorical labels (eg.: NEV, BCC, MEL, SCC, ACK)
```

The `dataset.csv` file contains metadata for each image in the dataset, structured as follows:
*It is used by the classifier.py to **train and evaluate the classifiers***
```
patient_id: unique identifier (e.g. PAT_1516)
image_fname: filename under data/images/
label_binary: ground-truth True (*indicating cancer*), False (*otherwise*) based on the categorical label
mask_fname: filename under data/lesion_masks/ (*blank if unavailable*)
feat_A to feat_F: extracted feature values
```

**Note:**
For clarification, here are the usernames and real-names of the contributors:
```
|   Username   |    Real Name     |
|:------------:|:----------------:|
|   mindenki   |    Anis Kadem    |
|  Oliverdron  |  Olivér Gyimóthy |
|   RudraKau   |   Rudra Kaushik  |
|    Etele8    |   Etele Kovács   |
| brute-yanka  |    Péter Ónadi   |
```