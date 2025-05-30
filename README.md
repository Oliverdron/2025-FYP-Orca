# Projects in Data Science (2025)
> Final Assignment

#### 1. Introduction

This project implements a complete modular pipeline for lesion image analysis, from raw data loading through preprocessing, feature extraction, model training, and evaluation.

#### 2. The program:
- **Loads RGB images** and corresponding lesion masks (if `dataset.csv` does not exist, it will create one using `metdata.csv`)
- Applies **pre-processing** methods on the images
- Extracts up to six **features** (A to F and exports them to `dataset.csv`)
- Trains and **evaluates classifiers** (based on `dataset.csv`) with cross-validation and hyperparameter tuning
- Exports detailed probability and metric files along with saved model objects

#### 3. Run the program
The code requires several libraries. Therefore, it uses a virtual environment to manage dependencies.\
The following steps will help you set up the environment and run the code.

> Run the provided code (after the '>') in a terminal or command prompt.
- Clone the repository: > `git clone https://github.com/Oliverdron/2025-FYP-Orca`
- Change directory to the project folder: > `cd (absolute path to the cloned repository)`
- Activate the virtual environment: > `python -m venv venv`
- Activate the virtual environment: > `venv\Scripts\activate`
- Install the required libraries: > `pip install -r requirements.txt`
- Run the scripts: > `python main_baseline.py`
- Alternatively, you can use **environment.yml** with conda

> If updates were made to the libraries, you can export by `pip freeze > requirements.txt`.

#### 4. Test outer source using loaded trained models
If you want to test already trained models on your dataset, you can do that using the LoadClassifier
class from classifier.py.
The following steps should be followed.

- Remove any models that do not correspond to the name of the main file(extended/baseline) from result/models
- Delete the `dataset.csv` from base directory
- In one of the main files (eg. `main_baseline.py`), delete everything after **clf=TrainClassifier** line until the end of main
- Initialize a `LoadClassifier` object with:
    `model_path(base/"result"/"models"),`
    `output_path,base_dir(base),`
    `feature_names(list(FEATURES.keys()))` parameters
- Load dataset using `load_dataset(source="dataset.csv")`
- Put in the following line:
    `clf.save_result_and_probabilities(`
    `    *clf.evaluate_classifiers(clf.X_test, clf.y_test),`
    `    *clf.evaluate_classifiers(clf.X, clf.y),`
    `    type="baseline",`
    `    save_visible=True`
    `)`
- After this, the results and the probabilities should be made in the result folder for each models.
- You can create plots using: `clf.visualize(clf.X, clf.y, "name")`

#### 5. File Hierarchy

**The program was designed and tested to run in the following structure.**
*Modifications to the file structure may lead to errors.*

```
2025-FYP/
├── data/                           # unzip the dataset and put it here (uploading data is ignored by git)
│   ├── images/                     # all images
│   └── lesion_masks/               # all lesion masks
│ 
├── result/
│   ├── models/                     # trained models of the baseline and extended setup(.pkl files)
│   ├── probabilities_baseline.csv  # probabilities of the baseline setup
│   ├── probabilities_extended.csv  # probabilities of the extended setup
│   ├── results_baseline.json       # results of the baseline setup
│   ├── results_extended.json       # results of the extended setup
│   └── report.pdf                  # the report in PDF
│ 
├── util/
│   ├── __init__.py                 # package initialization file
│   ├── inpaint.py                  # image inpainting function
│   ├── classifier.py               # code for training, validating, and testing the classifier
│   ├── feature_A.py                # feature A extraction
│   ├── feature_B.py                # feature B extraction
│   ├── feature_C.py                # feature C extraction
│   ├── feature_D.py                # feature D extraction
│   ├── feature_E.py                # feature E extraction
│   ├── feature_F.py                # feature F extraction
│   ├── img_preprocess.py           # image preprocessing functions
│   └── img_util.py                 # basic image read and write functions
│ 
├── dataset.csv                     # all image file names, mask names, ground-truth labels and the extracted feature values
├── environment.yml                 # required libraries needed for conda installation
├── main_baseline.py                # complete script (baseline setup)
├── main_extended.py                # complete script (extended setup)
├── metadata.csv                    # all image file names, mask names and ground-truth labels
├── README.md
└── requirements.txt                # required libraries needed for pip installation
```

#### 6. Datasets

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