# Naive Bayesian Classifier

Breast cancer classification using Naive Bayesian algorithm on Wisconsin Breast Cancer Database.

## Dataset

- **Source**: University of Wisconsin Hospitals
- **Records**: 699 total (400 train, 299 test)
- **Features**: 9 attributes (1-10 scale)
  - Clump Thickness
  - Uniformity of Cell Size
  - Uniformity of Cell Shape
  - Marginal Adhesion
  - Single Epithelial Cell Size
  - Bare Nuclei
  - Bland Chromatin
  - Normal Nucleoli
  - Mitoses
- **Classes**: 2 (benign, malignant)

## Build

```bash
g++ -o bayesian.exe main.cc bayesian.cc naivebayesian.cc machinelearning.cc
```

## Usage

```bash
./bayesian.exe <train_data> <test_data> <config> [method]
```

**Example:**
```bash
./bayesian.exe breast_cancer_data/breast-cancer-wisconsin-400-records-train breast_cancer_data/breast-cancer-wisconsin-299-records-test breast_cancer_data/breast-cancer-wisconsin.cfg 0
```

**Parameters:**
- `train_data` - path to training dataset
- `test_data` - path to test dataset
- `config` - path to configuration file
- `method` - 0 for Naive Bayesian (default)

## Structure

```
machinelearning/
├── MachineLearning      - abstract base class
└── baysian/
    ├── Bayesian         - bayesian base class
    └── NaiveBayesian    - naive bayesian implementation
```

## Configuration Format

```
<train_records> <test_records> <num_attributes>
<min_values...>
<max_values...> <num_classes>
```

## Output

Prediction results and execution time in seconds.
