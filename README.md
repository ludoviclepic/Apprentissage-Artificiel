# Breast Cancer Analysis Project

This project involves preprocessing breast cancer data, training machine learning models, and evaluating the best models. The project is organized into three main Python scripts:

1. **data_preprocessing.py**: This script loads the raw data, preprocesses it using a pipeline, and saves the preprocessed data as NumPy arrays.

2. **model_training.py**: This script loads the preprocessed data, splits it into training and test sets, and performs grid search to train and evaluate two models: Random Forest and SVM.

3. **run_best_models.py**: This script loads the preprocessed data, splits it into training and test sets, and trains the models using the best hyperparameters. It then evaluates the models and prints the results.

## Running the Project

To run the entire project, execute the `run_all.py` script. This script will sequentially run the three main scripts in the correct order.

```bash
python run_all.py
```

Ensure that you have all the necessary dependencies installed before running the script. You can install the required packages using:

```bash
pip install -r requirements.txt
```

## Dependencies

- numpy
- pandas
- scikit-learn

## Data Files

- `breast_cancer_data.tsv`: Contains the raw breast cancer data.
- `breast_cancer_pcr.tsv`: Contains the pCR status data.

## Output Files

- `X_preprocessed.npy`: Contains the preprocessed feature data.
- `y.npy`: Contains the target data.
