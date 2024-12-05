import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load data
df_data = pd.read_csv("breast_cancer_data.tsv", sep="\t")
df_pcr = pd.read_csv("breast_cancer_pcr.tsv", sep="\t")

# Extract features and target
X = df_data
y = df_pcr['pCR Status']

# Define preprocessing steps
# Standardize continuous variables
continuous_features = ['Age', 'ER Status',
       'PR Status', 'Ki67 25%', 'TILs 30%', 'US LN Cortex',
       'Intratumoral high SI on T2', 'Peritumoral Edema', 'Prepectoral Edema',
       'Subcutaneous Edema', 'Multifocality', 'Maximal MR Size',
       'Index Lesion MR Size', 'Size of Largest LN metastasis (mm)']
scaler = StandardScaler()

# One-hot encode categorical variables
categorical_features = ['Menopausal Status', 'T Stage', 'N Stage', 'Breast Density']
one_hot_encoder = OneHotEncoder()

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', scaler, continuous_features),
        ('cat', one_hot_encoder, categorical_features)
    ])

# Create preprocessing pipeline
preprocessing_pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

# Apply preprocessing
X_preprocessed = preprocessing_pipeline.fit_transform(X)

# Save preprocessed data
np.save('X_preprocessed.npy', X_preprocessed)
np.save('y.npy', y)
