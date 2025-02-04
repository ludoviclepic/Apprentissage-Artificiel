# %%
import subprocess

# Run data preprocessing
subprocess.run(['python', 'data_preprocessing.py'])

# Run model training
subprocess.run(['python', 'model_training.py'])

# Run best models
subprocess.run(['python', 'run_best_models.py'])

# %%