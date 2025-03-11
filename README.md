
# Human Activity Recognition with Diffusion Models

This repository contains the code for our Human Activity Recognition (HAR) project, which leverages advanced neural network architectures to analyze and classify types of human activities based on sensor data.

## Project Structure

- `diffusions/`: Contains scripts for the diffusion models.
- `networks/`: Contains different neural network architecture scripts.
- `results/`: Contains TensorBoard logs and other results outputs.
- `val_res/`: Validation results and data.
- `main.py`: The main script to run the experiments.
- `protonet.py`: Contains the implementation of the Protonet model.
- `tsne.py`: Script for T-SNE visualization of high-dimensional data.

## Data

The dataset required for this project is stored on Google Drive. Please download the dataset from the following link:

[Download Dataset](https://drive.google.com/drive/folders/1swkdEPGvxVEiahi_AYLVbnIWgEAHYSMF?usp=sharing)

### Setting Up Data

After downloading, create a folder named `data` in the main project directory. Place the `human_activity` folder, which you downloaded, inside the `data` folder. This step is crucial for the scripts to locate and use the dataset correctly.

## Requirements

Before running the project, install the required Python packages:
```
pip install -r requirements.txt
```

## Running the Code

To run the main project script, execute the following command:
```
XLA_PYTHON_CLIENT_PREALLOCATE=false python main.py --data human_activity
```

This command configures the `XLA_PYTHON_CLIENT_PREALLOCATE` environment variable to `false`, which can be beneficial for managing memory more efficiently when using hardware accelerators like TPUs.

