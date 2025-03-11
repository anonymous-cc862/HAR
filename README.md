
# Human Activity Recognition with Diffusion Models

This repository contains the code for our Human Activity Recognition (HAR) project.

## Project Structure

- `diffusions/`: Contains scripts for the diffusion models.
- `networks/`: Contains different neural network architecture scripts.
- `results/`: Contains TensorBoard logs and other results outputs.
- `val_res/`: Validation results and data.
- `main.py`: The main script to run the experiments.
- `protonet.py`: Contains the implementation of the Protonet model.
- `tsne.py`: Script for T-SNE visualization of high-dimensional data.

## Data

The dataset used in this project is hosted on Google Drive. Download it from the following link and add it to the main project folder before running the scripts:
[Download Dataset](https://drive.google.com/drive/folders/1swkdEPGvxVEiahi_AYLVbnIWgEAHYSMF?usp=sharing)

## Requirements

Before running the project, install the required Python packages:
```
pip install -r requirements.txt
```

## Running the Code

To run the main project script with the necessary environment variable settings for optimal performance, use the following command:
```
XLA_PYTHON_CLIENT_PREALLOCATE=false python main.py --data human_activity
```

This command sets the `XLA_PYTHON_CLIENT_PREALLOCATE` environment variable to `false`, which can help in managing memory more efficiently when using hardware accelerators like TPUs.



This project is released under the MIT License. See the `LICENSE` file for more details.

---

This README provides an overview of the project, instructions for setting up the environment, obtaining the data, and running the code. It also points to the structure of the project directories for better navigation and understanding of the repository.
