
---

# README

## Overview

This project was developed during the Hackmining to optimize the crusher process for a mining corporation. 

- **Model Architecture:**  

This implementation uses a Recurrent Equilibrium Network (REN) to predict new bin levels by combining historical data with current control inputs, such as traffic light status and feeder motor signals. The model updates its hidden state by merging past and present information and learns an equilibrium representation that captures the system's temporal patterns. Broyden’s method is used to solve for the hidden variable that reflects the dynamic state, and this learned memory, together with immediate control signals, is used to compute the predicted new bin level. The network is trained using Mean Squared Error (MSE) as the loss function with an Adam optimizer.


## Prerequisites

- Python 3.x
- [Pandas](https://pandas.pydata.org/)
- [PyTorch](https://pytorch.org/)



## Usage

1. Place your dataset (e.g., `merged.csv`or `export_1740063344367.parquet`)
2. Run the script with:
   ```bash
   python data_preprocessing_REN.py
   ```
3. The script will display training progress and save model checkpoints (e.g., `model_epoch_1.pth`) during training.

4. For plotting the comparison among model outputs and targets run the script:
    ```bash
   python test.py
   ```
---

