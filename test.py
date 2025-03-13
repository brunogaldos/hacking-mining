import torch
import matplotlib.pyplot as plt
import pandas as pd
from CustomNN_REN import CustomNN


# this is for printing the plot 


def create_train_test(tensor_list):
    #split_index = 100
    split_index = int(len(tensor_list) * 0.05)
    train = tensor_list[:split_index]  # 60%
    test = tensor_list[split_index:int(len(tensor_list) * 0.1)] 
    #test = tensor_list[split_index:110] # 40%
    return train, test

def load(file_name):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_name)
    #df = pd.read_parquet(file_name)
    return df

def load_data(filename):
    df_trial = load(filename)
    df_model = df_trial[['Bin Level', 'traffic_light', 'Real Tons', 'Bond Work Index',
                         'Deep Work Index', 'feeder 1st Motor']]
    columns_to_replace = ['Real Tons', 'Bond Work Index', 'Deep Work Index']
    df_model[columns_to_replace] = df_model[columns_to_replace].mask(df_model[columns_to_replace] == 0).ffill()
    first_rows = torch.tensor(df_model[['Real Tons', 'Bond Work Index', 'Deep Work Index']].values, dtype=torch.float32)
    control_inputs = torch.tensor(df_model[['traffic_light', 'feeder 1st Motor']].values, dtype=torch.float32)
    Bin_level_input = torch.tensor(df_model['Bin Level'].values[:-1], dtype=torch.float32)
    outputs = torch.tensor(df_model['Bin Level'].values[1:], dtype=torch.float32)

    return first_rows, control_inputs, Bin_level_input, outputs

# Load data
#first_rows, control_inputs, Bin_level_inputs, outputs = load_data('export_1740063344367.parquet')
first_rows, control_inputs, Bin_level_inputs, outputs = load_data('merged.csv')
X_train, X_test = create_train_test(first_rows)
X_control_train, X_control_test = create_train_test(control_inputs)
y_train, y_test = create_train_test(outputs)
Bin_level_inputs_train, Bin_level_inputs_test = create_train_test(Bin_level_inputs)

# Load model from a specified path

model_path = 'good_models/model_epoch_1.pth'
model = CustomNN(3, 2, 2, 10, 100)
model.load_state_dict(torch.load(model_path))
model.eval()  # Set the model to evaluation mode

# Collect outputs and targets for plotting
predicted_outputs = []
actual_targets = []

for i in range(X_test.shape[0]):
    inputs_1 = X_test[i].unsqueeze(0)  # First input
    inputs_2 = X_control_test[i].unsqueeze(0)  # Second input
    inputs_3 = Bin_level_inputs_test[i].unsqueeze(0)  # Third input
    target = y_test[i].unsqueeze(0)

    # Get the model output
    output = model(inputs_1, inputs_2, inputs_3).detach().numpy()

    # Collect the output and target
    predicted_outputs.append(output[0, 0])  # Assuming output shape [1, 1]
    actual_targets.append(target.item())

# Plot the model outputs and targets
plt.plot(predicted_outputs, label='Model Outputs')
plt.plot(actual_targets, label='Targets')
plt.legend()
plt.show()
