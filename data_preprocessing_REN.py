import pandas as pd
import ast
from CustomNN_REN import CustomNN
import torch
import torch.nn as nn 
import torch.optim as optim

def load(file_name):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_name)  
    #df = pd.read_parquet(file_name) # for parquet file
    return df 

def split_dataframe_on_traffic_light_change(df):
    # Find the indices where "Traffic light" changes from 0 to 100
    change_indices = []

    # Loop through the DataFrame starting from the second row (index 1)
    traffic_light = df['traffic_light'].to_list()
    for i in range(1, len(traffic_light)):
        if traffic_light[i] - traffic_light[i-1] > 0:
            change_indices.append(i)

    # Add the start of the DataFrame as the first split point
    split_dataframes = []
    start_index = 0

    for idx in change_indices:
        # Slice the DataFrame from the start index to the current change index
        split_dataframes.append(df[start_index:idx])
        start_index = idx  # Update the start index to the current split point

    # Include the last part from the last change index to the end of the DataFrame
    split_dataframes.append(df[start_index:])

    return split_dataframes

def create_tensors_from_dataframe(df, columns_to_ignore, columns_to_convert,bin_column):
    # Keep the first row but drop the columns to ignore
    first_row = df.iloc[0].drop(columns=columns_to_ignore)  # Get the first row without the ignored columns
    first_row_tensor = torch.tensor(first_row.values[1:].astype(float))
    bin_level = torch.tensor(df[bin_column].values)
    df_filtered = df.iloc[1:]  # Drop the first row

    # Now, create tensors from the specified columns in the filtered DataFrame
    tensors = {}
    for column in columns_to_convert:
        tensors[column] = torch.tensor(df_filtered[column].values)  # Convert each specified column to a tensor
    
    return first_row_tensor, tensors, bin_level
def load_data(filename):
    df_trial = load(filename)
    #df_trial = pd.DataFrame(fittet_data, columns=df_merged.columns)
    df_model = df_trial[['Bin Level', 'traffic_light', 'Real Tons', 'Bond Work Index',
       'Deep Work Index', 'feeder 1st Motor']]
    columns_to_replace = ['Real Tons', 'Bond Work Index', 'Deep Work Index']
    df_model[columns_to_replace] = df_model[columns_to_replace].mask(df_model[columns_to_replace]==0).ffill()
    first_rows = torch.tensor(df_model[['Real Tons', 'Bond Work Index','Deep Work Index']].values, dtype = torch.float32)
    control_inputs = torch.tensor(df_model[['traffic_light','feeder 1st Motor']].values, dtype = torch.float32)
    Bin_level_input = torch.tensor(df_model['Bin Level'].values[:-1], dtype = torch.float32)
    outputs = torch.tensor(df_model['Bin Level'].values[1:], dtype = torch.float32)

    return first_rows, control_inputs, Bin_level_input, outputs

def create_train_test(tensor_list):
    split_index = 100
    #split_index = int(len(tensor_list) * 0.05)
    train = tensor_list[:split_index]  # 60%
    #test = tensor_list[split_index:int(len(tensor_list) * 0.075)] 
    test = tensor_list[split_index:110] # 40%
    return train, test

def train_and_evaluate_model(model, X_train, X_control_train, bin_level_inputs_train, y_train, X_test, X_control_test, bin_level_inputs_test, y_test, num_epochs, learning_rate=0.1, save_interval=10):
    criterion = nn.MSELoss()  # Example loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print('BEGIIIIIIIN')
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        epoch_loss = 0
        
        # Training loop
        for i in range(X_train.shape[0]-1):
            inputs_1 = X_train[i].unsqueeze(0)  # First input
            inputs_2 = X_control_train[i].unsqueeze(0)  # Second input
            inputs_3 = bin_level_inputs_train[i].unsqueeze(0)
            targets = y_train[i].unsqueeze(0)

            optimizer.zero_grad()  # Zero the parameter gradients

            outputs = model(inputs_1, inputs_2,inputs_3)
            loss = criterion(outputs.to(torch.float32), targets.to(torch.float32)).to(torch.float32)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {epoch_loss / len(X_train)}')

        # Save model weights every `save_interval` epochs
        torch.save(model.state_dict(), f'model_epoch_{epoch + 1}.pth')

        # Testing loop
        model.eval()  # Set the model to evaluation mode
        test_loss = 0
        with torch.no_grad():
            for i in range(len(X_test)):
                inputs_1 = X_test[i].unsqueeze(0)  # First input
                inputs_2 = X_control_test[i].unsqueeze(0)  # Second input
                targets = y_test[i].unsqueeze(0)
                inputs_3 = bin_level_inputs_test[i].unsqueeze(0)
                outputs = model(inputs_1, inputs_2, inputs_3)
                loss = criterion(outputs, targets)
                test_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Test Loss: {test_loss / len(X_test)}')

    print("Training and evaluation complete.")


if __name__ == "__main__": 

    first_rows, control_inputs, Bin_level_inputs, outputs = load_data('merged.csv')
    #first_rows, control_inputs, Bin_level_inputs, outputs = load_data('export_1740063344367.parquet')
    X_train, X_test = create_train_test(first_rows)
    X_control_train, X_control_test = create_train_test(control_inputs)
    y_train,y_test = create_train_test(outputs)
    Bin_level_inputs_train, Bin_level_inputs_test = create_train_test(Bin_level_inputs)
    model = CustomNN(3,2,2,10,100)
    train_and_evaluate_model(model,X_train,X_control_train,Bin_level_inputs_train, y_train,X_test,X_control_test,Bin_level_inputs_test,y_test,200)