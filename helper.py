import pandas as pd

# Step 1: Import the Excel file into a pandas DataFrame
dispatcher_path1 = 'Final_data_dispatch/Dispatch_260924-251024.xlsx'  # Replace with your file path
dispatcher_path2 = 'Final_data_dispatch/Dispatch_261024-251124.xlsx'
dispatcher_path3 = 'Final_data_dispatch/Dispatch_261124-251224.xlsx'

dispacher_data = [dispatcher_path1, dispatcher_path2, dispatcher_path3]

crusher_path1 = "Final_data_crusher_every_second\September.xlsx"
crusher_path2 = "Final_data_crusher_every_second\October part 1.xlsx"
crusher_path3 = "Final_data_crusher_every_second\October part 2.xlsx"
crusher_path4 = "Final_data_crusher_every_second\October part 3.xlsx"
crusher_path5 = "Final_data_crusher_every_second\October part 1.xlsx"
crusher_path6 = "Final_data_crusher_every_second\October part 2.xlsx"
crusher_path7 = "Final_data_crusher_every_second\October part 3.xlsx"
crusher_path8 = "Final_data_crusher_every_second\December part 1.xlsx"
crusher_path9 = "Final_data_crusher_every_second\December part 2.xlsx"
crusher_path10 = "Final_data_crusher_every_second\December part 3.xlsx"

crusher_data = [crusher_path1 ,
                crusher_path2 ,
                crusher_path3,
                crusher_path4 ,
                crusher_path5 ,
                crusher_path6 ,
                crusher_path7 ,
                crusher_path8 ,
                crusher_path9 ,
                crusher_path10,
                ]
def clean_data_step1(df, data_source):
    if data_source == "crusher":
        pass
    raise NotImplementedError

def load_data_from_directory(data_source,):
    # data_source : "crusher" or "dispatcher"
    assert data_source in ["crusher", "dispatcher"]
    all_files = dispacher_data if "dispatcher" == data_source else crusher_data
    df_list = [pd.read_excel(file) for file in all_files]
    df_list = clean_data_step1(df_list, data_source)
    combined_df = pd.concat(df_list, axis=0, ignore_index=True)
    return combined_df


def plot_time_series(df, columns, plot_type='line', **kwargs):
    """
    Plot multiple time series attributes of a DataFrame one after the other.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the time series data.
    columns (list): List of column names to plot.
    plot_type (str): Type of plot to generate ('line', 'scatter').
    **kwargs: Additional keyword arguments to pass to the plotting functions.
    """
    #
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        raise ValueError("The DataFrame index must be a datetime type for time series plotting.")

    for column in columns:
        plt.figure(figsize=kwargs.get('figsize', (10, 6)))

        if plot_type == 'line':
            df[column].plot.line()
        elif plot_type == 'scatter':
            if 'y' in kwargs:
                plt.scatter(df.index, df[column])
                plt.xlabel('Time')
                plt.ylabel(column)
            else:
                raise ValueError("For scatter plot, 'y' value should be provided in kwargs.")
        else:
            raise ValueError(f"Unsupported plot_type for time series: {plot_type}")

        plt.title(f"{plot_type.capitalize()} Plot of {column} over Time")
        plt.xlabel('Time')
        plt.ylabel(column)
        plt.show()


if __name__=="__main__":
    df = load_data_from_directory("crusher")
    print(df[3])