import os
import pandas as pd


def get_data():


    # os.path.normpath(os.getcwd() + os.sep + os.pardir)
    # define the root path of the raw data
    parent_folder = os.path.normpath(os.getcwd() + os.sep + os.pardir)
    root_path = (os.path.join(parent_folder,'raw_data'))
    # define a list of all the csv list names
    csv_list = os.listdir(root_path)

    # define a dict to save the dataframes
    dataframes = {}

    # iterate over the list of csvs and save into a dict
    # for file in csv_list:
    #     dataframes[file] = pd.read_csv(f'{root_path}/{file}')

    property_metrics_daily_df = pd.read_csv(f'{root_path}/property_metrics_daily.csv')
    # external.epc
    external_epc_df = pd.read_csv(f'{root_path}/epc.csv')

    # switchee.hub
    switchee_hub_df = pd.read_csv(f'{root_path}/hub.csv')

    # switchee.property
    switchee_prperty_df = pd.read_csv(f'{root_path}/property.csv')

    # switchee.installation
    switchee_installation_df = pd.read_csv(f'{root_path}/installation.csv')

    # switchee.sensor
    switchee_sensor_df = pd.read_csv(f'{root_path}/sensor.csv')

    # to merge all the dataframes, first make a list of the dataframes

    dataframe_list = [switchee_sensor_df, switchee_installation_df, switchee_prperty_df, switchee_hub_df, external_epc_df, property_metrics_daily_df]


    from functools import reduce

    # Define the merge operation
    def custom_merge(df1, df2, iteration):
        # Create unique suffixes based on the merge iteration
        left_suffix = f"_left_{iteration}"
        right_suffix = f"_right_{iteration}"

        merged = pd.merge(df1, df2, on='property_id', how='outer', suffixes=(left_suffix, right_suffix))

        # Check for identical columns post-merge
        for col in merged.columns:
            if left_suffix in col:
                col_base = col.split(left_suffix)[0]
                col_right = col_base + right_suffix

                # Check if both columns are present and are identical
                if col_right in merged.columns and merged[col].equals(merged[col_right]):
                    merged[col_base] = merged[col]
                    merged.drop(columns=[col, col_right], inplace=True)

        return merged

    # Iteratively merge with unique suffixes for each iteration
    result = dataframe_list[0]
    for i, df in enumerate(dataframe_list[1:], 1):
        result = custom_merge(result, df, i)


    # Dropping columns that are duplicated with suffixes due to the above merge
    cols_to_drop = ['hub_id_right_2', 'hub_id_left_4', 'hub_id_right_4', 'occupied_right_5']

    result.drop(columns=cols_to_drop, inplace=True)

    # Renaming columns to remove the suffixes and provide clearer names

    cols_rename = {
        'hub_id_left_2': 'hub_id',
        'occupied_left_5': 'occupied',

    }

    result.rename(columns=cols_rename, inplace=True)
    return result
