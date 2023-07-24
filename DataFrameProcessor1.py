class DataFrameProcessor1:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    # Most of the columns contain uncertain values like "unspecified" or "Not provided". Converting them to NaN values and removing them from df will prevent leakage. 
    # This function converts whole uncertain vlaues. 

    def replace_values_non_provided(self):
        column_list = []
        for column_name in self.dataframe.columns:
            column_list.append(column_name)
            for column_name in column_list:
                self.dataframe.loc[self.dataframe[column_name].str.contains("unspecified", case=False, na=False), column_name] = np.nan
                self.dataframe.loc[self.dataframe[column_name].str.contains("Not provided", case=False, na=False), column_name] = np.nan
                self.dataframe.loc[self.dataframe[column_name].str.contains("Not sure", case=False, na=False), column_name] = np.nan

    # Some columns may contain NaN values and uncertain ones were converted to NaN with the function above.
    # This function finds columns which have nan values, gives the number of missing values and deletes them. 

    def find_nan(self):
        column_list = [i for i in self.dataframe.columns]
        print(f"Old shape of dataframe: {self.dataframe.shape}")
        for i in column_list:
            if self.dataframe[i].isna().sum() != 0:
                print(f"Columns which have nan values: {i} \nThere were {self.dataframe[i].isna().sum()} number of missing values and they are deleted.")
                self.dataframe.drop(self.dataframe[self.dataframe[i].isna()].index, axis=0, inplace=True)
        print(f"New shape of dataframe: {self.dataframe.shape}")


    # Huge ratio of metadata belongs to columns have ordinal-categorical values like "Daily" or "Rarely". But they were written with different shape like  "daily" and "Daily". 
    # This function collect the values those share same meaning into one value. 


    def replace_values_1(self):
        column_list = []
        for column_name in self.dataframe.columns:
            column_list.append(column_name)
            for i in list(self.dataframe[column_name].value_counts().keys()):
                for column_name in column_list:
                    self.dataframe.loc[self.dataframe[column_name].str.contains("Daily", case=False), column_name] = "Daily"
                    self.dataframe.loc[self.dataframe[column_name].str.contains("Regularly", case=False), column_name] = "Regularly"
                    self.dataframe.loc[self.dataframe[column_name].str.contains("Occasionally", case=False), column_name] = "Occasionally"
                    self.dataframe.loc[self.dataframe[column_name].str.contains("Rarely", case=False), column_name] = "Rarely"
                    self.dataframe.loc[self.dataframe[column_name].str.contains("Never", case=False), column_name] = "Never"


    # There are many variable type in df, you can check the columns have not processed, so you can handle with them one-by-one. 


    def checker(self):
        column_list = [i for i in self.dataframe.columns]
        columns_list_new = []
        for i in column_list:
            key_list = ['Daily', 'Occasionally', 'Regularly', 'Rarely', 'Never']
            unique_values = self.dataframe[i].value_counts().index
            all_present = all(value in key_list for value in unique_values)
            if not all_present:
                columns_list_new.append(i)

        print(f"There are {len(columns_list_new)} number of columns that should be checked. They are:\n{columns_list_new}")

   
