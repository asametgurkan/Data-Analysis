import pandas as pd

# Assume current folder is project folder
data = pd.read_table("data/raw/metadata_anonymized.txt")
# data=pd.read_table("/home/gurkan/Desktop/IBD Project/data/raw/metadata_anonymized.txt")
df = data.copy()

# Some columns are not storing important information for diagnosis so they are deleted. 
columns_to_drop=["sample_id","lat_lon","env_package","patient_id","age_group","BMI","age_of_child","time_of_veganism"]
df.drop(columns_to_drop,axis=1,inplace=True)

# There were a column with "income,1<1000,2<4000,3<8000,4>8000" title, it converted to "Income".
df.rename(columns={"income,1<1000,2<4000,3<8000,4>8000": "Income"}, inplace=True)


# Function that finds and deletes columns which have missing values above 50 per cent. If this function is not run, there 
# will be huge data loss in next function. "aktif0_remisyon1" and "exposed_to_antibiotics" columns thought to be important 
# for the diagnosis. They will be processed in next cell.  
def del_missing_val(df):
    column_list=[]
    for column_name in df.columns:
        if (df[column_name].isna().sum()/df[column_name].shape[0])*100>50:
            column_list.append(column_name)
            if column_name not in ["aktif0_remisyon1","exposed_to_antibiotics"]:
                df.drop(column_name,axis=1,inplace=True)
del_missing_val(df)


# Delete rows that store missing values. In this process, df_to_clean was created, which does not contain columns that 
# must be filled. Rows that contained missing values were deleted, and indexes of rows that did not have missing values were stored.
# Indexes of this cleaned DF and two target columns (aktif0_remisyon1,exposed_to_antibiotics) were concatenated.
df_to_clean=df.drop(["aktif0_remisyon1","exposed_to_antibiotics"],axis=1)
indexes_with_missing_values = df_to_clean.index[df_to_clean.isnull().any(axis=1)]
indexes_without_missing_values = df_to_clean.index.difference(indexes_with_missing_values)
print(f"First shape of data frame: {df_to_clean.shape}")
df_to_clean.dropna(inplace=True)
print(f"Final shape of data frame: {df_to_clean.shape}")
selected_rows = df.loc[indexes_without_missing_values, ["aktif0_remisyon1", "exposed_to_antibiotics"]]
new_df=pd.concat([df_to_clean,selected_rows],axis=1)
df=new_df.copy()


# A function for classifying columns by their data types for encoding process.
def data_type_categories(df):
    column_list_binary_numeric=[]
    column_list_binary_object=[]
    column_list_non_binary=[]
    for column_name in list(df.columns):
        if df[column_name].value_counts().shape[0]==2 and (df[column_name].dtype==float or df[column_name].dtype==int):
            column_list_binary_numeric.append(column_name)
        elif df[column_name].value_counts().shape[0]==2 and df[column_name].dtype==object:
            column_list_binary_object.append(column_name)
        elif df[column_name].value_counts().shape[0]>2:
            column_list_non_binary.append(column_name)
    print(f"Binary Columns (Object): {column_list_binary_object}")
    print(f"Binary Columns (Numerical): {column_list_binary_numeric}")
    print(f"Non-Binary Columns: {column_list_non_binary}")
data_type_categories(df)


# One-Hot encoding process. The last columns in the created dummy dataframes were deleted before concatenating.
one_hot_list=["category","gender","material","sample_type"]
def one_hot_encoder(df, column_list):
        for column_name in column_list:
                one_hot_encoded = pd.get_dummies(df[column_name], prefix=column_name)
                one_hot_encoded = one_hot_encoded.astype(float)
                one_hot_encoded=one_hot_encoded.iloc[:,:-1]
                df = pd.concat([one_hot_encoded, df], axis=1)
                df.drop(column_name, axis=1, inplace=True)
        return df
df = one_hot_encoder(df, one_hot_list)


# Ordinal encoding process for "occupation" column.
active_occup=["laborer","civil_worker","soldier","nurse","sports_coach","fireman"]
semi_active_occup=["security_guard","quality_control_specialist","teacher","mechanic","serbest","control","cleaner","technician"]
passive_occup=["pensioner","student","economist","banker","assistant","freelancer","medical_doctor","lecturer","house_wife","lawyer",
               "programmer","engineer","musician","artist","advertiser","exporter","script_writer","cleric"]

print(f"Values before encoding: \n {df.occupation.unique()}")
for job_name in active_occup:
    df.loc[df["occupation"].str.contains(job_name,case=False),"occupation"]="Active"
for job_name in semi_active_occup:
    df.loc[df["occupation"].str.contains(job_name,case=False),"occupation"]="Semi-Active"
for job_name in passive_occup:
    df.loc[df["occupation"].str.contains(job_name,case=False),"occupation"]="Passive"
print(f"Values after categorical encoding: \n{df.occupation.unique()}")

def ordinal_encoder(dataframe,column_name):
    from sklearn.preprocessing import OrdinalEncoder
    data = dataframe[column_name].values.reshape(-1, 1)
    ordinal_encoder = OrdinalEncoder(categories=[['Passive', 'Semi-Active', 'Active']])
    encoded_data = ordinal_encoder.fit_transform(data)
    dataframe[f"{column_name}_enc"] = encoded_data
    dataframe.drop(column_name,axis=1,inplace=True)
ordinal_encoder(df,"occupation")
print(f"Values after ordinal encoding: \n{df.occupation_enc.unique()}")


# Ordinal encoding process for "BMI_type" column.
def ordinal_encoder(dataframe,column_name):
    from sklearn.preprocessing import OrdinalEncoder
    data = dataframe[column_name].values.reshape(-1, 1)
    ordinal_encoder = OrdinalEncoder(categories=[['underweight', 'normal', 'overweight', 'severe obese','obese']])
    encoded_data = ordinal_encoder.fit_transform(data)
    dataframe[f"{column_name}_enc"] = encoded_data
    dataframe.drop(column_name,axis=1,inplace=True)
ordinal_encoder(df,"BMI_type")


# Scaling process for columns that store continous values.
from sklearn.preprocessing import StandardScaler, MinMaxScaler
continuous_cols=["weight","height","age"]
scaler = StandardScaler()
df[continuous_cols] = scaler.fit_transform(df[continuous_cols])
scaler = MinMaxScaler()
df[continuous_cols] = scaler.fit_transform(df[continuous_cols])
def ordinal_encoder(dataframe,column_name):
    from sklearn.preprocessing import OrdinalEncoder
    data = dataframe[column_name].values.reshape(-1, 1)
    ordinal_encoder = OrdinalEncoder(categories=[['0', 'UC', 'CD']])
    encoded_data = ordinal_encoder.fit_transform(data)
    dataframe[f"{column_name}_enc"] = encoded_data
    dataframe.drop(column_name,axis=1,inplace=True)
ordinal_encoder(df,"diagnosis")


###   Finding NaN values for two columns : exposed_to_antibiotics,aktif0_remisyon1
# aktif0_remisyon1
df_active=df.drop("exposed_to_antibiotics",axis=1)

indexes_with_missing_values = df_active.index[df_active.isnull().any(axis=1)]
indexes_without_missing_values = df_active.index.difference(indexes_with_missing_values)

train_df=df_active.loc[indexes_without_missing_values]
test_df=df_active.loc[indexes_with_missing_values]

X_train=train_df.drop("aktif0_remisyon1",axis=1)
y_train=train_df["aktif0_remisyon1"]
X_test=test_df.drop("aktif0_remisyon1",axis=1)

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
predictions=knn.fit(X_train,y_train).predict(X_test)
pred_series=pd.Series(predictions)

missing_indices = df_active[df_active['aktif0_remisyon1'].isnull()].index
imputed_values = {idx: pred for idx, pred in zip(missing_indices, predictions)}

# Fill missing values in "aktif0_remisyon1" column with 'predictions'
df_active["aktif0_remisyon1"].fillna(imputed_values, inplace=True)
new_active_col=df_active["aktif0_remisyon1"]



# exposed_to_antibiotics
df_antibiotics=df.drop("aktif0_remisyon1",axis=1)

indexes_with_missing_values = df_antibiotics.index[df_antibiotics.isnull().any(axis=1)]
indexes_without_missing_values = df_antibiotics.index.difference(indexes_with_missing_values)

train_df=df_antibiotics.loc[indexes_without_missing_values]
test_df=df_antibiotics.loc[indexes_with_missing_values]

X_train=train_df.drop("exposed_to_antibiotics",axis=1)
y_train=train_df["exposed_to_antibiotics"]
X_test=test_df.drop("exposed_to_antibiotics",axis=1)

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
predictions=knn.fit(X_train,y_train).predict(X_test)
pred_series=pd.Series(predictions)

missing_indices = df_antibiotics[df_antibiotics['exposed_to_antibiotics'].isnull()].index
imputed_values = {idx: pred for idx, pred in zip(missing_indices, predictions)}

# Fill missing values in "exposed_to_antibiotics" column with 'predictions'
df_antibiotics["exposed_to_antibiotics"].fillna(imputed_values, inplace=True)
new_antibiotics_col=df_antibiotics["exposed_to_antibiotics"]

filled_columns=pd.concat([new_active_col,new_antibiotics_col],axis=1)
df=df.drop(["exposed_to_antibiotics","aktif0_remisyon1"],axis=1)
new_df=pd.concat([filled_columns,df],axis=1)



# save file
df.to_csv("preprocessed_data.csv",index=False)