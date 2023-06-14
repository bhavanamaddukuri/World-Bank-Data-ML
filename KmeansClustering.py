import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer



#Importing dataset
climate_change_data = pd.read_csv('CO2 dataset.csv')
#to know the number of columns and rows
climate_change_data.shape

#Data Preprocessing
required_columns = ['country', 'year', 'co2', 'population']
data_df = climate_change_data[required_columns]

#check null values
data_df.isnull().sum()

#handle null values
imputer= SimpleImputer(missing_values = np.nan, strategy = 'mean')
data_df.iloc[:, 2:4] = imputer.fit_transform(data_df.iloc[:, 2:4])