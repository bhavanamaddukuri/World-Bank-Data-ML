import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


#Data Preprocessing
def pre_process_data(climate_change_data):    
    required_columns = ['country', 'year', 'co2', 'population']
    required_data_df = climate_change_data[required_columns]
    
    #check null values
    required_data_df.isnull().sum()
    
    #handle null values
    imputer= SimpleImputer(missing_values = np.nan, strategy = 'mean')
    required_data_df.iloc[:, 2:4] = imputer.fit_transform(required_data_df.iloc[:, 2:4])
    return required_data_df


#feature scaling (Normalisation)
def feature_scaling(required_data_df, sc):
    required_data_df.iloc[:, 2:4] = sc.fit_transform(required_data_df.iloc[:, 2:4])
    
    #consider only co2_pergdp and gdp colums for clustering
    final_columns = ['co2', 'population']
    normalised_data = required_data_df[final_columns]
    return normalised_data


#main function
def main():
    #Importing dataset
    climate_change_data = pd.read_csv('CO2 dataset.csv')
    
    #to know the number of columns and rows
    climate_change_data.shape
    
    #data preprocessing
    required_data_df = pre_process_data(climate_change_data)
    
    #store original dataframe of all columns 
    #to add dependent variable after fitting kmeans
    original_df = required_data_df.copy()

    #Store original values before normalising to plot in the end
    plot_columns = ['co2', 'population']
    plot_data = required_data_df[plot_columns]
    plot_data = np.array(plot_data)
    
    #feature scaling
    sc = StandardScaler()
    normalised_data = feature_scaling(required_data_df, sc)

if __name__ == "__main__":
    main()


