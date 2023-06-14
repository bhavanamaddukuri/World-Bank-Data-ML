import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.optimize import curve_fit
#Data Preprocessing
def pre_process_data(climate_change_data, indicator):    
    required_columns = ['country', 'year', 'co2', indicator]
    required_data_df = climate_change_data[required_columns]
    
    #check null values
    required_data_df.isnull().sum()
    
    #handle null values
    imputer= SimpleImputer(missing_values = np.nan, strategy = 'mean')
    required_data_df.iloc[:, 2:4] = imputer.fit_transform(required_data_df.iloc[:, 2:4])
    return required_data_df


#get visual insights before clustering
def visual_insights(required_data_df, year):
    shortlisted_countries_data_df = required_data_df[
        (required_data_df['country'].isin(['United States', 'United Kingdom', 
                                        'Africa', 'Antartica', 'India', 
                                        'China', 'Australia', 'Japan', 
                                        'Russia', 'Canada', 'Germany', 
                                        'Brazil', 'Argentina']) & 
         (required_data_df['co2'] > 0) & 
         (required_data_df['year'] == year))]
    shortlisted_countries_data_df.groupby(['country']).sum().plot(
        kind = 'pie', y = 'co2', legend = False, autopct='%1.0f%%')
    
    

#feature scaling (Normalisation)
def feature_scaling(required_data_df, sc, indicator):
    required_data_df.iloc[:, 2:4] = sc.fit_transform(required_data_df.iloc[:, 2:4])
    
    #consider only co2_pergdp and gdp colums for clustering
    final_columns = ['co2', indicator]
    normalised_data = required_data_df[final_columns]
    return normalised_data


#using elbow method to find the opitmal number of clusters
def elbow_method(normalised_data):
    wcss = []
    for i in range(1,11):
        #use k-means++ to avoid random initialisation trap
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kmeans.fit(normalised_data)
        wcss.append(kmeans.inertia_)
    
    #plot the graph to find n_clusters
    plt.plot(range(1,11), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()


#main method
def main():
    #Importing dataset
    climate_change_data = pd.read_csv('CO2 dataset.csv')
    
    #to know the number of columns and rows
    climate_change_data.shape
    
    #using different indicators to kmeans
    indicator = 'population' #gdp, gdp percapita
    
    #data preprocessing
    required_data_df = pre_process_data(climate_change_data, indicator)
    
    #get visual insights from dataset before clustering
    visual_insights(required_data_df, 2000)
    visual_insights(required_data_df, 2020)
    
    #store original dataframe of all columns 
    #to add dependent variable after fitting kmeans
    original_df = required_data_df.copy()

    #Store original values before normalising to plot in the end
    plot_columns = ['co2', indicator]
    plot_data = required_data_df[plot_columns]
    plot_data = np.array(plot_data)
    
    #feature scaling
    sc = StandardScaler()
    normalised_data = feature_scaling(required_data_df, sc, indicator)
    
    #elbow method
    elbow_method(normalised_data) #from graph n_clusters = 6

if __name__ == "__main__":
    main()
