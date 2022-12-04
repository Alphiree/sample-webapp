import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from streamlit_option_menu import option_menu

st.set_page_config(initial_sidebar_state="expanded",page_title='Calfironia Housing Prediction by Alphire')

with st.sidebar:
    selected = option_menu(
                menu_title='Main Menu',
                options = ['Home','Predict'],
                icons = ['house','book'],
                menu_icon='cast',
                default_index = 0,
        )

def column_ratio(X):
    return X[:, [0]] / X[:, [1]]

def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.base import BaseEstimator, TransformerMixin

# using cluster similarity, this is the better method since this uses fit and transform 

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self  # always return self!

    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)
    
    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]

def save_fig(fig_id,path, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(path, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


def get_dimension(length):
    if length == 1:
        m = 1
        n = 1
    elif length == 2:
        m = 1
        n = 2
    else:
        initial = 2
        initial_2 = 1
        while length > initial**2:
            initial = initial + 1
        
        n = initial

        while length > initial_2*n:
            initial_2 = initial_2 + 1
        
        m = initial_2

    return m,n
        

def uni_num_hist(df,num,figsize=(12,8),
                 nrows=None,ncol=None,
                 specific=None,save_path=None,
                 bins=50):
    cmap = plt.get_cmap("Dark2")
    val = 0

    if specific != None:
        fig, axes = plt.subplots(1, 1, figsize=figsize)
        sns.histplot(x=df[specific],
                    bins=bins,kde=True,color=cmap(val))
        axes.set_title(f'{str(specific)} Distribution')

        st.pyplot(fig)
        
        if save_path != None:
            name = specific+'_Distribution'
            save_fig(name,save_path)

        return None
    
    if nrows == None and ncol == None:
        nrows, ncol = get_dimension(len(num))
    
    

    fig, axes = plt.subplots(nrows, ncol, figsize=figsize)

    if nrows == 1 and ncol == 1:
        sns.histplot(x=df[num[val]],
                    bins=bins,kde=True,color=cmap(val))
        axes.set_title(f'{str(num[val])} Distribution')

        st.pyplot(fig)

        if save_path != None:
            name = 'Numerical_Distribution'
            save_fig(name,save_path)

    elif (nrows == 1 and ncol == 2):
        for i in range(nrows):
            for j in range(ncol):
                if val in range(len(num)):
                    sns.histplot(x=df[num[val]],
                        bins=bins,kde=True,color=cmap(val),ax=axes[j])
                    axes[j].set_title(f'{str(num[val])} Distribution')
                    val += 1
                    plt.tight_layout()
                else:
                    ## This is to remove the extra plots
                    fig.delaxes(axes[i,j])
        st.pyplot(fig)
        
        if save_path != None:
            name = 'Numerical_Distribution'
            save_fig(name,save_path)
    
    elif (nrows == 2 and ncol == 1):
        for i in range(nrows):
            for j in range(ncol):
                if val in range(len(num)):
                    sns.histplot(x=df[num[val]],
                        bins=bins,kde=True,color=cmap(val),ax=axes[i])
                    axes[i].set_title(f'{str(num[val])} Distribution')
                    val += 1
                    plt.tight_layout()
                else:
                    ## This is to remove the extra plots
                    fig.delaxes(axes[i,j])
        st.pyplot(fig)




    else:
        for i in range(nrows):
            for j in range(ncol):
                if val in range(len(num)):
                    sns.histplot(x=df[num[val]],ax=axes[i,j],
                                bins=bins,kde=True,color=cmap(val))
                    axes[i,j].set_title(f'{str(num[val])} Distribution')
                    val += 1
                    plt.tight_layout()
                else:
                        ## This is to remove the extra plots
                        fig.delaxes(axes[i,j])
        st.pyplot(fig)

        if save_path != None:
            name = 'Numerical_Distribution'
            save_fig(name,save_path)

## Note: we can use plt.tight_layout when plotting on subplots





















@st.cache(allow_output_mutation=True)
def model():
    final_model_reloaded = joblib.load("housing.pkl")
    return final_model_reloaded

final_model_reloaded = model()



@st.cache
def get_data():
    housing = pd.read_csv("housing.csv")
    return housing




# selected = option_menu(
#                 menu_title=None,
#                 options = ['Home','Predict'],
#                 icons = ['house','book'],
#                 menu_icon='cast',
#                 default_index = 0,
#                 orientation='horizontal'
#         )

housing = get_data()

numerical = housing.select_dtypes(exclude=['object']).columns.tolist()

if selected == 'Home':
    header = st.container()
    dataset = st.container()

    with header:
        st.title('Machine Learning Algorithms for predicting California House Prices.')

    with dataset:
        st.header('Dataset')
        st.text('This is the dataset I used to create machine learning models. \n')
        st.text(f'This dataset consists of {housing.shape[0]} rows and {housing.shape[1]} columns.')
        st.text('Here is the first 5 rows')
        housing = get_data()
        st.write(housing.head())
        corr_matrix = housing[numerical].corr()

        st.header('Exploratory Data Analysis')

        st.subheader('Numerical Features Distribution')
        st.write('This is the distribution of our numerical features')
        uni_num_hist(housing,numerical)

        st.subheader('Correlation Matrix')
        st.write('This is the correlation matrix of our numerical features')
        fig = plt.figure(figsize=(12,8))
        sns.heatmap(corr_matrix,annot=True,fmt='.2f',cmap='RdBu_r',vmin=-1, vmax=1)
        st.write(fig) 

if selected == 'Predict':
    with st.form(key='submit this'):
        long = st.number_input('Enter the longitude')
        lat = st.number_input('Enter the latitude')
        house_age = st.number_input('Enter the housing_median_age')
        total_rooms = st.number_input('Enter the total_rooms')
        total_bedrooms = st.number_input('Enter the total_bedrooms')
        population = st.number_input('Enter the population')
        households = st.number_input('Enter the households')
        median_income = st.number_input('Enter the median_income')
        ocean_proximity = st.selectbox('Enter the ocean_proximity',('<1H OCEAN','INLAND','ISLAND','NEAR BAY','NEAR OCEAN'))

        submit_button = st.form_submit_button(label='Predict')

    new_input = [long,lat,house_age , total_rooms, 
                total_bedrooms, population, households,
                median_income, ocean_proximity]

    df = pd.DataFrame([new_input])
    df.columns = ['longitude',
    'latitude',
    'housing_median_age',
    'total_rooms',
    'total_bedrooms',
    'population',
    'households',
    'median_income',
    'ocean_proximity']

    if submit_button:
        try:
            predict_value = final_model_reloaded.predict(df)
            st.write('The Predicted Price is:',predict_value[0])

            
        except:
            st.error('Enter valid values to show the results')
            #pass









