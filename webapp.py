import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from streamlit_option_menu import option_menu

st.set_page_config(initial_sidebar_state="expanded")

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

if selected == 'Home':
    header = st.container()
    dataset = st.container()

    with header:
        st.title('alphire pogi hehe latest version')
        st.header("this is a machine learning program for predicting the price of california houses")

    with dataset:
        st.header('this is our dataset')
        st.subheader('Dataset')
        st.text('this is the dataset that I got from the book')
        housing = get_data()
        st.write(housing.head())
        corr_matrix = housing.corr()
        st.subheader('Correlation Matrix')
        st.write('This is the correlation matrix for our numerical variables')
        fig = plt.figure(figsize=(12,8))
        sns.heatmap(corr_matrix,annot=True,fmt='.2f',cmap='summer')
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









