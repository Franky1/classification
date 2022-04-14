#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import pandas as pd
import numpy as np
import sklearn
import joblib
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import requests
import pickle
from io import BytesIO

# In[3]:


#---------------------------------#
# Page layoutF
## Page expands to full width
st.set_page_config(page_title='Modelo predictivo de defaults',
    layout='wide')

# In[12]:



#---------------------------------#
# Model building
def build_model(score_data):
        
        #turn amounts from COP (Colombia), USD (Panama), Guatemala into USD
    def currency(country_id, amount):

        if country_id==2:
            amount=amount*.13
        elif country_id==3:
            amount=amount*0.00027
        else:
            amount=amount
        return amount
    
    score_data['salary'] = score_data.apply(lambda x: currency(x['country_id'], x['salary']), axis=1)
    
        ########## creating dummy variables######
    #### ['gender', 'marital_status','profession_type','country_id']####
    score_data["gender_Female"]= np.where(score_data["gender"]=='Female',1,0)

    score_data['marital_status_single']= np.where(score_data["marital_status"]=='single',1,0)
    score_data['marital_status_married']= np.where(score_data["marital_status"]=='married',1,0)
    score_data['marital_status_divorced']= np.where(score_data["marital_status"]=='divorced',1,0)
    score_data['marital_status_widower']= np.where(score_data["marital_status"]=='widower',1,0)
    score_data['marital_status_unknown']= np.where(score_data["marital_status"].isna(),1,0)

    score_data['profession_type_independant']= np.where(score_data["profession_type"]=='independant',1,0)
    score_data['profession_type_employed']= np.where(score_data["profession_type"]=='employed',1,0)
    score_data['profession_type_unemployed']= np.where(score_data["profession_type"]=='unemployed',1,0)
    score_data['profession_type_unknown']= np.where(score_data["profession_type"].isna(),1,0)

    score_data['country_id_1']= np.where(score_data["country_id"]==1,1,0)
    score_data['country_id_2']= np.where(score_data["country_id"]==2,1,0)
    score_data['country_id_3']= np.where(score_data["country_id"]==3,1,0)
    
    #standardize credit score by country####################
    def credit_score_scale(country_id, CS):
        if country_id==1:
            CS_scaled=(CS-0)/(900-0)
        elif country_id==2:
            CS_scaled=(CS-(-99))/(1000-(-99))
        else:
            CS_scaled=(CS-0)/(950-0)
        return CS_scaled
    
    score_data['credit_score_scaled'] = score_data.apply(lambda x: credit_score_scale(x['country_id'], x['credit_score']), axis=1)
    
    ####### Separar datos X (variables independientes, se utilizan como input). y
    X_test = score_data[['age', 'salary',
                            'gender_Female', 'marital_status_divorced',
                        'marital_status_married', 'marital_status_single',
                        'marital_status_unknown', 'marital_status_widower',
                        'profession_type_employed', 'profession_type_independant',
                        'profession_type_unemployed', 'profession_type_unknown', 'country_id_1',
                        'country_id_2', 'country_id_3', 'credit_score_scaled']].values
    
    # Load the model from the file
    
    mLink = 'https://github.com/marcebejarano/classification/blob/main/default_class_model.pkl'
    mfile = BytesIO(requests.get(mLink).content)
    model_from_joblib = joblib.load(mfile)

    # Use the loaded model to make predictions
    
    scoring = model_from_joblib.predict(X_test)
    
    probability_scoring = pd.DataFrame(scoring).set_axis(['prob_score'], axis=1)
    probability_scoring["default_pred"]=np.where(probability_scoring["prob_score"]>.2,1,0)
    #merge data on customer scores
    scored_data = pd.merge(score_data[['customer_id']], probability_scoring,left_index=True, right_index=True)
    
    return scored_data
    
    
    
#     X = df.iloc[:,:-1] # Using all column except for the last column as X
#     Y = df.iloc[:,-1] # Selecting the last column as Y

#     # Data splitting
#     X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=(100-split_size)/100)
    
#     st.markdown('**1.2. Data splits**')
#     st.write('Training set')
#     st.info(X_train.shape)
#     st.write('Test set')
#     st.info(X_test.shape)

#     st.markdown('**1.3. Variable details**:')
#     st.write('X variable')
#     st.info(list(X.columns))
#     st.write('Y variable')
#     st.info(Y.name)

#     rf = RandomForestRegressor(n_estimators=parameter_n_estimators,
#         random_state=parameter_random_state,
#         max_features=parameter_max_features,
#         criterion=parameter_criterion,
#         min_samples_split=parameter_min_samples_split,
#         min_samples_leaf=parameter_min_samples_leaf,
#         bootstrap=parameter_bootstrap,
#         oob_score=parameter_oob_score,
#         n_jobs=parameter_n_jobs)
#     rf.fit(X_train, Y_train)

#     st.subheader('2. Model Performance')

#     st.markdown('**2.1. Training set**')
#     Y_pred_train = rf.predict(X_train)
#     st.write('Coefficient of determination ($R^2$):')
#     st.info( r2_score(Y_train, Y_pred_train) )

#     st.write('Error (MSE or MAE):')
#     st.info( mean_squared_error(Y_train, Y_pred_train) )

#     st.markdown('**2.2. Test set**')
#     Y_pred_test = rf.predict(X_test)
#     st.write('Coefficient of determination ($R^2$):')
#     st.info( r2_score(Y_test, Y_pred_test) )

#     st.write('Error (MSE or MAE):')
#     st.info( mean_squared_error(Y_test, Y_pred_test) )

#     st.subheader('3. Model Parameters')
#     st.write(rf.get_params())

#---------------------------------#


# In[5]:


#---------------------------------#
st.write("""
# Modelo predictivo de clasificación de default

En esta aplicación se debe subir un CSV con las variables:

customer_id, gender, age, marital_status, profession_type, salary, country_id, credit_score

Advertencia: las variables se deben llamar de esta misma forma y los campos de "age" y "salario" no pueden ser nulos o datos invalidos (e.g. 200 años)

El output tendrá la siguientes variables:

customer_id

prob_score: Probabilidad de hacer un default

default_pred: 1 si se predice un default, 0 si no


""")

#---------------------------------#
# Sidebar - Collects user input features into dataframe
with st.sidebar.header('1. Subir datos de clientes en CSV'):
    uploaded_file = st.sidebar.file_uploader("Subir CSV", type=["csv"])


# In[ ]:



# #---------------------------------#
# # Sidebar - Collects user input features into dataframe
# with st.sidebar.header('1. Subir datos de clientes en CSV'):
#     uploaded_file = st.sidebar.file_uploader("Subir CSV", type=["csv"])

# # Sidebar - Specify parameter settings
# with st.sidebar.header('2. Set Parameters'):
#     split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)

# with st.sidebar.subheader('2.1. Learning Parameters'):
#     parameter_n_estimators = st.sidebar.slider('Number of estimators (n_estimators)', 0, 1000, 100, 100)
#     parameter_max_features = st.sidebar.select_slider('Max features (max_features)', options=['auto', 'sqrt', 'log2'])
#     parameter_min_samples_split = st.sidebar.slider('Minimum number of samples required to split an internal node (min_samples_split)', 1, 10, 2, 1)
#     parameter_min_samples_leaf = st.sidebar.slider('Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2, 1)

# with st.sidebar.subheader('2.2. General Parameters'):
#     parameter_random_state = st.sidebar.slider('Seed number (random_state)', 0, 1000, 42, 1)
#     parameter_criterion = st.sidebar.select_slider('Performance measure (criterion)', options=['mse', 'mae'])
#     parameter_bootstrap = st.sidebar.select_slider('Bootstrap samples when building trees (bootstrap)', options=[True, False])
#     parameter_oob_score = st.sidebar.select_slider('Whether to use out-of-bag samples to estimate the R^2 on unseen data (oob_score)', options=[False, True])
#     parameter_n_jobs = st.sidebar.select_slider('Number of jobs to run in parallel (n_jobs)', options=[1, -1])


# In[6]:


#---------------------------------#
# Main panel

# Displays the dataset
st.subheader('1. Dataset')

if uploaded_file is not None:
    df= pd.read_csv(uploaded_file)
    score_model = build_model(df)
    
    st.markdown('**1.1. Glimpse of the results**')
    st.write(score_model)

else:
    st.info('A la espera de un CSV')


# In[ ]:




