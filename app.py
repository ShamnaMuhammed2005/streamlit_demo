import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,classification_report
import streamlit as st
import pickle


st.title('Diabetes Prediction')
with st.sidebar:
    options=st.radio('options',['Model','Analytics'])

if options=='Model':
    preg=st.number_input('enter value for preg',value=0, help=str,step=1,min_value=0)
    plas=st.number_input('enter value for plas',value='min', help=str,step=1,min_value=0)
    pres=st.number_input('enter value for pres', min_value=0, max_value=10, 
                        value="min", step=1, format=None, help=str, on_change="ignore", args=None, 
                        kwargs=None, placeholder=None, disabled=False, required=False, 
                        label_visibility="visible", icon=None, width="stretch", bind=None, persist_state=None)

    skin=st.number_input('enter value for skin',min_value=None, max_value=None, value="min", step=1, 
                        format=None, key=None, help=str, on_change="ignore", args=None, kwargs=None,
                          placeholder=None, disabled=False, required=False, label_visibility="visible",
                            icon=None, width="stretch", bind=None, persist_state=None)
    
    test=st.number_input('enter value for test',min_value=None, max_value=None, value="min", step=1, 
                        format=None, key=None, help=str, on_change="ignore", args=None, kwargs=None,
                          placeholder=None, disabled=False, required=False, label_visibility="visible",
                            icon=None, width="stretch", bind=None, persist_state=None)

    mass=st.number_input('enter value for mass',min_value=None, max_value=None, value="min", step=1, 
                        format=None, key=None, help=str, on_change="ignore", args=None, kwargs=None,
                          placeholder=None, disabled=False, required=False, label_visibility="visible",
                            icon=None, width="stretch", bind=None, persist_state=None)

    pedi=st.number_input('enter value for pedi',min_value=None, max_value=None, value="min", step=1, 
                        format=None, key=None, help=str, on_change="ignore", args=None, kwargs=None,
                          placeholder=None, disabled=False, required=False, label_visibility="visible",
                            icon=None, width="stretch", bind=None, persist_state=None)

    age=st.number_input('enter value for age',min_value=None, max_value=None, value="min", step=1, 
                        format=None, key=None, help=str, on_change="ignore", args=None, kwargs=None,
                          placeholder=None, disabled=False, required=False, label_visibility="visible",
                            icon=None, width="stretch", bind=None, persist_state=None)

    with open('rf_model.pkl','rb') as f:
        model=pickle.load(f)

    if st.button('predict'):
        predicted=model.predict(np.array([[preg,plas,pres,skin,test,mass,pedi,age]]))
        if predicted==np.array([1]):
            st.markdown('Diabetic')
        else:
            st.markdown('Not Diabetic')
