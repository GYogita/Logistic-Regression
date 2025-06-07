#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import joblib
import pandas as pd
import numpy as np
model=joblib.load('Titanic Survivor Prediction Model.pkl')


# In[2]:


st.title('Titanic Survival Prediction')
st.write('Provide Your Details Below:')
pclass=st.selectbox("Passenger Class (Pclass)",[1,2,3])
age=st.slider("Age",0,100,25)
sibsp=st.number_input("Siblings/Spouses Aboard (Sibsp)",0,10,0)
parch=st.number_input("Parents/Children Aboard (Parch)",0,10,0)
fare=st.number_input("Budget",0.0,600.0,30.0)
gender=st.radio("Gender",["Female","Male"])
embarked=st.selectbox("Embarked",["C","Q","S"])

gender= 1 if gender=='Male' else 0
embarked_q= 1 if embarked=='Q' else 0
embarked_s= 1 if embarked=='S' else 0
total_fam=sibsp+parch

user_input=np.array([pclass,age,sibsp,parch,fare,gender,embarked_q,embarked_s,total_fam])

if st.button('Predict Survival'):
    pred=model.predict(user_input.reshape(1, -1))[0]
    result = "Congraulations! You Survived Titanic!" if pred==1 else "RIP! You Did Not Survive Titanic"
    st.subheader(f"Prediction: {result}")


# In[ ]:




