import streamlit as st
import numpy as np
import pickle
      
experience = 0
  
st.title('Linear Regression salary prediction based on experience')

experience = st.number_input('experience')


if experience > 0:
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)

    with open('regression.pkl', 'rb') as file:
        regression = pickle.load(file)
    scaler_experience = scaler.transform([[experience]])
    st.write(f' The salary is = {regression.predict(scaler_experience)}')
    