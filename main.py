import streamlit as st
import numpy as np
import os
import pickle
import warnings
warnings.filterwarnings('ignore')
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/heart_disease_model.sav"
with open(model_path,'rb') as file:
  model = pickle.load(file)
st.title("Heart Disease Prediction App") 
class_names = ['positive','Negative']
col1, col2, col3 = st.columns([1,1,1])
with col1:
  age = st.number_input("Enter Age")
  trestbps = st.number_input("Enter trestbps")
  restecg = st.number_input("Enter restecg")
  ca = st.number_input("Enter ca")
  thal = st.number_input("Enter thal")
with col2:
  sex = st.number_input("Enter sex",step=1,min_value=0,max_value=1)
  chol = st.number_input("Enter chol")
  thalach = st.number_input("Enter thalach")
  slope = st.number_input("Enter slope")
  
with col3:
  cp = st.number_input("Enter cp")
  fbs = st.number_input("Enter fbs")
  exang = st.number_input("Enter exang")
  oldpeak = st.number_input("Enter oldpeak")


#age	sex	cp	trestbps	chol	fbs	restecg	thalach	exang	oldpeak	slope	ca	thal
if st.button('Classify'):
  #input_data = (age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal)
  input_data = (49,1,2,118,149,0,0,126,0,0.8,2,3,2)
  #change this tuple into numpy array
  input_data_numpy_array = np.asarray(input_data)
  #change the numpy array to single dimensional array
  input_data_reshape = input_data_numpy_array.reshape(1,-1)
  predict = model.predict(input_data_reshape)
  print(predict)
  if predict == 0:
    st.success('Heart disease:NEGATIVE')
  else:
    st.success('Heart disease:POSITIVE')

    
 
    

  
  



  

  


 











  


