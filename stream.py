import numpy as np
import joblib
import streamlit as st

#load the trained model
model = joblib.load('model_scaled.pkl')
scaler = joblib.load('scaled.pkl')

#streamlit app title
st.title("Diabetes Prediction App")
st.write("enter your medical details to know about your diabetes status")

#define the input fields

st.sidebar.header("your medical records")

preg=st.sidebar.number_input("preg",min_value=0.0,max_value=100.0,value=50.0,step=0.1)
plas=st.sidebar.number_input("plas",min_value=0.0,max_value=100.0,value=50.0,step=0.1)
pres=st.sidebar.number_input("pres",min_value=0.0,max_value=100.0,value=50.0,step=0.1)
skin=st.sidebar.number_input("skin",min_value=0.0,max_value=100.0,value=50.0,step=0.1)
test=st.sidebar.number_input("test",min_value=0.0,max_value=100.0,value=50.0,step=0.1)
mass=st.sidebar.number_input("mass",min_value=0.0,max_value=100.0,value=50.0,step=0.1)
pedi=st.sidebar.number_input("pedi",min_value=0.0,max_value=100.0,value=50.0,step=0.1)
age=st.sidebar.number_input("age",min_value=0.0,max_value=100.0,value=50.0,step=0.1)

input_data = np.array([[preg, plas, pres, skin, test, mass, pedi, age]])
scaled_input = scaler.transform(input_data)

if st.sidebar.button("predict"):
    prediction = model.predict(scaled_input)
    st.success(f"prediction: {prediction[0]}")

    if prediction[0]==1:
        st.success("you have diabetes")
    else:
        st.success("you don't have diabetes")
