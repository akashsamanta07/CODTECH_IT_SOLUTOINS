import pandas as pd
import pandas as pd
import numpy as np
import joblib as jb
import streamlit as st
st.header("House prices Prediction ML Model")
st.caption("Welcome to my App")
dataset=pd.read_csv("dataset.csv")
st.divider()
year = st.slider("Select Build Year", 2015,1970)
land = st.slider("Select Total Area of the Land", 800,50000)
area = st.slider("Select Total Area of the House", 500,6000)
floor = st.slider("Select No of floors", 4,1)
bedroom = st.slider("Select No of Bedrooms", 30,2)
bathroom = st.slider("Select No of Bathrooms", 10,1)
school = st.slider("Select No of schools nearby House", 3,1)
airport = st.slider("Select the House Distance from the airport", 100,40)
rating = st.slider("Enter Rating of the House",10,4)
Button=st.button("Predict the prices!")
st.divider()
model=jb.load("zipmodel.pkl")
df = pd.DataFrame([[bedroom,bathroom,area,land,floor,rating,year,school,airport]],columns=['number of bedrooms', 'number of bathrooms', 'living_area_renov',
       'lot_area_renov', 'number of floors', 'grade of the house','Built Year', 'Number of schools nearby', 'Distance from the airport'])
output = model.predict(df)
if Button:
    st.markdown("PREDICTING AGGREGATE RATINGS:" + str(output))
    