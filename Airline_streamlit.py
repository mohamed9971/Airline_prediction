import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost
from xgboost import XGBRegressor



Model = joblib.load('Airline_XGB.h5')
columns = joblib.load('Airline_model_columns.h5')

def main() :
    
    st.write("Airline Fare Prediction")
    
    airline = st.selectbox('Please Select Airline' , ('IndiGo', 'Air India', 'Jet Airways', 'SpiceJet','Multiple carriers', 'GoAir', 'Vistara', 'Air Asia'))
    
    source = st.selectbox('Please Select Source' , ('Banglore', 'Kolkata', 'Delhi', 'Chennai', 'Mumbai'))
    
    destination = st.selectbox('Please Select Destination' , ('New Delhi', 'Banglore', 'Cochin', 'Kolkata', 'Delhi', 'Hyderabad'))
    
    stops = st.selectbox("Please Select Total Stops" , (0,1,2,3,4))
    
    month = st.selectbox('Select Flight Month' , (1,2,3,4,5,6,7,8,9,10,11,12))
    
    day = st.selectbox("Select journet Day" , ('Sunday', 'Saturday', 'Friday', 'Thursday', 'Monday', 'Tuesday',
       'Wednesday'))
    
    dep_hour = st.selectbox("Select Departure Hour" , tuple([x for x in range(24)]))
    
    dep_min = st.selectbox('Select Departure Minute' , tuple([x for x in range(60)]))
    
    arrival_hour = st.selectbox("Select Arrival Hour" , tuple([x for x in range(24)]))
    
    arrival_min = st.selectbox('Select Arrival Minute' , tuple([x for x in range(60)]))
    
    duration = st.number_input('Please Enter Flight Duration')
    
    prediction = 'Prediction is not made yet, Click Predict make prediction.'
    
    input_data = [airline,source,destination,stops,month,day,dep_hour,dep_min,arrival_hour,arrival_min,duration]
    
    input_df = pd.DataFrame([input_data] , columns=columns)
    
    if st.button('Predict Flight Price.'):
        prediction = f'Flight Price is {"{}".format(Model.predict(input_df)[0])}'
        
    st.success(prediction)
    
    
if __name__ == '__main__' :
    main()
