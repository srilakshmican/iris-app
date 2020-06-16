import streamlit as st
import pandas as pd 
from sklearn import datasets

#st.write("""# Iris Flower Prediction App""" )
st.title('Iris Classification App')
'Please adjust the input parameters in the left side menu bar  '
'ML model will auto run in the background & automatically classifies the category of Iris'

st.sidebar.header('User Input Parameters')

def user_input_features():

	sl = st.sidebar.slider('Sepal Length', 4.3, 7.9, 5.4)
	sw = st.sidebar.slider('Sepal Width', 2.0, 4.4, 3.4)
	pl = st.sidebar.slider('Petal Length', 1.0, 6.9, 1.3)
	pw = st.sidebar.slider('Petal Width', 0.1, 2.5, 0.2)

	data = {'sl':sl, 
				'sw':sw, 
				'pl':pl, 
				'pw':pw}

	features = pd.DataFrame(data, index=[0])
	return features

df = user_input_features() #side menu appears

st.subheader('User Input')
st.write(df)
