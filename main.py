import streamlit as st
import pickle as pkl
import numpy as np

class_list = {'1': 'Positive', '2': 'Negative', '0': 'Normal'}

st.title('Emotional Prediction')
input = open('ec_vsfc.pkl', 'rb')
encoder = pkl.load(input)

input = open('lrc_vsfc_1.pkl', 'rb')
model = pkl.load(input)

st.header('Write Vietnamese Sentence')
txt = st.text_area("","")
                         
if txt != '':
  if st.button('Predict'):
    feature_vector = encoder.transform([txt])
    label = str((model.predict(feature_vector))[0])
    st.header('Result')
    st.text(class_list[label])
