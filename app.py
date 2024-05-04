import streamlit as st
from PIL import Image
import os
import numpy as np

# Page title
st.title('Genrative recurrent neural network')

# Text input for the image name
peptide_sequence = st.text_input('Enter the peptide sequence:')
peptide_charge = st.text_input('Enter the peptide charge:')

my_estimate = 2*peptide_charge
st.write(my_estimate)




