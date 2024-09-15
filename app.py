import streamlit as st
from PIL import Image
import os
import numpy as np
from src.plot import plot
import pandas as pd
from src.helper import seq_vec

# Page title
st.title('API access to Generative RNN:')

def convert_df(df): return df.to_csv(index=False).encode('utf-8')



# Text input for the image name
default_value = "MKWVTFISLLFLFSSAYSR"
sequence = st.text_input('Enter the peptide sequence:', value=default_value )
charge = st.number_input('Enter the peptide charge:', value=2, step=1, min_value=2, max_value=6)
NCE = st.number_input('Enter the normalized collision energy:', value=0.20, step=0.01, min_value=0.20, max_value=0.30)
sequence1 = str(sequence)
charge1 = int(charge)
NCE1 = float(NCE)

if sequence is not None:
    plot(model_path= "D:/ms2/src/model_20Epoch_protein.keras", input_sequence=sequence1, charge=charge1, NCE=NCE1)

csv = convert_df(pd.DataFrame(plot(model_path= "D:/ms2/src/model_20Epoch_protein.keras", input_sequence=sequence1, charge=charge1, NCE=NCE1)))

st.download_button( "Press to Download prediction", csv, "file.csv", "text/csv", key='download-csv' )

csv1 = convert_df(pd.DataFrame(list(seq_vec(sequence1))))

st.download_button( "Press to Download fragments", csv1, "file.csv", "text/csv", key='download-csv1' )

# Path to the folder containing the images
folder_path = 'D:/repo/'
file_name = sequence1 + "_" + str(charge1) + "_" + str(round(NCE1, 2)) + ".jpeg"
image_path = folder_path + file_name

if os.path.exists(image_path):
    # Display the uploaded image
    image = Image.open(image_path)
    st.image(image, caption='Uploaded Image', use_column_width=True)
else:
    st.write('The specified image file does not exist.')
