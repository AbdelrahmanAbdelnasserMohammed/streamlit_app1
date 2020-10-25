import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import streamlit as st 
import numpy as np 
import pandas as pd
import pickle
import base64


def download_link(object_to_download, download_filename, download_link_text):
    
    if isinstance(object_to_download,pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(object_to_download.encode()).decode()

    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'




# To disable warnings
st.set_option('deprecation.showfileUploaderEncoding', False)


st.title('Modifizierte Daten Zusammengefasst')

st.write("""
	# Predict Labels for CSV File!
""")



file = st.file_uploader("Please upload your csv file: ", type ="csv")

if not file:
	st.stop()

st.success('Thank you for Uploading file')

# Load Transformers required to perform the same preprocessing
lda_transformer = pickle.load(open('LDA_Transformer.sav', 'rb'))
encoders = pickle.load(open('encoder.obj', 'rb'))



#Load Data
df = pd.read_csv(file)

# Clean the Data Frame
df.pah.replace({'y': 'n'}, inplace = True)
mylist = list(df.select_dtypes(include=['object']).columns)

i = 0
for col in mylist:
    df[col]=df[col].fillna(np.random.choice(encoders[i].classes_))
    i += 1

mylist_num=(df.select_dtypes(include=['int64','float64']).columns)
for col in mylist_num:
    df[col]=df[col].fillna(df[col].mean())

# Load Model
model = pickle.load(open('model_MLP.sav', 'rb'))



# list with string features to encode them
mylist = list(df.select_dtypes(include=['object']).columns)

df_enc = df.copy()
i = 0
for col in mylist:
	df_enc[col] = encoders[i].transform(df[col])
	i += 1

lda_trans = lda_transformer.transform(df_enc)


# number of observations in DataFrame
nObservations = lda_trans.shape[0]
st.write(f"## Numer of observation: {nObservations} observations")


# Predict Labels and probabilities
labels = model.predict(lda_trans)
probs = model.predict_proba(lda_trans)
labels_str = ["No", "Yes"]

# Print Observations
if nObservations < 4:
	for i in range(labels.shape[0]):
		label = labels[i]
		lbl_str = labels_str[label]
		prob = probs[i, label]
		st.write(f"## for observation {i+1}:")
		st.write(f"Class is:  {lbl_str}")	
		st.write(f"Class Probability:  {round(prob, 2)}")



	
if st.button('Download Data Labeled as CSV'):
	lbls = [labels_str[lbl] for lbl in labels]
	c_probs = [round(probs[i, labels[i]], 2) for i in range(nObservations)]
	df['therapie_relevantes_delir_ja1'] = lbls
	df['Class_Prob'] = c_probs
	tmp_download_link = download_link(df, 'Labeled_data.csv', 'Click here to download your data!')
	st.markdown(tmp_download_link, unsafe_allow_html=True)
	




