# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import easyocr
from PIL import Image
import re
from bs4 import BeautifulSoup
from transformers import AlbertTokenizer, TFAlbertModel
from keras.models import load_model
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")




@st.cache(allow_output_mutation=True,max_entries=10, ttl=3600)
def loading_model():
  bert  = TFAlbertModel.from_pretrained('albert-base-v2')
  model=tf.keras.models.load_model('model.h5')
  return model


@st.cache(allow_output_mutation=True,max_entries=10, ttl=3600)
def  cleaning_text(sentence):
    
    #Replace empois to meaningful words
    sentence = re.sub(r'😍|🥰|😘|😻|❤️|🧡|💛|💚|💙|💜|🖤|🤍|🤎|💕|💞|💓|💗|💖|💘|💝', 'love ', sentence)
    sentence = re.sub(r'😀|😃|😄|😁|😆|😂|🤣|😹|😸', 'excited ', sentence)
    sentence = re.sub(r'🙏|👌|💪|👊|👍|✌️|👏|🙌|💯|✅|✔️|👐', 'optimistic ', sentence) 
    sentence = re.sub(r'😢|😭|😿|😔|😩|😰|😪|💔|😓|😐', 'sad ', sentence)
    sentence = re.sub(r'✈️|🔥|💫|⭐️|🌟|✨|💥|🛩|🚀', 'wow ', sentence)
    sentence = re.sub(r'😷', 'mask ', sentence)
    sentence = re.sub(r'💉', 'vaccine ', sentence)
    
    #Text cleaning
    sentence = re.sub(r'http\S+', '', sentence) #Removes urls
    sentence = re.sub(r'[^a-zA-Z0-9 ]',r'',sentence) #Removes special charaters
    sentence = sentence.encode("ascii", "ignore") #Removes Non-ASCII characters
    sentence = sentence.decode()
    sentence = " ".join(re.split("\s+", sentence, flags=re.UNICODE)) #Remove spaces in the BEGINNING of a string
    sentence = re.sub("^\s+|\s+$", "", sentence, flags=re.UNICODE) #Remove spaces both in the BEGINNING and in the END of a string and also consecutive spaces
    soup     = BeautifulSoup(sentence) #Remove HTML Tags
    sentence = soup.get_text()
    sentence = sentence.lower() #Converts all characters to lowercase
       
    return sentence



@st.cache(allow_output_mutation=True,max_entries=10, ttl=3600)
def extracting_text(url):
    reader = easyocr.Reader(['en'])
    result = reader.readtext(url,paragraph="False")
    sent=""
    for i in result:
        sent+=" "+(i[-1])
    
    return sent[1:]

@st.cache(allow_output_mutation=True,max_entries=10, ttl=3600)
def tokenize(sentences,max_len):
    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2') 
    bert  = TFAlbertModel.from_pretrained('albert-base-v2')
    bert_inp=tokenizer.encode_plus(sentences,max_length=max_len,add_special_tokens = True,pad_to_max_length = True,return_attention_mask = True,truncation = True)
    
    input_ids=np.asarray(bert_inp['input_ids']).reshape(-1,max_len)
    attention_masks=np.asarray([bert_inp['attention_mask']]).reshape(-1,max_len)
    
    return input_ids,attention_masks

@st.cache(allow_output_mutation=True,max_entries=10, ttl=3600)
def preparing_data(sentences,sentences1,url):

    input_ids,attention_masks=tokenize(sentences,250)
    input_ids1,attention_masks1=tokenize(sentences1,245)
    input_arr = np.array(url,dtype='float32') 
    image = tf.convert_to_tensor(input_arr,dtype=tf.float32)  
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, size=(200, 200))
    image = tf.reshape(image,(-1,200,200,3))
    
    data={'input_ids': input_ids, 'attention_mask': attention_masks, 
            'input_ids2': input_ids1, 'attention_mask2': attention_masks1,'images': image}
    dataset = tf.data.Dataset.from_tensor_slices(data).batch(1)
    
    return dataset

@st.cache(allow_output_mutation=True,max_entries=10, ttl=3600,suppress_st_warning=True)
def plotting_positive_plot(pos,neg):
  st.success('The result of sentimental analysis is POSITIVE')
  col1, col2, col3 = st.columns(3)
  with col1:
    st.write(' ')
  with col2:
    st.image("Happy.PNG", width=340, use_column_width=350)
  with col3:
    st.write(' ')

  dataframe = pd.DataFrame({'Polarity':['Positive', 'Negative'],'Percentage':[pos,neg]})
  graph=px.bar(dataframe,x='Polarity',y='Percentage',color='Polarity')

  return st.plotly_chart(graph,use_container_width=True)

@st.cache(allow_output_mutation=True,max_entries=10, ttl=3600,suppress_st_warning=True)
def plotting_negative_plot(pos,neg):
  st.warning('The result of sentimental analysis is NEGATIVE')
  col1, col2, col3 = st.columns(3)
  with col1:
    st.write(' ')
  with col2:
    st.image("Sad.PNG", width=340, use_column_width=350)
  with col3:
    st.write(' ')

  dataframe = pd.DataFrame({'Polarity':['Negative', 'Positive'],'Percentage':[pos,neg]})
  graph=px.bar(dataframe,x='Polarity',y='Percentage',color='Polarity')
    
  return st.plotly_chart(graph,use_container_width=True)
      
    
model=loading_model()

def main():
    st.title("SENTIMENTAL ANALYSIS")
    
    tweet_text = st.text_area("Please Tweet")
    
    image_file = st.file_uploader("Please upload an image", type=["jpg"])
    
    if st.button("Predict"):
      if image_file is None and len(tweet_text)==0:
        st.warning("Please Tweet and upload an image!")
        
      elif image_file is None and len(tweet_text)>=1:
        st.warning("Image has not been uploaded yet, Please upload an image")
        
      elif image_file is not None and len(tweet_text)==0:
        st.warning("Tweet is empty, Please Tweet!")
        
      else:
        image = Image.open(image_file)
        if image.format != 'JPEG':
          st.warning("Uploaded image is not in .jpg format, please re-upload an image with .jpg fomat")
        else:
          tweet_text = cleaning_text(tweet_text)
          extracted_image_text = extracting_text(image)
          extracted_image_text = cleaning_text(extracted_image_text)
          if len(extracted_image_text)==0:
            st.warning("Uploaded image contains no textual data, please re-upload an image with text inside it")
          else:
            col1, col2, col3 = st.columns(3)
            with col1:
              st.write(' ')
            with col2:
              st.image(image, width=400, use_column_width=370,caption='UPLOADED IMAGE')
            with col3:
              st.write(' ')

            data = preparing_data(tweet_text,extracted_image_text,image)
            prediction = model.predict(data)[0]
            predictions = prediction[0]
            if predictions>=0.50:
              pos=predictions*100
              neg=(100-(predictions*100))
              plotting_positive_plot(pos,neg)
            else:
              neg=predictions*100
              pos=(100-(predictions*100))
              plotting_negative_plot(pos,neg)      
       
if __name__ == '__main__':
    main()
    
