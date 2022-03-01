import streamlit as st
st.set_page_config(layout="wide",page_title='Search Engine', page_icon="🔍")
from streamlit import caching
import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import random
import time

hide_streamlit_style = """
                    <style>
                    #MainMenu {visibility: hidden;}
                    footer {visibility: hidden;}
                    </style>
                    """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.markdown("<h1><center>Search Engine</center></h1>",unsafe_allow_html=True)
@st.experimental_memo(show_spinner=False)
def process(data_file):
    data = json.load(data_file)
    random.shuffle(data)
    json.dump(data, open("faq.json", "wt", encoding="utf-8"), ensure_ascii=False, indent=4)

@st.experimental_memo(show_spinner=False)
def encode_standard_question(faq_data,pretrained_model):
    bc = SentenceTransformer(pretrained_model)
    standard_questions = [each['question'] for each in faq_data]
    print("Standard question size", len(standard_questions))
    print("Start to calculate encoder....")
    standard_questions_encoder = torch.tensor(bc.encode(standard_questions)).numpy()

    np.save("./questions", standard_questions_encoder)
    standard_questions_encoder_len = np.sqrt(np.sum(standard_questions_encoder * standard_questions_encoder, axis=1))
    np.save("./questions_len", standard_questions_encoder_len)

@st.experimental_memo(show_spinner=False)
def load_files(pretrained_model):
    faq_data = json.load(open("./faq.json", "rt", encoding="utf-8"))
    questions_encoder = np.load("./questions.npy")
    questions_encoder_len = np.load("./questions_len.npy")
    bc = SentenceTransformer(pretrained_model)
    return bc,questions_encoder,questions_encoder_len,faq_data

def get_most_similar_question_id(bc,questions_encoder,questions_encoder_len,query_question, top_n=1):
    query_vector = torch.tensor(bc.encode([query_question])[0]).numpy()
    
    score = np.sum((questions_encoder * query_vector), axis=1) / (questions_encoder_len * (np.sum(query_vector * query_vector) ** 0.5))
    top_id = np.argsort(score)[::-1][:top_n]
    score = sorted(score,reverse=True)[:top_n]
    return top_id, score

def run(bc,questions_encoder,questions_encoder_len,faq_data,query,top_n):
        
        related_resp = []
        related_ques = []
        most_similar_id, score = get_most_similar_question_id(bc,questions_encoder,questions_encoder_len,query,top_n)
        for idx,(i,j) in enumerate(zip(most_similar_id,score)):
            if float(j) > score_threshold:
                if idx==0:
                    response = faq_data[i]['answer']
                else:
                    related_ques.append(faq_data[i]['question'])
                    related_resp.append(faq_data[i]['answer'])
            else:
                response = "No results found for this query!"
        
        return response,related_ques,related_resp


data_file = st.sidebar.file_uploader("Upload Data")
pretrained_model = st.sidebar.text_input("Select Model name", value = 'all-MiniLM-L12-v2', help="Get all models list here: https://huggingface.co/models?library=sentence-transformers (Default: all-MiniLM-L12-v2)")

if data_file!=None:

    
    if st.sidebar.button("Train"):
        load_files.clear()
        
        with st.spinner("Training model on your data, Pls wait... (This may take time depending on data size)"):
            
            process(data_file)
            encode_standard_question(json.load(open("./faq.json", "rt", encoding="utf-8")),pretrained_model)

    score_threshold = st.sidebar.slider("Model Sensitivity",min_value=0.5,max_value=1.0,value=0.7,step=0.05)

            
    query = st.text_input("Write your query...",key=1)
    
    if query!="":
        start_time = time.time()
        try:
            bc,questions_encoder,questions_encoder_len,faq_data = load_files(pretrained_model)
            responses = run(bc,questions_encoder,questions_encoder_len,faq_data,query,top_n = 5)
            end_time = time.time() - start_time

            st.markdown(f"<small>Query Took {int(end_time/60)} minutes, {(end_time%60):.3f} seconds</small>",unsafe_allow_html=True)
            
            st.info(responses[0])
            if responses[1]!=[]:
                st.markdown("<b>Related Queries:</b>",unsafe_allow_html=True)
                for related_ques,related_ans in zip(responses[1],responses[2]):
                    with st.expander(related_ques):
                        st.write(related_ans)
        except:
            st.error("Please train the model first from sidebar!")
        
        

else:
    with st.expander("Steps:",expanded=True):
        st.markdown("1. Upload data file in sidebar in following <a href='https://drive.google.com/file/d/1uwJdMqhF_VaiyngQkZWh02l93HlwIFRw/view?usp=sharing' target='_blank'>format</a> only. \n\
2. Select model of your choice or leave it default.  \n\
3. Train model on data.  \n\
4. Search for queries on home page.",unsafe_allow_html=True)

    st.text_input("Write your query...",key=2)


st.markdown("<small><center><b>Made with ♥ by <a href='https://www.linkedin.com/in/aaryanverma' target='_blank'>Aaryan Verma</a></b></center></small>",unsafe_allow_html=True)