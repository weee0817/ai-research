import os
import re
import requests
import sys
from num2words import num2words
import os
import pandas as pd
import numpy as np
import tiktoken
from openai import AzureOpenAI

df=pd.read_csv(os.path.join(os.getcwd(),'bill_sum_data.csv')) # This assumes that you have placed the bill_sum_data.csv in the same directory you are running Jupyter Notebooks
df_bills = df[['text', 'summary', 'title']]

pd.options.mode.chained_assignment = None #https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#evaluation-order-matters

# s is input text
def normalize_text(s, sep_token = " \n "):
    s = re.sub(r'\s+',  ' ', s).strip()
    s = re.sub(r". ,","",s)
    # remove all instances of multiple spaces
    s = s.replace("..",".")
    s = s.replace(". .",".")
    s = s.replace("\n", "")
    s = s.strip()
    
    return s

df_bills['text']= df_bills["text"].apply(lambda x : normalize_text(x))
# print(df_bills)

## remove any bills that are too long for the token limit (8192 tokens)
tokenizer = tiktoken.get_encoding("cl100k_base")
df_bills['n_tokens'] = df_bills["text"].apply(lambda x: len(tokenizer.encode(x)))
df_bills = df_bills[df_bills.n_tokens<8192]
# print(len(df_bills)) # output: 20
# print(df_bills)

sample_encode = tokenizer.encode(df_bills.text[0]) 
decode = tokenizer.decode_tokens_bytes(sample_encode)
# print(len(decode)) # output: 1466

client = AzureOpenAI(
  api_key = "API_KEY",  
  api_version = "2024-02-01",
  azure_endpoint = "ENDPOINT"
)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def generate_embeddings(text, model="text-embedding-ada-002"): # model = "deployment_name"
    return client.embeddings.create(input = [text], model=model).data[0].embedding

def search_docs(df, user_query, top_n=4, to_print=True):
    embedding = generate_embeddings(
        user_query,
        model="text-embedding-ada-002" # model should be set to the deployment name you chose when you deployed the text-embedding-ada-002 (Version 2) model
    )
    df["similarities"] = df.ada_v2.apply(lambda x: cosine_similarity(x, embedding))

    res = (
        df.sort_values("similarities", ascending=False)
        .head(top_n)
    )
    if to_print:
        print(res)
    return res

# df_bills['ada_v2'] = df_bills["text"].apply(lambda x : generate_embeddings (x, model = 'text-embedding-ada-002')) # model should be set to the deployment name you chose when you deployed the text-embedding-ada-002 (Version 2) model
# # print(df_bills)
# df_bills.to_csv('bill_sum_data_embeddings.csv')

# res = search_docs(df_bills, "Can I get information on cable company tax revenue?", top_n=4)
df_bills = pd.read_csv('bill_sum_data_embeddings.csv')
df_bills['ada_v2'] = df_bills['ada_v2'].apply(eval).apply(np.array)
res = search_docs(df_bills, "Can I get information on cable company tax revenue?", top_n=4)
print(res["summary"])