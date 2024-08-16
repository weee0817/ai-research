from openai import AzureOpenAI
import numpy as np
import pandas as pd

client = AzureOpenAI(
  api_key = "API_KEY",  
  api_version = "2024-02-01",
  azure_endpoint = "ENDPOINT"
)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# def generate_embeddings(text, model="text-embedding-ada-002"): # model = "deployment_name"
#     return np.array(client.embeddings.create(input = [text], model=model).data[0].embedding)
def generate_embeddings(text, model="text-embedding-ada-002"): # model = "deployment_name"
    return client.embeddings.create(input = [text], model=model).data[0].embedding

# df = pd.read_csv('words.csv')
# df['embedding'] = df['text'].apply(lambda x : generate_embeddings(x))
# df['embedding'] = df['embedding'].apply(lambda x: x.tolist())
# df.to_csv('word_embeddings.csv')

df = pd.read_csv('word_embeddings.csv', usecols=['text', 'embedding'])
df['embedding'] = df['embedding'].apply(eval).apply(np.array)
search_term = input('Enter a search term: ')
search_term_vector = generate_embeddings(search_term)
df['similarities'] = df['embedding'].apply(lambda x: cosine_similarity(x, search_term_vector))
df_result = df.sort_values('similarities', ascending=False).head(5)
print(df_result)

## DEMO 
# coffee = generate_embeddings("coffee")
# milk = generate_embeddings("milk")
# latte = generate_embeddings("latte")
# v1 = np.array(coffee) + np.array(milk)
# cs = cosine_similarity(v1, latte)
# print(cs)