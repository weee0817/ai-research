import os  
import requests  
from openai import AzureOpenAI  
import json  
  
endpoint = "ENDPOINT"  
deployment = "Dalle3"  
key = "API_KEY"  
client = AzureOpenAI(  
    azure_endpoint=endpoint,  
    api_key=key,  
    api_version="2024-02-01",  
)  
  
result = client.images.generate(  
    model=deployment,  
    prompt="長著蝙蝠翅膀的丹麥公主在玫瑰藤寄生的高塔上和邪惡的闇之騎士爭吵 遠方魔女騎著噴火龍向他們飛來",  
    n=1  
)  
  
image_url = json.loads(result.model_dump_json())['data'][0]['url']  
print(image_url)  
  
# Save image 
response = requests.get(image_url)  
  
if response.status_code == 200:  
    image_path = "PATH"  
    with open(image_path, 'wb') as f:  
        f.write(response.content)  
    print(f"Image saves to {image_path}")  
else:  
    print("Unable to save image")  
