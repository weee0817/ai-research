import unittest  
from unittest.mock import MagicMock  
from openai import AzureOpenAI  
from test2 import ask_question  # 假設你的程式碼存放在 your_module.py 中  
  
class TestAskQuestion(unittest.TestCase):  
      
    def setUp(self):  
        # 模擬 AzureOpenAI 客戶端  

        # Azure OpenAI 端點  
        endpoint = "ENDPOINT"  
        
        # 部署的模型名稱  
        deployment = "GPT"  
        
        # API 金鑰  
        key = "API_KEY"   
        
        # 初始化 AzureOpenAI 客戶端  
        client = AzureOpenAI(  
            azure_endpoint=endpoint,  # 設定端點  
            api_key=key,              # 設定 API 金鑰  
            api_version="2024-02-15-preview",  # 設定 API 版本  
        )  

        self.mock_client = client
        self.deployment = "GPT"  
        self.messages = [  
            {  
                "role": "system",  
                "content": "always return result number only !"  
            }  
        ]  
  
    def test_ask_question(self):  
        # 隨機生成 5 個數學問題  
        math_questions = [  
            "What is 2 + 2?",  
            "What is the square root of 16?",  
            "What is 10 divided by 2?",  
            "What is 3 times 4?",  
            "What is 5 minus 3?"  
        ]  
  
        # 模擬 AzureOpenAI 回應  
        for question in math_questions:  
            # self.mock_client.chat.completions.create.return_value = MagicMock(  
            #     choices=[MagicMock(message=MagicMock(content="4"))]  # 假設 AI 回答 "4"  
            # )  

            print(question)
            response = ask_question(self.mock_client, self.deployment, self.messages, question)  
              
            completion = self.mock_client.chat.completions.create(  
                model=self.deployment,  
                messages=self.messages,  
                max_tokens=800,  
                temperature=0.7,  
                top_p=0.95,  
                frequency_penalty=0,  
                presence_penalty=0,  
                stop=None,  
                stream=False  
            )

            ai_response = completion.choices[0].message.content  
            # 檢查回應是否正確  
            self.assertEqual(response, ai_response)  
  
if __name__ == '__main__':  
    unittest.main()  
