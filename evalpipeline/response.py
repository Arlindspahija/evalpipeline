from openai import AzureOpenAI
from typing import TypedDict

class GenerateAnswerResponse(TypedDict):
    response: str

class generate_answer:
    def __init__(self, azure_endpoint: str, api_key: str, api_version: str, azure_deployment: str, model: str, **kwargs):
        self.system_message = ""
        self.endpoint = azure_endpoint
        self.api_key = api_key
        self.api_version = api_version
        self.deployment = model

    def __call__(self, query: str, user_instructions: str, context: str) -> GenerateAnswerResponse:

        client = AzureOpenAI(
            azure_endpoint = self.endpoint, 
            api_key=self.api_key,  
            api_version=self.api_version,
        )

        self.system_message = f"""
        You are an Email Assistant and provide answers for emails given a context and user instructions.

        user instructions:
        {user_instructions}
        
        context: 
        {context}
        """

        response = client.chat.completions.create(
            model=self.deployment,
            messages=[
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": query}
            ]
        )
        if response and response.choices:
            answer = response.choices[0].message.content
            print(answer)
            return {
                "response": answer
            }
        else:
            return {
                "response": "Error in generating a response."
            }