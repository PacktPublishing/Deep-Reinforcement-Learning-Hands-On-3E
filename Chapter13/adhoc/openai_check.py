import os
import openai

OPENAI_KEY = os.environ["OPENAI_API_KEY"]

def check_openai_api_key(api_key):
    client = openai.OpenAI(api_key=api_key)
    try:
        client.models.list()
    except openai.AuthenticationError as e:
        print(e)
        return False
    else:
        return True

# Check the validity of the API key
api_key_valid = check_openai_api_key(OPENAI_KEY)
print("API key is valid:", api_key_valid)