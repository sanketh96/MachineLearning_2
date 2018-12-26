"""
Given the user name and password gets the API key from server
"""
import requests
import json
import time

# API_ENDPOINT_URL = "http://localhost:5000/"
API_ENDPOINT_URL = "http://jnresearchlabs.com/"
API_PRIVATE_URL = API_ENDPOINT_URL + "api/v1/private"
AUTH_URL = API_ENDPOINT_URL + "auth"
TEST_KEY = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJleHAiOjE1NTQyMjMwNzgsImlhdCI6MTUzODY3MTA3OCwibmJmIjoxNTM4NjcxMDc4LCJpZGVudGl0eSI6MTAwfQ.RPxXYSpX9UfkVr2WDGtNWdRYYD15n8sVPKM4pym2iCQ"

class ApiClient(object):
    def __init__(self, auth_key=None):
        if auth_key is None:
            auth_key = TEST_KEY
        self.data = auth_key
        return

    def echo(self, text):
        val = requests.get(API_PRIVATE_URL, 
                            headers={"Authorization":"JWT "+ self.data}, 
                            params={"command": "echo", "text": text})
        return(json.loads(val.text)["result"]["text"])

    def get_amazon_product_reviews(self, num_samples):
        val = requests.get(API_PRIVATE_URL, 
                            headers={"Authorization":"JWT "+ self.data}, 
                            params={"command": "amazon_product_reviews", "num_samples": num_samples})
        val = json.loads(val.text)
        return val["result"]["data"]

    def algebra(self, num_samples):
        val = requests.get(API_PRIVATE_URL, 
                            headers={"Authorization":"JWT "+ self.data}, 
                            params={"command": "algebra", "num_samples": num_samples})
        val = json.loads(val.text)
        return val["result"]["data"]
    
    def w2v(self, words):
        """Given a list of words get their vector representation using Glove"""
        print(words)
        val = requests.get(API_PRIVATE_URL, 
                            headers={"Authorization":"JWT "+ self.data}, 
                            params={"command": "w2v", "words": json.dumps(words)})
        val = json.loads(val.text)
        return val["result"]["data"]
    
    def get_kaggle_quora_data(self, num_samples):
        val = requests.get(API_PRIVATE_URL, 
                            headers={"Authorization":"JWT "+ self.data}, 
                            params={"command": "get_kaggle_quora_data", "num_samples": num_samples})
        val = json.loads(val.text)
        return val["result"]["data"]

