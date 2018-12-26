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

# API_PRIVATE_URL = API_ENDPOINT_URL + "api/v1/private"
# AUTH_URL = API_ENDPOINT_URL + "auth"


def get_api_key(name, password):
    response = requests.post(AUTH_URL, json={'username': name, 'password': password})
    print(response)
    response = json.loads(response.text)
    key = response.get("access_token", None)
    if key is not None:
        response["description"] = "Credentials Validated"
        response["error"] = "Success"
        response["status_code"] = 200
    else:
        response["description"] += ", please contact JNResearch!"
        response["access_token"] = None
    return key, response

if __name__ == "__main__":
    uname = input("Enter user name: ")
    pw = input("Enter password: ")
    data, resp = get_api_key(uname, pw)
    print("Your API Key is:", data)
    print("Status Returned from Service: ", resp )

