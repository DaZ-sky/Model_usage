import requests
import base64
import json
import sys

def pic_to_str(pic_path):
    with open(pic_path, "rb") as image:
        b64string = base64.b64encode(image.read()).decode('ASCII')
    return b64string

args = sys.argv

url = args[1]
data = "{\"pic\":\"" + pic_to_str(args[2]) + "\"}"
headers = {'Content-type': 'application/json'}
response = requests.post(url, data=data, headers=headers)

print(response.status_code)
print(response.json())