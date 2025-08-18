import requests
response=requests.post('http://localhost:8008/generate',json={'prompt': '你好，世界！'})
print(response.json())