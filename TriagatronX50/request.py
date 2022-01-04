import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'journal_entry':200})

print(r.json())