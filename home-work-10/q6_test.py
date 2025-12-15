import requests

url = "http://localhost:9696/predict"

client = {
    "contract": 0,
    "tenure": 12,
    "monthlycharges": 20.0,
    "totalcharges": 240.0
}

response = requests.post(url, json=client).json()
print(response)
