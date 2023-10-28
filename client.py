import requests
body = {
    "City" : "El Paso",
    "State" : "TX",
    "Vin" : "19VDE2E53EE000083",
    "Make" : "Acura",
    "Model" : "ILX6-Speed"
    }
response = requests.post(url = 'https://heart-ml-service-dianabeja.cloud.okteto.net/score',
              json = body)
print (response.json())
# output: {'score': 0.866490130600765}
