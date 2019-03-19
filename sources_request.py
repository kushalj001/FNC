import requests


url = 'http://localhost:5000/sources'
post_request = requests.post(url,json={'headline':'Congress will scrap Citizenship Bill forever: Rahul'})
print(post_request.json())
