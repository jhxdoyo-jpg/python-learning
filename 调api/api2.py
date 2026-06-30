import requests
try:
    r=requests.get("https://dummyjson.com/quotes/random")
    r.raise_for_status()
    data=r.json()
    print(f"{data['quote']}--{data['author']}")
except Exception as e:
    print("出错了",e)    