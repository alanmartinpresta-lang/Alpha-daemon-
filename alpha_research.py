import argparse,json
from aire.alpha_internet import AlphaInternet
p=argparse.ArgumentParser()
p.add_argument("url")
a=p.parse_args()
print(json.dumps(AlphaInternet().get(a.url),ensure_ascii=False,indent=2))
