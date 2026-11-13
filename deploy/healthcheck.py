import json,urllib.request
r=json.load(urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=4))
assert r['status']=='ok' and r['loaded'] and r['max_context']>0 and r['api_key']
