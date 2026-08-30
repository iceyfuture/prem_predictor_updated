"""Build standalone.html (data + CSS inlined) and floodlit_artifact.html (wrapper-stripped)
from index.html + dashboard.json + minimal.css. Run after build_dashboard.py."""
import json, re, os
H=os.path.dirname(os.path.abspath(__file__))
html=open(os.path.join(H,"index.html"),encoding="utf-8").read()
mincss=open(os.path.join(H,"minimal.css"),encoding="utf-8").read()
data=json.load(open(os.path.join(H,"dashboard.json")))
html=html.replace('<link rel="stylesheet" href="minimal.css">',"<style>\n"+mincss+"\n</style>")
inject='<script>window.__DASHBOARD__='+json.dumps(data,separators=(",",":"))+';</script>\n'
html=html.replace("<script>\nconst $=id=>", inject+"<script>\nconst $=id=>",1)
open(os.path.join(H,"standalone.html"),"w",encoding="utf-8").write(html)
head=html.split("</head>",1)[0]
title=re.search(r"<title>.*?</title>",head,re.S).group(0)
styles="".join(re.findall(r"<style>.*?</style>",head,re.S))
body=html.split("<body>",1)[1].rsplit("</body>",1)[0]
open(os.path.join(H,"floodlit_artifact.html"),"w",encoding="utf-8").write(title+"\n"+styles+"\n"+body)
print("wrote standalone.html + floodlit_artifact.html (%d KB)"%(os.path.getsize(os.path.join(H,"floodlit_artifact.html"))//1024))
