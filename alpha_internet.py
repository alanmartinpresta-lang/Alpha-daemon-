
"""Controlled, read-only Internet access for Alpha."""
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import json, time

class AlphaInternet:
    def __init__(self, log="ALPHA_INTERNET_LOG.jsonl", allowed_domains=None,
                 max_bytes=1_000_000, timeout=10):
        self.log_path=Path(log)
        self.allowed_domains=set(allowed_domains or {
            "docs.python.org","www.wikipedia.org","en.wikipedia.org",
            "fr.wikipedia.org","arxiv.org","www.arxiv.org",
            "github.com","raw.githubusercontent.com"
        })
        self.max_bytes=max_bytes; self.timeout=timeout

    def get(self,url):
        u=urlparse(url)
        if u.scheme!="https" or not u.hostname:
            raise ValueError("Only HTTPS URLs are allowed.")
        if u.hostname.lower().rstrip(".") not in self.allowed_domains:
            raise ValueError("Domain is not allowlisted.")
        t=time.time()
        try:
            req=Request(u.geturl(),method="GET",
                        headers={"User-Agent":"Alpha-Research-Agent/1.0"})
            with urlopen(req,timeout=self.timeout) as r:
                data=r.read(self.max_bytes+1)
                result={"ok":True,"url":u.geturl(),
                        "status":getattr(r,"status",200),
                        "content_type":r.headers.get("Content-Type",""),
                        "body":data[:self.max_bytes].decode("utf-8","replace"),
                        "truncated":len(data)>self.max_bytes,
                        "elapsed_s":round(time.time()-t,3)}
        except Exception as e:
            result={"ok":False,"url":u.geturl(),"error":str(e),
                    "elapsed_s":round(time.time()-t,3)}
        with self.log_path.open("a",encoding="utf-8") as f:
            f.write(json.dumps({k:v for k,v in result.items() if k!="body"},
                               separators=(",",":"))+"\n")
        return result
