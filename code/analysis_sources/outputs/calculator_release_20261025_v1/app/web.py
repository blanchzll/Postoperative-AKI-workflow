"""Loopback-only research UI; no patient input logging, external scripts or uploads."""
import argparse,json,time,threading,signal
from http.server import HTTPServer,BaseHTTPRequestHandler
from pathlib import Path
from engine import Calculator,parse_json
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--project-root");ap.add_argument("--port",type=int,default=8765);ap.add_argument("--duration",type=int,default=3600)
    a=ap.parse_args();calc=Calculator(a.project_root)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def send(self,status,data,ctype="application/json; charset=utf-8"):
            raw=data.encode() if isinstance(data,str) else data
            self.send_response(status);self.send_header("Content-Type",ctype);self.send_header("Content-Length",str(len(raw)));self.send_header("Cache-Control","no-store");self.send_header("X-Content-Type-Options","nosniff");self.end_headers();self.wfile.write(raw)
        def do_GET(self):
            if self.path=="/":self.send(200,Path(__file__).with_name("index.html").read_bytes(),"text/html; charset=utf-8")
            elif self.path=="/api/catalog":self.send(200,json.dumps(calc.public_catalog(),ensure_ascii=False))
            elif self.path=="/health":self.send(200,'{"status":"ok"}')
            else:self.send(404,'{"error":"Not found"}')
        def do_POST(self):
            origin=self.headers.get("Origin");expected="http://"+self.headers.get("Host","")
            if origin and origin!=expected:return self.send(403,'{"error":"Origin rejected"}')
            if self.path!="/api/predict":return self.send(404,'{"error":"Not found"}')
            if not self.headers.get("Content-Type","").startswith("application/json"):return self.send(415,'{"error":"JSON required"}')
            try:
                size=int(self.headers.get("Content-Length","0"))
                if size<=0 or size>8*1024*1024:raise ValueError("Request size invalid")
                q=calc.predict(parse_json(self.rfile.read(size).decode("utf-8")))
                self.send(200,json.dumps(q,ensure_ascii=False,allow_nan=False))
            except (ValueError,TypeError,KeyError,UnicodeError) as e:self.send(400,json.dumps({"error":str(e)},ensure_ascii=False))
            except Exception:self.send(500,'{"error":"Prediction unavailable; check private source-state qualification"}')
    server=HTTPServer(("127.0.0.1",a.port),Handler)
    signal.signal(signal.SIGTERM,lambda *args:threading.Thread(target=server.shutdown,daemon=True).start())
    timer=threading.Timer(a.duration,server.shutdown);timer.daemon=True;timer.start()
    try:server.serve_forever()
    finally:timer.cancel();server.server_close();calc.cache.clear()
if __name__=="__main__":main()
