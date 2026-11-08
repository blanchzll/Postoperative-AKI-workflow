from pathlib import Path
import os,sys,json,time,subprocess,socket,urllib.request,urllib.error,copy,hashlib
R=Path(__file__).resolve().parents[2];O=R/"outputs/calculator_release_20261025_v1";A=O/"app";sys.path.insert(0,str(A))
from engine import Calculator,parse_json
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
def main():
    calc=Calculator(R);ex=json.loads((O/"synthetic_examples/geography_preop_preop_compact.json").read_text())
    accepted=0;invalid=[]
    def reject(name,x):
        nonlocal accepted
        try:calc.validate(x)
        except (ValueError,TypeError):accepted+=1;invalid.append(name);return
        raise AssertionError("Invalid input accepted: "+name)
    for k,v in [("bad units","mg/dL"),("missing units",None)]:
        x=copy.deepcopy(ex)
        if v is None:x["units"].pop("PreopCr")
        else:x["units"]["PreopCr"]=v
        reject(k,x)
    for name,value in [("bool",True),("numeric string","80"),("infinity",float("inf")),("NaN",float("nan")),("negative creatinine",-1),("giant integer",10**500)]:
        x=copy.deepcopy(ex);x["records"][0]["PreopCr"]=value;reject(name,x)
    for name,value in [("invalid category","3"),("integer category",1),("bool category",False)]:
        x=copy.deepcopy(ex);x["records"][0]["Sex"]=value;reject(name,x)
    for k in [""," ", "key\n","patient/../../x","a"*129]:
        x=copy.deepcopy(ex);x["records"][0]["case_key"]=k;reject("invalid key",x)
    for name in ["duplicate case","omitted predictor","extra predictor","empty batch","oversize batch","unknown route","under18","over125"]:
        x=copy.deepcopy(ex)
        if name=="duplicate case":x["records"]*=2
        elif name=="omitted predictor":x["records"][0].pop("PreopAlb")
        elif name=="extra predictor":x["records"][0]["postop_creatinine"]=90
        elif name=="empty batch":x["records"]=[]
        elif name=="oversize batch":x["records"]*=1001
        elif name=="unknown route":x["route"]="../model"
        elif name=="under18":x["records"][0]["Age"]=17
        elif name=="over125":x["records"][0]["Age"]=126
        reject(name,x)
    for text in ['{"route":"a","route":"b"}','{"x":NaN}','{"x":Infinity}']:
        try:parse_json(text)
        except ValueError:accepted+=1;invalid.append("JSON parser");continue
        raise AssertionError("Invalid JSON was accepted")
    x=copy.deepcopy(ex);x["records"][0]["PreopCr"]=None;copy0=copy.deepcopy(x);calc.validate(x);assert x==copy0
    signed=json.loads((O/"synthetic_examples/geography_preop_preop_full.json").read_text());signed["records"][0]["WeightLoss"]=-2;calc.validate(signed)
    # Register an ephemeral private loopback server; only this process owns it.
    sock=socket.socket();sock.bind(("127.0.0.1",0));port=sock.getsockname()[1];sock.close();base=f"http://127.0.0.1:{port}"
    log=(O/"http_qa_server.log").open("w")
    proc=subprocess.Popen([sys.executable,str(A/"web.py"),"--project-root",str(R),"--port",str(port),"--duration","300"],stdout=log,stderr=log,env={**os.environ,"OPENBLAS_NUM_THREADS":"1","OMP_NUM_THREADS":"1"})
    comparisons=[]
    def request(payload=None,origin=None,content="application/json"):
        headers={"Content-Type":content}
        if origin:headers["Origin"]=origin
        req=urllib.request.Request(base+"/api/predict",data=json.dumps(payload,ensure_ascii=False).encode(),headers=headers)
        try:
            with urllib.request.urlopen(req,timeout=90) as res:return res.status,json.loads(res.read())
        except urllib.error.HTTPError as e:return e.code,json.loads(e.read())
    try:
        for _ in range(100):
            try:
                with urllib.request.urlopen(base+"/health",timeout=1) as res:assert res.status==200
                break
            except Exception:
                if proc.poll() is not None:raise RuntimeError("QA server exited")
                time.sleep(.1)
        for p in sorted((O/"synthetic_examples").glob("*.json")):
            payload=json.loads(p.read_text());expected=calc.predict(payload)
            status,actual=request(payload);assert status==200 and actual==expected,(p.name,status)
            cli=subprocess.run([sys.executable,str(A/"cli.py"),"--project-root",str(R),"--input",str(p)],capture_output=True,text=True,timeout=90)
            assert cli.returncode==0 and json.loads(cli.stdout)==expected,p.name
            comparisons.append({"route":payload["route"],"HTTP_equals_offline":True,"CLI_equals_offline":True})
        assert request(ex,origin="https://untrusted.invalid")[0]==403
        assert request(ex,content="text/plain")[0]==415
        bad=copy.deepcopy(ex);bad["records"][0]["PreopCr"]="BAD_PATIENT_VALUE_MARKER"
        status,err=request(bad);assert status==400 and "BAD_PATIENT_VALUE_MARKER" not in json.dumps(err)
        save(O/"IO_PARTIAL_ACCEPTANCE.json",{"invalid_cases_rejected":accepted,"route_equivalence":comparisons,"HTTP_and_CLI_passed_before_browser":True})
        browser=subprocess.run(["node",str(Path(__file__).with_name("browser_qa.js")),base,str(O)],capture_output=True,text=True,timeout=120)
        (O/"browser_qa.log").write_text(browser.stdout+"\n"+browser.stderr)
        assert browser.returncode==0,"Browser QA failed; inspect BROWSER_ACCEPTANCE.json"
        save(O/"IO_ACCEPTANCE.json",{"status":"PASS_validation_HTTP_CLI_and_actual_browser","invalid_cases_rejected":accepted,"invalid_case_categories":invalid,"ordinary_missing_permitted":True,"signed_weight_loss_preserved":True,"original_payload_not_mutated":True,"route_equivalence":comparisons,"private_origin_and_JSON_content_checks":True,"errors_do_not_echo_values":True,"server_host":"127.0.0.1","patient_inputs_not_persisted":True})
    finally:
        proc.terminate()
        try:proc.wait(timeout=10)
        except subprocess.TimeoutExpired:proc.kill();proc.wait()
        log.close()
        save(O/"HTTP_SERVER_EXIT.json",{"pid":proc.pid,"exit_code":proc.returncode,"owned_server_exited":True,"normal_shutdown":proc.returncode==0})
if __name__=="__main__":main()
