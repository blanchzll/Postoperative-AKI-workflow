"""JSON stdin/file CLI; same Calculator.predict as HTTP."""
import argparse,json,sys
from engine import Calculator,parse_json
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--project-root");ap.add_argument("--input");ap.add_argument("--output");ap.add_argument("--validate-only",action="store_true");ap.add_argument("--catalog",action="store_true")
    a=ap.parse_args();calc=Calculator(a.project_root)
    try:
        if a.catalog:q=calc.public_catalog()
        else:
            text=open(a.input).read() if a.input else sys.stdin.read()
            p=parse_json(text)
            if a.validate_only:
                route,rows=calc.validate(p);q={"status":"VALID_INPUT","route":route,"records":len(rows),"model_loaded":False}
            else:q=calc.predict(p)
        out=json.dumps(q,ensure_ascii=False,allow_nan=False)+"\n"
        if a.output:
            with open(a.output,"w") as f:f.write(out)
        else:sys.stdout.write(out)
    except (ValueError,TypeError,KeyError,json.JSONDecodeError) as e:
        sys.stderr.write(json.dumps({"error":str(e)},ensure_ascii=False)+"\n");return 2
    return 0
if __name__=="__main__":raise SystemExit(main())
