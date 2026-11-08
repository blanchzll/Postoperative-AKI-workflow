"""Apply independently regenerated exact-decimal target to a fresh output CSV, never overwrite accepted inputs."""
from pathlib import Path
import csv,argparse,json
p=argparse.ArgumentParser();p.add_argument("--candidate-metadata",required=True);p.add_argument("--exact-target-table",required=True);p.add_argument("--out",required=True);p.add_argument("--verify-against")
a=p.parse_args()
def rows(p):
 with Path(p).open(newline="") as f:return list(csv.DictReader(f))
base=rows(a.candidate_metadata);truth={r["record_key"]:r for r in rows(a.exact_target_table)}
assert len(base)==len(truth)==759 and set(r["record_key"] for r in base)==set(truth)
for r in base:
 r["y_scr_calendar"]=truth[r["record_key"]]["y_exact"]
 r["first_recorded_positive_charttime"]=truth[r["record_key"]]["first_positive_exact"]
out=Path(a.out);assert not out.exists()
with out.open("w",newline="") as f:
 w=csv.DictWriter(f,fieldnames=list(base[0]));w.writeheader();w.writerows(base)
assert sum(int(r["y_scr_calendar"]) for r in base)==59
if a.verify_against:
 expected=rows(a.verify_against);assert base==expected
print(json.dumps({"status":"PASS_EXACT_TARGET_REBUILD","n":len(base),"events":59,"all_metadata_columns_match_current":bool(a.verify_against)}))
