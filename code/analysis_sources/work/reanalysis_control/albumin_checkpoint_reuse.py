from pathlib import Path
import json,hashlib
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify(root):
 root=Path(root);o=root/"outputs/albumin_continuous_20261024_v2"
 seal=json.loads((root/"outputs/reanalysis_control/albumin_metadata_repair_20261028_v1/RETAINED_FIT_REUSE_SEAL.json").read_text())
 assert sha(root/"outputs/v4_development_inputs_20261018/SOURCE_LOCK.json")==seal["source_lock_sha256"]
 assert sha(o/"CONTRACT.json")==seal["source_contract_sha256"]
 for rel,h in seal["code_sha256"].items():assert sha(root/rel)==h
 for path,h in seal["files_sha256"].items():assert sha(path)==h
 contract=json.loads((o/"CONTRACT.json").read_text())
 for rel,h in contract["sources"].items():assert sha(root/rel)==h
 assert sha(o/"SOURCE_LOAD_QUALIFICATION.json")==seal["true_RRT_source_qualification_sha256"]
 return seal
