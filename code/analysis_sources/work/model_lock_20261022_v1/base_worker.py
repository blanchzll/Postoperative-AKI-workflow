from pathlib import Path
import sys,json,time,resource
from runtime import fit_base,atomic,OUT,base_id
t=json.loads(Path(sys.argv[1]).read_text());start=time.time()
fit_base(t)
atomic(OUT/'base_fits'/base_id(t)/'WORKER_RESOURCE.json',{'pid':__import__('os').getpid(),
 'seconds':time.time()-start,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024})
