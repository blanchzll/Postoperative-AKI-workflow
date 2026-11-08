from pathlib import Path
import subprocess,sys,json,datetime,hashlib
R=Path('/home/lei/codex-workspaces/aki-prediction-manuscript/correction_runs/full_data_20261025_postslot')
W=R/'work/analysis_figures_current_20261029_v1'
O=R/'outputs/final_analysis_delivery_20261025_v1'
try:
 for name in ['render_primary.py','render_shap.py','render_curves.py']:
  (O/'RENDER_STATUS.json').write_text(json.dumps({'status':'RENDERING','script':name,'time':datetime.datetime.now().astimezone().isoformat()},indent=2)+'\n')
  subprocess.run([sys.executable,'-B',str(W/name)],check=True,cwd=R)
 (O/'RENDER_STATUS.json').write_text(json.dumps({'status':'RENDERED_VISUAL_REVIEW_PENDING','time':datetime.datetime.now().astimezone().isoformat(),'all_stages_exited':True,'final_science_complete':False},indent=2)+'\n')
except Exception as e:
 (O/'RENDER_FAILURE.json').write_text(json.dumps({'error':repr(e),'time':datetime.datetime.now().astimezone().isoformat()},indent=2)+'\n')
 raise
