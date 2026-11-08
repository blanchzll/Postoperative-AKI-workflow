"""Independent scalar definitions; stdlib only, no native metric imports."""
import math
def sigmoid(x):
 if x>=0:return 1/(1+math.exp(-x))
 a=math.exp(x);return a/(1+a)
def logit(p):
 p=min(1-1e-6,max(1e-6,float(p)));return math.log(p/(1-p))
def quantile(values,q):
 a=sorted(float(x) for x in values if x is not None and math.isfinite(x))
 if not a:return None
 position=(len(a)-1)*q;i=math.floor(position);j=math.ceil(position)
 return a[i]+(a[j]-a[i])*(position-i)
def close(actual,expected,tolerance=1e-10):
 if expected is None:return actual is None or (isinstance(actual,float) and math.isnan(actual))
 return actual is not None and math.isfinite(actual) and abs(float(actual)-float(expected))<=tolerance
def offset_intercept(y,p,w):
 n=sum(w);e=sum(a*b for a,b in zip(y,w))
 if not 0<e<n:return None
 z=[logit(x) for x in p];low,high=-60.,60.
 for _ in range(100):
  mid=(low+high)/2
  if math.fsum(b*sigmoid(a+mid) for a,b in zip(z,w))>e:high=mid
  else:low=mid
 return (low+high)/2
def joint_score_residual(y,p,w,intercept,slope):
 if intercept is None or slope is None:return None
 n=sum(w);z=[logit(x) for x in p];mu=math.fsum(a*b for a,b in zip(z,w))/n
 sd=math.sqrt(math.fsum(b*(a-mu)**2 for a,b in zip(z,w))/n)
 if sd<1e-10:return None
 err=[sigmoid(intercept+slope*a)-b for a,b in zip(z,y)]
 return max(abs(math.fsum(e*v for e,v in zip(err,w))/n),
            abs(math.fsum(e*v*(a-mu)/sd for e,v,a in zip(err,w,z))/n))
def thresholds(y,p):
 positive=sorted((float(a) for a,b in zip(p,y) if b),reverse=True)
 assert positive
 a=sorted(map(float,p));n=len(a)
 return {"training_sensitivity80":positive[math.ceil(.8*len(positive))-1],
         "training_capacity10":a[math.ceil(.9*(n-1))],
         "training_capacity20":a[math.ceil(.8*(n-1))]}
def metrics(y,p,g,r,w=None,cuts=None,dca=(),include_offset=False):
 y=list(y);assert all(a in (0,1) for a in y);y=list(map(int,y));p=list(map(float,p));g=list(map(str,g));r=list(map(str,r))
 w=[1]*len(y) if w is None else list(w)
 assert all(math.isfinite(a) and a==int(a) for a in w);w=list(map(int,w))
 assert len(y)==len(p)==len(g)==len(r)==len(w) and len(set(zip(g,r)))==len(y)
 assert all(a in (0,1) for a in y) and all(math.isfinite(a) and 0<=a<=1 for a in p) and min(w)>=0
 n=sum(w);assert n>0
 e=sum(a*b for a,b in zip(y,w));negative=n-e;expected=math.fsum(a*b for a,b in zip(p,w))
 # Ascending score blocks count negative records below each positive block.
 blocks={}
 for a,b,c in zip(p,y,w):
  z=blocks.setdefault(a,[0,0]);z[b]+=c
 below=0.;numerator=0.
 for value in sorted(blocks):
  non,pos=blocks[value];numerator+=pos*(below+.5*non);below+=non
 auc=numerator/(e*negative) if e and negative else None
 tp=fp=0;ap_sum=0.
 for value in sorted(blocks,reverse=True):
  non,pos=blocks[value];tp+=pos;fp+=non
  if tp+fp:ap_sum+=pos*tp/(tp+fp)
 out={"n":n,"events":e,"event_rate":e/n,"AUROC":auc,"AP":ap_sum/e if e else None,
      "Brier":math.fsum(b*(a-c)**2 for a,b,c in zip(y,w,p))/n,
      "observed_expected":e/expected if expected else None}
 def capture(prefix,selected):
  reviewed=sum(selected);captured=sum(a*b for a,b in zip(y,selected));false=reviewed-captured;missed=e-captured
  z={"reviewed_n":reviewed,"captured_n":captured,"missed_n":missed,"false_alerts_n":false,
     "sensitivity":captured/e if e else None,"PPV":captured/reviewed if reviewed else None,
     "specificity":1-false/negative if negative else None,"captured_per1000":1000*captured/n,
     "missed_per1000":1000*missed/n,"reviewed_per1000":1000*reviewed/n,"false_alerts_per1000":1000*false/n}
  out.update({prefix+"_"+k:v for k,v in z.items()})
 # Explicit expanded ordering verifies the native weighted capacity shortcut.
 ordered=sorted(((p[i],g[i],r[i],copy,i) for i in range(len(y)) for copy in range(w[i])),
                key=lambda x:(-x[0],x[1],x[2],x[3]))
 for fraction in (.1,.2):
  selected=[0]*len(y)
  for row in ordered[:math.ceil(n*fraction)]:selected[row[-1]]+=1
  capture("capacity"+str(int(fraction*100)),selected)
 if cuts:
  for name in ("training_sensitivity80","training_capacity10","training_capacity20"):
   capture(name,[b if a>=cuts[name] else 0 for a,b in zip(p,w)])
 for t in dca:
  chosen=[a>=t for a in p];tp=sum(a*b*c for a,b,c in zip(y,w,chosen));fp=sum((1-a)*b*c for a,b,c in zip(y,w,chosen))
  out["net_benefit_"+str(t)]=tp/n-fp/n*t/(1-t)
  out["treat_all_net_benefit_"+str(t)]=e/n-(n-e)/n*t/(1-t)
 if include_offset:out["calibration_intercept"]=offset_intercept(y,p,w)
 return out
def selftest():
 checks=0
 y=[0,1,0,1];p=[.1,.4,.4,.9];g=["a","b","c","d"];r=list("1234")
 a=metrics(y,p,g,r,cuts=thresholds(y,p),dca=(.4,),include_offset=True)
 for k,v in {"AUROC":.875,"AP":5/6,"Brier":.135,"capacity10_captured_n":1,
             "capacity20_captured_n":1,"net_benefit_0.4":1/3,
             "training_capacity20_reviewed_n":1}.items():
  assert close(a[k],v), (k,a[k],v);checks+=1
 w=[2,1,0,3];a=metrics(y,p,g,r,w)
 assert close(a["AUROC"],1.) and close(a["AP"],1.);checks+=2
 assert a["n"]==6 and a["events"]==4 and a["capacity20_reviewed_n"]==2;checks+=3
 a=metrics([0,1],[.5,.5],["a","b"],["1","2"])
 assert a["AUROC"]==.5 and a["AP"]==.5 and a["capacity20_captured_n"]==0;checks+=3
 for yy in ([0,0],[1,1]):
  a=metrics(yy,[.1,.8],["a","b"],["1","2"],include_offset=True)
  assert a["AUROC"] is None and a["calibration_intercept"] is None;checks+=2
 assert quantile([0,10],.025)==.25 and quantile([None,float("nan")],.5) is None;checks+=2
 assert thresholds([1,1,1,1,1,0],[.9,.8,.7,.7,.1,.6])=={"training_sensitivity80":.7,"training_capacity10":.9,"training_capacity20":.8};checks+=1
 assert joint_score_residual([0,1,0,1],[.2,.2,.8,.8],[1,1,1,1],0,0)<1e-12;checks+=1
 return {"checks":checks,"fixture":"Hand-solvable synthetic metric, tie, weighted-capacity and undefined-outcome operators only; no clinical experiment"}
if __name__=="__main__":print(selftest())
