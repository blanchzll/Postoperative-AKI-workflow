const fs=require("fs"),cp=require("child_process"),path=require("path");
const [base,out]=process.argv.slice(2);let seq=0,buffer="",session;const pending=new Map();let stderr="";
const chrome=cp.spawn("/usr/bin/google-chrome",["--headless=new","--no-sandbox","--disable-gpu","--disable-dev-shm-usage","--remote-debugging-pipe","--no-first-run","--user-data-dir="+path.join(out,"qa_chrome_profile"),"--window-size=1280,1000","--lang=zh-CN"],{stdio:["ignore","pipe","pipe","pipe","pipe"]});
chrome.stderr.on("data",x=>{stderr=(stderr+x.toString()).slice(-12000)});
chrome.stdio[4].on("data",x=>{buffer+=x.toString();let i;while((i=buffer.indexOf("\0"))>=0){let raw=buffer.slice(0,i);buffer=buffer.slice(i+1);if(!raw)continue;const q=JSON.parse(raw);if(q.id&&pending.has(q.id)){let [resolve,reject]=pending.get(q.id);pending.delete(q.id);q.error?reject(Error(JSON.stringify(q.error))):resolve(q.result)}}});
chrome.on("exit",code=>{for(const [resolve,reject] of pending.values())reject(Error("Chrome exited "+code+" "+stderr.slice(-1000)));pending.clear()});
function send(method,params={},scoped=true){return new Promise((resolve,reject)=>{let id=++seq;pending.set(id,[resolve,reject]);let q={id,method,params};if(session&&scoped)q.sessionId=session;chrome.stdio[3].write(JSON.stringify(q)+"\0");setTimeout(()=>{if(pending.has(id)){pending.delete(id);reject(Error("CDP timeout "+method))}},30000).unref()})}
async function evaluate(expression){let q=await send("Runtime.evaluate",{expression,awaitPromise:true,returnByValue:true});if(q.exceptionDetails)throw Error(JSON.stringify(q.exceptionDetails));return q.result.value}
async function until(expr){for(let i=0;i<120;i++){if(await evaluate(expr))return;await new Promise(r=>setTimeout(r,100))}throw Error("Browser condition timeout "+expr)}
(async()=>{let report={status:"RUNNING",browser:"Chrome headless via native CDP pipe",checks:[]};
try{
 const version=await send("Browser.getVersion",{},false);report.browser_version=version.product;
 const target=await send("Target.createTarget",{url:base},false);let a=await send("Target.attachToTarget",{targetId:target.targetId,flatten:true},false);session=a.sessionId;
 await send("Page.enable");await send("Runtime.enable");await until("document.querySelectorAll('#fields .field').length===9");
 report.checks.push("Initial compact route renders nine labelled fields");
 await evaluate("document.getElementById('example').click();document.getElementById('form').requestSubmit();");
 await until("!document.getElementById('result').hidden");
 report.compact=await evaluate("({risk:last.predictions[0].risk,route:last.route,display:document.getElementById('risk').textContent,missing:last.predictions[0].missing_fields.length})");
 let ss=await send("Page.captureScreenshot",{format:"png",captureBeyondViewport:true});fs.writeFileSync(path.join(out,"UI_DESKTOP.png"),Buffer.from(ss.data,"base64"));report.checks.push("Actual form submission returns compact model risk");
 await evaluate("document.getElementById('route').value='temporal_paired_endop_full';document.getElementById('route').dispatchEvent(new Event('change'));");
 await until("document.querySelectorAll('#fields .field').length===38");
 if(!(await evaluate("document.getElementById('result').hidden")))throw Error("Stale result remained after route change");
 await evaluate("document.getElementById('example').click();document.getElementById('form').requestSubmit();");await until("!document.getElementById('result').hidden");
 report.endop=await evaluate("({risk:last.predictions[0].risk,route:last.route,fields:document.querySelectorAll('#fields .field').length})");report.checks.push("38-input route renders and calculates; route change clears old result");
 await evaluate("document.getElementById('route').value='geography_preop_preop_compact';document.getElementById('route').dispatchEvent(new Event('change'));document.getElementById('example').click();document.getElementById('f_PreopAlb').value='';document.getElementById('form').requestSubmit();");await until("!document.getElementById('result').hidden");
 const misses=await evaluate("last.predictions[0].missing_fields");if(!misses.includes("PreopAlb"))throw Error("Missing field was not carried to source imputer");
 report.checks.push("Blank albumin uses source imputation and is disclosed");
 await send("Emulation.setDeviceMetricsOverride",{width:390,height:844,deviceScaleFactor:1,mobile:true});
 await evaluate("scrollTo(0,0)");ss=await send("Page.captureScreenshot",{format:"png",captureBeyondViewport:true});fs.writeFileSync(path.join(out,"UI_MOBILE.png"),Buffer.from(ss.data,"base64"));
 const overflow=await evaluate("document.documentElement.scrollWidth>innerWidth+1");if(overflow)throw Error("Mobile horizontal overflow");
 report.checks.push("390px mobile layout has no horizontal overflow");
 await evaluate("document.getElementById('case').value='invalid key with spaces';document.getElementById('form').requestSubmit();");await until("!document.getElementById('error').hidden");
 report.checks.push("Invalid anonymous key shows an error without stale risk");
 report.status="PASS_actual_browser_interaction_and_screenshots";
 fs.writeFileSync(path.join(out,"BROWSER_ACCEPTANCE.json"),JSON.stringify(report,null,2)+"\n");
 await send("Browser.close",{},false);
}catch(e){report.status="ERROR";report.error=String(e);try{report.DOM_diagnostic=await evaluate("({url:location.href,title:document.title,ready:document.readyState,body:document.body?.innerText?.slice(0,5000),fields:document.querySelectorAll(\"#fields .field\").length})");const shot=await send("Page.captureScreenshot",{format:"png"});fs.writeFileSync(path.join(out,"UI_FAILURE_DIAGNOSTIC.png"),Buffer.from(shot.data,"base64"));}catch(diag){report.diagnostic_error=String(diag);}report.chrome_stderr_tail=stderr.slice(-2000);fs.writeFileSync(path.join(out,"BROWSER_ACCEPTANCE.json"),JSON.stringify(report,null,2)+"\n");chrome.kill("SIGTERM");process.exitCode=1;}
})();
