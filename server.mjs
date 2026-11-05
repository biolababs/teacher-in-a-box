import http from 'node:http';
import {readFile,writeFile,mkdir,rename,stat} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {randomUUID} from 'node:crypto';
import {spawn} from 'node:child_process';
import {TASKS,SKILLS,SYSTEM,messages,prompt,parsePlan,simulate,referencePlan,fingerprint} from './lib/classroom.mjs';
import {localEndpoint,call,completion} from './lib/lemonade.mjs';

const ROOT=path.dirname(fileURLToPath(import.meta.url));
const DATA=path.join(ROOT,'.local');
await mkdir(DATA,{recursive:true});
const empty={settings:{endpoint:'http://127.0.0.1:13305/api/v1'},lessons:[],runs:[],exports:[]};
let state;
try {state=JSON.parse(await readFile(path.join(DATA,'classroom.json'),'utf8'));} catch(e) {if(e.code!=='ENOENT') throw e; state=empty;}
let writes=Promise.resolve();
function persist() {const snapshot=JSON.stringify(state,null,2); writes=writes.catch(()=>{}).then(async()=>{await writeFile(path.join(DATA,'classroom.tmp'),snapshot);await rename(path.join(DATA,'classroom.tmp'),path.join(DATA,'classroom.json'));});return writes;}
let job=null, controller=null;
const PUBLIC=path.join(ROOT,'public');
const publicTasks=TASKS.filter(t=>t.split==='diagnostic');
const fail=(message,status=400)=>Object.assign(new Error(message),{status});
async function body(req) {let data='';for await(const chunk of req){data+=chunk;if(Buffer.byteLength(data)>256000) throw fail('Request too large.',413);}try{return JSON.parse(data||'{}');}catch{throw fail('Invalid JSON.');}}
function json(res,value,status=200) {res.writeHead(status,{'Content-Type':'application/json','Cache-Control':'no-store'});res.end(JSON.stringify(value));}
function begin(type,fn) {
  if(job?.status==='running') throw fail('Another local job is running.',409);
  controller=new AbortController();const signal=controller.signal;
  job={id:randomUUID(),type,status:'running',progress:0,total:0,log:[],started:new Date().toISOString()};const active=job;
  Promise.resolve().then(()=>fn(active,signal)).then(()=>{active.status='complete';}).catch(e=>{active.status=signal.aborted?'cancelled':'failed';active.error=e.message;}).finally(()=>{active.finished=new Date().toISOString();});
  return job;
}
async function requireModel(id) {if(typeof id!=='string'||!id) throw fail('Choose a downloaded local model.');const list=await call(state.settings.endpoint,'models',{timeout:8000});if(!list.data?.some(m=>m.id===id))throw fail('Model is not available locally. Download it in Lemonade first.');}
function summary(run) {const skills={};for(const skill of Object.keys(SKILLS)){const rows=run.results.filter(r=>r.skill===skill);skills[skill]={passed:rows.filter(r=>r.passed).length,total:rows.length};}return {passed:run.results.filter(r=>r.passed).length,total:run.results.length,skills};}
function safeId(value) {if(typeof value!=='string'||!/^[a-zA-Z0-9-]+$/.test(value)) throw fail('Invalid identifier.');return value;}

async function api(req,res,url) {
  const key=`${req.method} ${url.pathname}`;
  if(key==='GET /api/state') return json(res,{...state,tasks:publicTasks,skills:SKILLS,job,system:SYSTEM});
  if(key==='GET /api/connection') {
    try {const [health,models,system]=await Promise.all([call(state.settings.endpoint,'health',{timeout:5000}),call(state.settings.endpoint,'models',{timeout:5000}),call(state.settings.endpoint,'system-info',{timeout:5000}).catch(()=>null)]);return json(res,{connected:true,version:health.version,models:models.data||[],hardware:system?{cpu:system.Processor,memory:system['Physical Memory'],devices:system.devices}:null});}
    catch(e) {return json(res,{connected:false,error:e.message,models:[]});}
  }
  if(key==='POST /api/settings') {const b=await body(req);state.settings.endpoint=localEndpoint(b.endpoint);await persist();return json(res,state.settings);}
  if(key==='POST /api/simulate') {const b=await body(req),task=publicTasks.find(t=>t.id===b.taskId);if(!task)throw fail('Choose a diagnostic task.');const actions=b.reference?referencePlan(task):parsePlan(b.plan);return json(res,{...simulate(task,actions),actions,source:b.reference?'Reference solver (not AI)':'Your plan'});}
  if(key==='POST /api/evaluate') {
    const b=await body(req);await requireModel(b.model);
    if(!['diagnostic','exam'].includes(b.split)||!['zero-shot','few-shot'].includes(b.mode))throw fail('Invalid evaluation configuration.');
    const base=state.settings.endpoint;
    return json(res,begin('evaluation',async(active,signal)=>{
      const tasks=TASKS.filter(t=>t.split===b.split);active.total=tasks.length;
      const run={id:active.id,model:b.model,endpoint:base,split:b.split,mode:b.mode,created:active.started,results:[],suiteHash:fingerprint(tasks),systemHash:fingerprint(SYSTEM),settings:{temperature:0,max_tokens:2048},complete:false};
      try {for(const task of tasks) {
        signal.throwIfAborted();const start=performance.now();let result;
        try {const generated=await completion(base,b.model,messages(task,b.mode),signal);let sim;try{sim=simulate(task,parsePlan(generated.text));}catch(e){sim={passed:false,error:e.message,steps:0};}
          result={taskId:task.id,skill:task.skill,...generated,passed:sim.passed,error:sim.error,steps:sim.steps};
        }catch(e){if(signal.aborted)throw e;result={taskId:task.id,skill:task.skill,passed:false,error:e.message,elapsedMs:Math.round(performance.now()-start),transportError:true};}
        run.results.push(result);active.progress++;active.log.push(`${task.id}: ${result.passed?'PASS':result.error}`);
      }run.complete=true;}finally{run.summary=summary(run);state.runs.push(run);await persist();active.runId=run.id;}
    }),202);
  }
  if(key==='POST /api/lessons') {
    const b=await body(req);await requireModel(b.model);
    const latest=[...state.runs].reverse().find(r=>r.split==='diagnostic'&&r.complete);
    const scores=latest?.summary.skills;
    const weakest=Object.keys(SKILLS).sort((a,b)=>(scores? scores[a].passed/scores[a].total-scores[b].passed/scores[b].total:0))[0];
    const base=state.settings.endpoint;
    return json(res,begin('lessons',async(active,signal)=>{
      const tasks=TASKS.filter(t=>t.split==='train'&&t.skill===weakest&&!state.lessons.some(l=>l.taskId===t.id)).slice(0,6);active.total=tasks.length;
      if(!tasks.length)throw Error('All lessons for this skill have already been generated. Review the existing lessons.');
      for(const task of tasks){signal.throwIfAborted();const answer=await completion(base,b.model,messages(task),signal);let actions=null,check;
        try{actions=parsePlan(answer.text);check=simulate(task,actions);}catch(e){check={passed:false,error:e.message};}
        state.lessons.push({id:randomUUID(),taskId:task.id,skill:task.skill,instruction:task.instruction,prompt:prompt(task),actions,raw:answer.text,valid:check.passed,error:check.error,approved:false,teacher:b.model,created:new Date().toISOString()});await persist();active.progress++;active.log.push(`${task.id}: ${check.passed?'Validated — ready for review':check.error}`);
      }
    }),202);
  }
  if(key==='POST /api/seed-lessons') {
    if(job?.status==='running')throw fail('Wait for the current job.',409);
    let added=0;for(const task of TASKS.filter(t=>t.split==='train')) {if(state.lessons.some(l=>l.taskId===t.id))continue;
      state.lessons.push({id:randomUUID(),taskId:task.id,skill:task.skill,instruction:task.instruction,prompt:prompt(task),actions:referencePlan(task),valid:true,error:null,approved:false,teacher:'Deterministic solver — not AI-generated',created:new Date().toISOString()});added++;
    }await persist();return json(res,{added});
  }
  if(key==='POST /api/approve') {const b=await body(req),lesson=state.lessons.find(l=>l.id===b.id);if(!lesson)throw fail('Lesson not found.',404);if(!lesson.valid)throw fail('Invalid lessons cannot be approved.');lesson.approved=b.approved===true;await persist();return json(res,lesson);}
  if(key==='POST /api/export') {
    const lessons=state.lessons.filter(l=>l.valid&&l.approved);if(!lessons.length)throw fail('Review and approve at least one lesson first.');
    const id=randomUUID(),folder=path.join(DATA,'exports',id);await mkdir(folder,{recursive:true});
    const rows=lessons.map(l=>({messages:[{role:'system',content:SYSTEM},{role:'user',content:l.prompt},{role:'assistant',content:JSON.stringify(l.actions)}],metadata:{taskId:l.taskId,teacher:l.teacher,lessonId:l.id}}));
    const data=rows.map(r=>JSON.stringify(r)).join('\n')+'\n';await writeFile(path.join(folder,'train.jsonl'),data);
    const manifest={id,created:new Date().toISOString(),examples:rows.length,sha256:fingerprint(data),systemHash:fingerprint(SYSTEM),split:'train only',notes:'No diagnostic or exam examples included. Keep exam results out of lesson generation.'};
    await writeFile(path.join(folder,'manifest.json'),JSON.stringify(manifest,null,2));state.exports.push(manifest);await persist();return json(res,manifest);
  }
  if(key==='POST /api/train') {
    const b=await body(req),id=safeId(b.exportId);if(!state.exports.some(e=>e.id===id))throw fail('Export reviewed lessons first.');
    if(typeof b.modelPath!=='string'||!path.isAbsolute(b.modelPath))throw fail('Choose an absolute local Hugging Face model folder (not a GGUF file).');
    if(!(await stat(b.modelPath)).isDirectory())throw fail('Model path must be a folder.');
    const steps=Number(b.steps);if(!Number.isInteger(steps)||steps<1||steps>500)throw fail('Choose between 1 and 500 training steps.');
    return json(res,begin('training',async(active,signal)=>{
      active.total=steps;const output=path.join(DATA,'training',active.id);await mkdir(output,{recursive:true});
      active.output=output;
      const python=process.env.TEACHER_PYTHON||'python';
      await new Promise((resolve,reject)=>{
        const child=spawn(python,['-s',path.join(ROOT,'training','train.py'),'--model',b.modelPath,'--data',path.join(DATA,'exports',id,'train.jsonl'),'--output',output,'--steps',String(steps)],{cwd:ROOT,windowsHide:true,env:{...process.env,HF_HUB_OFFLINE:'1',TRANSFORMERS_OFFLINE:'1',HF_HUB_DISABLE_TELEMETRY:'1',WANDB_DISABLED:'true',PYTHONUNBUFFERED:'1'}});
        const abort=()=>child.kill();signal.addEventListener('abort',abort,{once:true});if(signal.aborted)abort();
        let pending='';function consume(chunk){pending+=chunk.toString();const lines=pending.split(/\r?\n/);pending=lines.pop();for(const line of lines){active.log.push(line);const m=line.match(/STEP (\d+)\//);if(m)active.progress=Number(m[1]);}active.log=active.log.slice(-200);}
        child.stdout.on('data',consume);child.stderr.on('data',consume);child.on('error',reject);child.on('close',code=>{signal.removeEventListener('abort',abort);if(pending)active.log.push(pending);code===0?resolve():reject(Error(`Local trainer exited with code ${code}. See the training log.`));});
      });
    }),202);
  }
  if(key==='POST /api/cancel') {controller?.abort();return json(res,{requested:true});}
  if(req.method==='GET'&&url.pathname.startsWith('/api/download/')) {
    const parts=url.pathname.split('/'),id=safeId(parts[3]),name=parts[4];if(!['train.jsonl','manifest.json'].includes(name))throw fail('Unknown export file.');
    if(!state.exports.some(e=>e.id===id))throw fail('Export not found.',404);
    const contents=await readFile(path.join(DATA,'exports',id,name));res.writeHead(200,{'Content-Type':'application/octet-stream','Content-Disposition':`attachment; filename="${name}"`});return res.end(contents);
  }
  if(key==='GET /api/report') {res.writeHead(200,{'Content-Type':'application/json','Content-Disposition':'attachment; filename="classroom-report.json"'});return res.end(JSON.stringify({generated:new Date().toISOString(),runs:state.runs},null,2));}
  throw fail('Not found.',404);
}

export const server=http.createServer(async(req,res)=>{
  try {
    const host=req.headers.host||'';
    if(!/^(127\.0\.0\.1|localhost|\[::1\])(:\d+)?$/.test(host))throw fail('Loopback Host required.',403);
    if(req.headers.origin && ![`http://${host}`].includes(req.headers.origin))throw fail('Cross-origin requests are not allowed.',403);
    res.setHeader('X-Content-Type-Options','nosniff');res.setHeader('Content-Security-Policy',"default-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'");
    const url=new URL(req.url,`http://${host}`);
    if(url.pathname.startsWith('/api/')) {
      if(req.method==='POST'&&!req.headers['content-type']?.startsWith('application/json'))throw fail('JSON content type required.',415);
      return await api(req,res,url);
    }
    if(req.method!=='GET')throw fail('Method not allowed.',405);
    const allowed={'/':'index.html','/app.js':'app.js','/style.css':'style.css'};
    const name=allowed[url.pathname];if(!name)throw fail('Not found.',404);
    res.writeHead(200,{'Content-Type':name.endsWith('.css')?'text/css':name.endsWith('.js')?'text/javascript':'text/html','Cache-Control':'no-cache'});res.end(await readFile(path.join(PUBLIC,name)));
  } catch(e){json(res,{error:e.message},e.status||500);}
});
if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url))server.listen(Number(process.env.PORT||4173),'127.0.0.1',()=>console.log(`Teacher in a Box → http://127.0.0.1:${server.address().port}\nLocal data → ${DATA}`));
