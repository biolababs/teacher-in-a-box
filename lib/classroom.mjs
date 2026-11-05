import { createHash } from 'node:crypto';

export const SYSTEM = `You control a robot on a 5 by 5 board. Coordinates are [x,y], x eastward, y southward, from 0 to 4. Return only a JSON array of actions. Actions: {"op":"move","direction":"north|south|east|west"}, {"op":"pick","block":"red|blue"}, {"op":"drop"}. A move is one cell. Walls cannot be entered. You may stand on a block's cell; pick requires standing on it with empty hands. Drop requires carrying a block and a cell without another block. Remain inside the board. At most 100 actions. Satisfy every goal and ordering rule. No explanation outside the JSON.`;
export const SKILLS = { transport: 'Pick & place', obstacles: 'Route around walls', sequence: 'Respect the order' };
const DIRS = { north: [0,-1], east: [1,0], south: [0,1], west: [-1,0] };
const same = (a,b) => a[0] === b[0] && a[1] === b[1];
export const fingerprint = value => createHash('sha256').update(JSON.stringify(value)).digest('hex');

export function simulate(task, actions) {
  const state = structuredClone(task.initial);
  state.carrying = null;
  const frames = [structuredClone(state)];
  let error = null;
  if (!Array.isArray(actions) || actions.length > 100) return {passed:false, error:'Expected an array of at most 100 actions.', frames, steps:0};
  for (const [i,a] of actions.entries()) {
    try {
      if (!a || typeof a !== 'object' || Array.isArray(a)) throw Error('Invalid action.');
      if (a.op === 'move') {
        if (!DIRS[a.direction] || Object.keys(a).some(k=>!['op','direction'].includes(k))) throw Error('Invalid direction or extra action fields.');
        const d=DIRS[a.direction], p=[state.robot[0]+d[0],state.robot[1]+d[1]];
        if (p.some(v=>v<0||v>4)) throw Error('Robot left the board.');
        if (state.walls.some(w=>same(w,p))) throw Error('Robot hit a wall.');
        state.robot=p;
      } else if (a.op === 'pick') {
        if (Object.keys(a).some(k=>!['op','block'].includes(k)) || !Object.hasOwn(state.blocks,a.block)) throw Error('Unknown block or extra action fields.');
        if (state.carrying || !state.blocks[a.block] || !same(state.robot,state.blocks[a.block])) throw Error('Stand on the block with empty hands before picking it up.');
        if (task.redFirst && a.block==='blue' && !same(state.blocks.red||[-1,-1],task.goals.red)) throw Error('Place red at its goal before picking up blue.');
        state.carrying=a.block; state.blocks[a.block]=null;
      } else if (a.op === 'drop') {
        if (Object.keys(a).some(k=>k!=='op')) throw Error('Extra action fields.');
        if (!state.carrying) throw Error('Nothing to drop.');
        if (Object.values(state.blocks).some(p=>p&&same(p,state.robot))) throw Error('Another block occupies this cell.');
        state.blocks[state.carrying]=[...state.robot]; state.carrying=null;
      } else throw Error('Unknown operation.');
      frames.push(structuredClone(state));
    } catch(e) { error=`Action ${i+1}: ${e.message}`; break; }
  }
  const goalsMet=Object.entries(task.goals).every(([color,p])=>state.blocks[color]&&same(state.blocks[color],p));
  return {passed:!error&&goalsMet&&!state.carrying,error:error||(!goalsMet?'Goal not reached.':state.carrying?'Still carrying a block.':null),frames,steps:frames.length-1};
}

export function parsePlan(raw) {
  const cleaned=String(raw).trim().replace(/^```(?:json)?\s*/,'').replace(/\s*```$/,'');
  const actions=JSON.parse(cleaned);
  if(!Array.isArray(actions)) throw Error('Model must return a JSON array.');
  return actions;
}

function path(start,end,walls) {
  const queue=[[start,[]]], seen=new Set([start.join(',')]);
  for(let i=0;i<queue.length;i++) {
    const [p,actions]=queue[i]; if(same(p,end)) return actions;
    for(const [direction,d] of Object.entries(DIRS)) {
      const next=[p[0]+d[0],p[1]+d[1]], key=next.join(',');
      if(next.some(v=>v<0||v>4)||seen.has(key)||walls.some(w=>same(w,next))) continue;
      seen.add(key); queue.push([next,[...actions,{op:'move',direction}]]);
    }
  }
  throw Error('Unreachable goal.');
}

export function referencePlan(task) {
  let robot=task.initial.robot; const actions=[];
  for(const [color,dest] of Object.entries(task.goals)) {
    if(same(task.initial.blocks[color],dest)) continue;
    actions.push(...path(robot,task.initial.blocks[color],task.initial.walls),{op:'pick',block:color},...path(task.initial.blocks[color],dest,task.initial.walls),{op:'drop'});
    robot=dest;
  }
  return actions;
}

function rng(seed) { return ()=> { seed|=0; seed=seed+0x6D2B79F5|0; let t=Math.imul(seed^seed>>>15,1|seed); t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296; }; }
function makeTask(split,index,skill,seed) {
  const random=rng(seed), cells=Array.from({length:25},(_,i)=>[i%5,Math.floor(i/5)]);
  for(let i=cells.length-1;i>0;i--) {const j=Math.floor(random()*(i+1)); [cells[i],cells[j]]=[cells[j],cells[i]];}
  const [robot,red,blue,redGoal,blueGoal,...rest]=cells;
  const initial={robot,blocks:{red,blue},walls:skill==='transport'?[]:rest.slice(0,3)};
  const goals=skill==='sequence'?{red:redGoal,blue:blueGoal}:{red:redGoal};
  const task={id:`${split}-${index}`,split,skill,initial,goals,redFirst:skill==='sequence'};
  task.instruction=skill==='sequence'?`Move red to [${redGoal}], then move blue to [${blueGoal}]. Red must be placed before blue is picked up.`:`Move the red block to [${redGoal}]. ${skill==='obstacles'?'Route around the walls.':'Leave the blue block where it is.'}`;
  if(skill!=='sequence') task.goals.blue=blue;
  return task;
}

// Fixed seeds and disjoint scenes; exam tasks are never used for lessons or exports.
export const TASKS=[];
const scenes=new Set();
for(const [split,seed,count] of [['train',1200,30],['diagnostic',7200,9],['exam',17200,12]]) {
  for(let i=0;i<count;i++) {
    let task;
    for(let attempt=0;attempt<100;attempt++) {
      task=makeTask(split,i,Object.keys(SKILLS)[i%3],seed+i*101+attempt);
      const key=fingerprint({initial:task.initial,goals:task.goals});
      try { if(!scenes.has(key)&&simulate(task,referencePlan(task)).passed) {scenes.add(key); break;} task=null; } catch {task=null;}
    }
    if(!task) throw Error('Could not generate valid curriculum.');
    TASKS.push(task);
  }
}
export function prompt(task) { return `${task.instruction}\nInitial board: ${JSON.stringify(task.initial)}\nGoals: ${JSON.stringify(task.goals)}\nRed must be placed before picking blue: ${task.redFirst}`; }
export function messages(task,mode='zero-shot') {
  const msgs=[{role:'system',content:SYSTEM}];
  if(mode==='few-shot') for(const sample of TASKS.filter(t=>t.split==='train').slice(0,3)) msgs.push({role:'user',content:prompt(sample)},{role:'assistant',content:JSON.stringify(referencePlan(sample))});
  msgs.push({role:'user',content:prompt(task)}); return msgs;
}
