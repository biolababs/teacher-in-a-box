export function localEndpoint(value='http://127.0.0.1:13305/api/v1') {
  const u=new URL(value);
  if(u.protocol!=='http:' || !['127.0.0.1','localhost','[::1]'].includes(u.hostname) || u.username || u.password || u.search || u.hash) throw Error('Use an HTTP loopback Lemonade URL, such as http://127.0.0.1:13305/api/v1.');
  // Normalize localhost to a literal loopback address; never resolve an arbitrary hostname.
  if(u.hostname==='localhost') u.hostname='127.0.0.1';
  return u.href.replace(/\/$/,'');
}
export async function call(base,route,{body,signal,timeout=120000}={}) {
  const response=await fetch(`${localEndpoint(base)}/${route}`,{method:body?'POST':'GET',redirect:'error',headers:{'Content-Type':'application/json'},body:body?JSON.stringify(body):undefined,signal:signal?AbortSignal.any([signal,AbortSignal.timeout(timeout)]):AbortSignal.timeout(timeout)});
  if(!response.ok) throw Error(`Lemonade ${response.status}: ${(await response.text()).slice(0,500)}`);
  return response.json();
}
export async function completion(base,model,messages,signal) {
  const started=performance.now();
  const data=await call(base,'chat/completions',{body:{model,messages,temperature:0,max_tokens:2048,stream:false},signal});
  const text=data.choices?.[0]?.message?.content;
  if(typeof text!=='string') throw Error('Lemonade returned no text completion.');
  return {text,elapsedMs:Math.round(performance.now()-started),usage:data.usage||null};
}
