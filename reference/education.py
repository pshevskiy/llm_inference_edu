"""Opt-in independent CPU FP32 explanation probe, not an approved teacher.
Fixed public problems; streams each sampled vocabulary row. Facts and prose
quality are distinct. No production Rust or grading implementation is imported.
"""
import argparse, hashlib, importlib.metadata, json, platform, resource
from pathlib import Path
import agent_qwen as ref

SYSTEM='Explain row-major offsets. Reply with one JSON object only. Fields: row, width, column, offset (integers), formula (exactly "row*width+column"), explanation (one short sentence). Do not use Markdown.'
CASES=[('x-offset','X is row-major [3,4]. Explain the element offset of X[2,1].',2,4,1,9),('w-offset','W is row-major [2,4]. Explain the element offset of W[1,2].',1,4,2,6),('y-offset','Y is row-major [3,2]. Explain the element offset of Y[1,1].',1,2,1,3)]
ACTIVE_OUTPUT=None
def unique_object(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise ValueError('duplicate JSON field')
        result[key]=value
    return result
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',required=True,type=Path);p.add_argument('--output',required=True,type=Path);args=p.parse_args()
    if args.output.exists():p.error('fresh evidence path required')
    inventory=json.loads((ref.ROOT/'fixtures/qwen-inventory.json').read_text())
    for name,entry in inventory['assets'].items():
        path=args.model/name
        if path.stat().st_size!=entry['bytes']:raise ValueError('asset length')
        h=hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
        if h.hexdigest()!=entry['sha256']:raise ValueError('asset hash')
    versions={n:importlib.metadata.version(n) for n in ('torch','transformers','tokenizers','numpy','safetensors')}
    if versions!={'torch':'2.6.0','transformers':'4.46.3','tokenizers':'0.20.3','numpy':'2.3.2','safetensors':'0.6.2'}:raise ValueError('versions')
    args.output.mkdir(parents=True);global ACTIVE_OUTPUT;ACTIVE_OUTPUT=args.output
    manifest={'status':'started','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'helper_sha256':hashlib.sha256(Path(ref.__file__).read_bytes()).hexdigest(),'repository':inventory['repository'],'revision':inventory['revision'],'weight_sha256':inventory['assets']['model.safetensors']['sha256'],'versions':versions,'device':'cpu','dtype':'FP32','threads':1,'capacity':512,'max_new':96,'vocab_size':151936,'atol':0.001,'rtol':0.0001,'host':{'system':platform.system(),'machine':platform.machine(),'version':ref.command(['sw_vers','-productVersion']),'build':ref.command(['sw_vers','-buildVersion']),'chip':ref.command(['sysctl','-n','machdep.cpu.brand_string']),'memory_bytes':ref.command(['sysctl','-n','hw.memsize'])},'cases':[]}
    (args.output/'identity.json').write_text(json.dumps(manifest,indent=2)+'\n')
    ref.torch.set_num_threads(1);ref.torch.set_num_interop_threads(1);ref.torch.set_grad_enabled(False)
    tokenizer=ref.AutoTokenizer.from_pretrained(args.model,local_files_only=True,trust_remote_code=False)
    model=ref.AutoModelForCausalLM.from_pretrained(args.model,local_files_only=True,trust_remote_code=False,torch_dtype=ref.torch.float32,attn_implementation='eager',low_cpu_mem_usage=False).eval().to('cpu')
    stops=set(model.generation_config.eos_token_id)
    for index,(id,question,row,width,column,offset) in enumerate(CASES):
        messages=[{'role':'system','content':SYSTEM},{'role':'user','content':question}]
        text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True);prompt=tokenizer.encode(text,add_special_tokens=False)
        if len(prompt)+96>512:raise ValueError('context admission')
        output=model(input_ids=ref.torch.tensor([prompt]),use_cache=True,num_logits_to_keep=1);cache=output.past_key_values
        generated=[];h=hashlib.sha256();file=f'case{index}.bin'
        with (args.output/file).open('wb') as f:
            for _ in range(96):
                logits=output.logits[0,-1].numpy().astype('<f4',copy=True)
                if not ref.np.isfinite(logits).all():raise ValueError('nonfinite logits')
                data=logits.tobytes();f.write(data);h.update(data)
                token=int(ref.np.argmax(logits));generated.append(token)
                if token in stops or len(generated)==96:break
                output=model(input_ids=ref.torch.tensor([[token]]),past_key_values=cache,use_cache=True,num_logits_to_keep=1);cache=output.past_key_values
        decoded=tokenizer.decode(generated,skip_special_tokens=True);factual=False
        try:
            data=json.loads(decoded,object_pairs_hook=unique_object)
            factual=(len(decoded.encode())<=4096 and set(data)=={'row','width','column','offset','formula','explanation'} and all(type(data[k]) is int for k in ('row','width','column','offset')) and [data['row'],data['width'],data['column'],data['offset']]==[row,width,column,offset] and data['formula']=='row*width+column' and isinstance(data['explanation'],str) and bool(data['explanation'].strip()) and len(data['explanation'].encode())<=512)
        except (ValueError,TypeError,KeyError):pass
        factual=factual and generated[-1] in stops
        case={'id':id,'prompt_ids':prompt,'generated_ids':generated,'decoded':decoded,'file':file,'bytes':len(generated)*151936*4,'sha256':h.hexdigest(),'fact_gate':factual,'ended_eos':generated[-1] in stops,'prose_review':'pending; numeric facts do not approve prose'}
        manifest['cases'].append(case);print(id,repr(decoded),'fact gate',factual,flush=True)
        (args.output/'progress.json').write_text(json.dumps(manifest,indent=2)+'\n');del output,cache
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;manifest['peak_rss_bytes']=peak if platform.system()=='Darwin' else peak*1024
    if manifest['peak_rss_bytes']>6*1024**3:raise ValueError('reference RSS exceeds6GiB')
    manifest['status']='reference_complete';(args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');(args.output/'completion.json').write_text('{"status":"reference_complete"}\n')
if __name__=='__main__':
    try:main()
    except BaseException as e:
        if ACTIVE_OUTPUT is not None:(ACTIVE_OUTPUT/'completion.json').write_text(json.dumps({'status':'failed','error':str(e)})+'\n')
        raise
