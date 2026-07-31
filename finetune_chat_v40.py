#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Instruct fine-tuning AETHER v4.0: delta memory + local attention.
Usa el mismo masking de respuestas, acumulación real, resume, EMA y LoRA.
"""
from __future__ import annotations
import argparse, math, random, time, logging
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from aether_v4 import AetherEngine,KairosTokenizer,amp_autocast,make_grad_scaler,CFG as BASE_CFG,PAD_ID,BOS_ID,EOS_ID,SYS_ID,USR_ID,AST_ID
logging.basicConfig(level=logging.INFO,format="[%(asctime)s] %(levelname)s | %(message)s",datefmt="%H:%M:%S"); log=logging.getLogger("KAIROS-FT40")
SYSTEM="Eres Kairos, una inteligencia artificial de razonamiento lógico, útil y honesta creada por brido."
SYNTH=[
    {"user":"Hola, ¿quién eres?","assistant":"<think>\nIdentificar la identidad de la IA.\n</think>\nHola. Soy Kairos, una inteligencia artificial creada por brido. ¿En qué te ayudo?"},
    {"user":"¿Cuánto es 37 por 3?","assistant":"<think>\nMultiplicar 37 por 3:\n30 * 3 = 90\n7 * 3 = 21\n90 + 21 = 111.\n</think>\nEl resultado de multiplicar 37 por 3 es 111."},
    {"user":"¿Qué puedes hacer?","assistant":"<think>\nListar mis capacidades principales de razonamiento y asistencia.\n</think>\nPuedo razonar paso a paso, explicar conceptos, programar y resolver problemas complejos."}
]

def pairs(n, use_think=True):
    try:
        from datasets import load_dataset
        out=[]
        for repo,ik,ck,ok in [("bertin-project/alpaca-spanish","instruction","input","output"),("databricks/databricks-dolly-15k","instruction","context","response")]:
            try:
                for r in load_dataset(repo,split="train",streaming=True):
                    u=str(r.get(ik) or "").strip(); c=str(r.get(ck) or "").strip(); a=str(r.get(ok) or "").strip()
                    if c:u += "\nContexto:\n"+c
                    if len(u)>2 and len(a)>2:
                        if use_think and not a.startswith("<think>"):
                            a = f"<think>\nComprender la consulta: {u[:60]}...\nFormular respuesta estructurada y lógica.\n</think>\n" + a
                        out.append({"user":u,"assistant":a})
                    if len(out)>=n:break
            except Exception as e:log.warning("%s: %s",repo,e)
            if len(out)>=n:break
        if out:return out
    except Exception as e:log.warning("datasets: %s",e)
    out=[]
    while len(out)<n:out+=SYNTH
    random.shuffle(out);return out[:n]

def encode(tok,r,L):
    p=[BOS_ID,SYS_ID]+tok.encode(SYSTEM)+[USR_ID]+tok.encode(r["user"])+[AST_ID]; a=tok.encode(r["assistant"])+[EOS_ID]; ids=(p+a)[:L]; y=([-100]*len(p)+a)[:L]; n=L-len(ids); return torch.tensor(ids+[PAD_ID]*max(0,n)),torch.tensor(y+[-100]*max(0,n))
class DS(Dataset):
    def __init__(self,rows,tok,L):self.x=[encode(tok,r,L) for r in rows]
    def __len__(self):return len(self.x)
    def __getitem__(self,i):return self.x[i]
class LoRA(nn.Module):
    def __init__(self,b,r=16,a=32):
        super().__init__();self.base=b;self.s=a/r
        for p in b.parameters():p.requires_grad_(False)
        self.A=nn.Parameter(torch.empty(r,b.in_features));self.B=nn.Parameter(torch.zeros(b.out_features,r));nn.init.kaiming_uniform_(self.A,a=math.sqrt(5))
    def forward(self,x):return self.base(x)+(x@self.A.t()@self.B.t())*self.s
def inject(m,r):
    for p in m.parameters():p.requires_grad_(False)
    n=0
    for mod in m.modules():
        for name,ch in list(mod.named_children()):
            if isinstance(ch,nn.Linear) and name in {"qkv","o_proj","w1","w3","k_proj","v_proj","q_proj","out"}:setattr(mod,name,LoRA(ch,r));n+=1
    log.info("LoRA: %d módulos, %.2fM parámetros",n,sum(p.numel() for p in m.parameters() if p.requires_grad)/1e6)
def lr_at(s,total,peak):
    if s<200:return peak*(s+1)/200
    p=min(1,(s-200)/max(1,total-200));return peak*.1+.5*peak*.9*(1+math.cos(math.pi*p))
def save(p,m,o,sc,ema,s,c):
    t=p.with_suffix('.tmp');torch.save({'step':s,'model':m.state_dict(),'optimizer':o.state_dict(),'scaler':sc.state_dict(),'ema':ema,'cfg':c},t);t.replace(p)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base',default='checkpoints/latest.pt');ap.add_argument('--tokenizer',default='kairos_tokenizer.json');ap.add_argument('--out',default='checkpoints_instruct40');ap.add_argument('--steps',type=int,default=2500);ap.add_argument('--batch-size',type=int,default=4);ap.add_argument('--accum',type=int,default=8);ap.add_argument('--seq-len',type=int,default=512);ap.add_argument('--lr',type=float,default=2e-5);ap.add_argument('--max-samples',type=int,default=20000);ap.add_argument('--lora',action='store_true');ap.add_argument('--lora-r',type=int,default=16);ap.add_argument('--resume',action='store_true');ap.add_argument('--no-mtp',action='store_true');args=ap.parse_args()
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');out=Path(args.out);out.mkdir(parents=True,exist_ok=True);latest=out/'latest.pt';tok=KairosTokenizer.load(args.tokenizer);cfg=dict(BASE_CFG);cfg['mtp_weight']=0 if args.no_mtp else BASE_CFG['mtp_weight'];V=((tok.vocab_size+63)//64)*64
    m=AetherEngine(V,cfg).to(dev);ck=torch.load(args.base,map_location=dev,weights_only=False);m.load_state_dict(ck.get('model',ck),strict=False)
    if args.lora:inject(m,args.lora_r)
    train=[p for p in m.parameters() if p.requires_grad];o=torch.optim.AdamW(train,lr=args.lr,betas=(.9,.95),weight_decay=.01);sc=make_grad_scaler(dev.type=='cuda');ema={k:v.detach().float().clone() for k,v in m.state_dict().items() if v.dtype.is_floating_point};start=0
    if args.resume and latest.exists():
        z=torch.load(latest,map_location=dev,weights_only=False);m.load_state_dict(z['model'],strict=False);o.load_state_dict(z['optimizer']);sc.load_state_dict(z['scaler']);ema=z.get('ema',ema);start=z['step']
    dl=DataLoader(DS(pairs(args.max_samples),tok,args.seq_len),batch_size=args.batch_size,shuffle=True,drop_last=True,num_workers=2 if dev.type=='cuda' else 0,pin_memory=dev.type=='cuda',persistent_workers=dev.type=='cuda');it=iter(dl);m.train();t0=time.time()
    for s in range(start,args.steps):
        o.zero_grad(set_to_none=True);loss_sum=0;lr=lr_at(s,args.steps,args.lr)
        for g in o.param_groups:g['lr']=lr
        for _ in range(args.accum):
            try:x,y=next(it)
            except StopIteration:it=iter(dl);x,y=next(it)
            x=x.to(dev,non_blocking=True);y=y.to(dev,non_blocking=True)
            with amp_autocast(enabled=dev.type=='cuda'):
                lg=m(x);loss=F.cross_entropy(lg[:,:-1].float().reshape(-1,V),y[:,1:].reshape(-1),ignore_index=-100)/args.accum
            sc.scale(loss).backward();loss_sum+=float(loss.detach())*args.accum
        sc.unscale_(o);gn=torch.nn.utils.clip_grad_norm_(train,1.0);sc.step(o);sc.update()
        with torch.no_grad():
            for k,v in m.state_dict().items():
        avg_loss = loss_sum / args.accum
        if (s+1)%20==0:log.info('paso %d/%d loss %.4f (acum %.4f) ppl %.1f lr %.2e grad %.2f tok/s %.0f',s+1,args.steps,avg_loss,loss_sum,math.exp(min(avg_loss,20)),lr,float(gn),((s+1)*args.batch_size*args.accum*args.seq_len)/max(1,time.time()-t0))
        if (s+1)%500==0:save(latest,m,o,sc,ema,s+1,vars(args))
    save(latest,m,o,sc,ema,args.steps,vars(args));torch.save(m.state_dict(),out/'model.pt');log.info('v4.0 terminado')
if __name__=='__main__':main()
