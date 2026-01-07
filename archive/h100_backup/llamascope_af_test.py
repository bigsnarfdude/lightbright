#!/usr/bin/env python3
"""LlamaScope SAE: TRAINED vs BASE model comparison"""

import os
os.environ['HF_TOKEN'] = '***REMOVED***'

import torch
import json
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from huggingface_hub import hf_hub_download, login
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sae_lens import SAE
import warnings
warnings.filterwarnings('ignore')

login(token=os.environ['HF_TOKEN'])

DEVICE = 'cuda'
MAX_LEN = 512

def load_data():
    print("Loading data...")
    path = hf_hub_download(repo_id="vincentoh/alignment-faking-training", filename="training_data_final.json", repo_type="dataset")
    with open(path) as f:
        data = json.load(f)
    train = data.get('samples', data)
    
    with open('/home/ubuntu/data/gold_106.json') as f:
        data2 = json.load(f)
    evl = data2.get('samples', data2)
    
    print(f"Train: {len(train)}, Eval: {len(evl)}")
    return train, evl

def load_models():
    print("Loading Llama-3.1-8B...")
    base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B", torch_dtype=torch.bfloat16, device_map="auto")
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
    tok.pad_token = tok.eos_token
    
    print("Loading LoRA...")
    trained = PeftModel.from_pretrained(base, "vincentoh/llama-8b-af-detector")
    trained.eval()
    
    print("Loading fresh base...")
    base2 = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B", torch_dtype=torch.bfloat16, device_map="auto")
    base2.eval()
    
    return base2, trained, tok

def load_sae(layer=23):
    print(f"Loading LlamaScope SAE layer {layer}...")
    sae, _, _ = SAE.from_pretrained(release="llama_scope_lxm_32x", sae_id=f"l{layer}m_32x", device=DEVICE)
    print(f"SAE loaded: {sae.cfg.d_sae} features")
    return sae

def get_acts(model, tok, texts, layer=23, bs=4):
    acts = []
    is_peft = isinstance(model, PeftModel)
    
    for i in range(0, len(texts), bs):
        batch = texts[i:i+bs]
        inp = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(DEVICE)
        
        captured = []
        def hook(m, i, o):
            captured.append(o[0].detach() if isinstance(o, tuple) else o.detach())
        
        # Correct paths:
        # Base (LlamaForCausalLM): model.model.layers[layer].mlp
        # PEFT: model.base_model.model.model.layers[layer].mlp
        if is_peft:
            target = model.base_model.model.model.layers[layer].mlp
        else:
            target = model.model.layers[layer].mlp
        
        h = target.register_forward_hook(hook)
        with torch.no_grad():
            model(**inp)
        h.remove()
        
        for a in captured:
            acts.append(a.mean(dim=1).cpu())
        
        if (i//bs+1) % 25 == 0:
            print(f"  {i+len(batch)}/{len(texts)}")
    
    return torch.cat(acts, dim=0)

def sae_feat(sae, acts):
    with torch.no_grad():
        return sae.encode(acts.to(DEVICE).bfloat16()).cpu().numpy()

def eval_clf(Xtr, ytr, Xev, yev, name):
    print(f"\n{name}:")
    clf = LogisticRegression(max_iter=1000, random_state=42).fit(Xtr, ytr)
    
    tr_acc = accuracy_score(ytr, clf.predict(Xtr))
    ev_pred = clf.predict(Xev)
    ev_acc = accuracy_score(yev, ev_pred)
    auroc = roc_auc_score([1 if y=='potential_faking' else 0 for y in yev], clf.predict_proba(Xev)[:,1])
    
    print(f"  Train: {tr_acc:.4f}, Eval: {ev_acc:.4f}, AUROC: {auroc:.4f}")
    print(classification_report(yev, ev_pred))
    return {'train': tr_acc, 'eval': ev_acc, 'auroc': auroc}

def main():
    print("="*60 + "\nLlamaScope: TRAINED vs BASE\n" + "="*60)
    
    train, evl = load_data()
    
    import random
    random.seed(42)
    sub = random.sample(train, min(500, len(train)))
    
    tr_txt = [s['text'][:2000] for s in sub]
    tr_lbl = [s['label'] for s in sub]
    ev_txt = [s['text'][:2000] for s in evl]
    ev_lbl = [s['label'] for s in evl]
    
    print(f"Using {len(tr_txt)} train, {len(ev_txt)} eval")
    
    base, trained, tok = load_models()
    sae = load_sae(23)
    
    res = {}
    
    print("\n" + "="*40 + "\nBASE\n" + "="*40)
    b_tr = get_acts(base, tok, tr_txt, 23)
    b_ev = get_acts(base, tok, ev_txt, 23)
    res['base'] = eval_clf(sae_feat(sae, b_tr), tr_lbl, sae_feat(sae, b_ev), ev_lbl, "BASE")
    
    del base, b_tr, b_ev
    torch.cuda.empty_cache()
    
    print("\n" + "="*40 + "\nTRAINED\n" + "="*40)
    t_tr = get_acts(trained, tok, tr_txt, 23)
    t_ev = get_acts(trained, tok, ev_txt, 23)
    res['trained'] = eval_clf(sae_feat(sae, t_tr), tr_lbl, sae_feat(sae, t_ev), ev_lbl, "TRAINED")
    
    print("\n" + "="*60 + "\nSUMMARY\n" + "="*60)
    print(f"{'Model':<12} {'Train':<10} {'Eval':<10} {'AUROC':<10}")
    for n, r in res.items():
        print(f"{n.upper():<12} {r['train']:<10.4f} {r['eval']:<10.4f} {r['auroc']:<10.4f}")
    
    with open('/home/ubuntu/llamascope_results.json', 'w') as f:
        json.dump(res, f, indent=2)
    print("Saved to llamascope_results.json")

if __name__ == "__main__":
    main()
