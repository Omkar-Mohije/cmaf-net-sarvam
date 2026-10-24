"""Train CMAF-Net from manifest CSV: image,audio,text,label,split (train/val/test)."""
import argparse
from pathlib import Path
import json
import pandas as pd
import numpy as np
import soundfile as sf
import torch
from torch import nn
from sklearn.metrics import accuracy_score, f1_score
from cmaf.features import image_features, text_features, audio_features, missing_feature, collate_feature_rows, FEATURE_DIMS
from cmaf.model import CMAFNet, ConcatBaseline, modality_drop
from PIL import Image


def read_manifest(path):
    df = pd.read_csv(path).fillna('')
    required = {'image','audio','text','label','split'}
    if not required.issubset(df.columns):
        raise ValueError(f'Manifest needs columns {sorted(required)}')
    if not {'train','val','test'}.issubset(set(df.split)):
        raise ValueError('Provide disjoint train, val, test splits')
    if df.label.nunique() < 2:
        raise ValueError('Need at least two classes')
    return df

def extract(df, root, cache_path):
    if cache_path.exists():
        return torch.load(cache_path, map_location='cpu', weights_only=False)
    labels = sorted(str(s) for s in df.label.unique())
    rows=[]
    for k, record in enumerate(df.itertuples(index=False)):
        rec = record._asdict()
        paths = [root / str(rec[c]) if str(rec[c]).strip() else None for c in ('image','audio')]
        mask = [p is not None and p.is_file() for p in paths] + [bool(str(rec['text']).strip())]
        if not any(mask):
            raise ValueError(f'Row {k} has no available modalities')
        im = image_features(Image.open(paths[0]).convert('RGB')) if mask[0] else missing_feature(FEATURE_DIMS[0])
        tx = text_features(str(rec['text'])) if mask[2] else missing_feature(FEATURE_DIMS[1])
        if mask[1]:
            a,sr=sf.read(paths[1],dtype='float32')
            au=audio_features(a,sr)
        else:
            au=missing_feature(FEATURE_DIMS[2])
        rows.append(([im,tx,au], [mask[0],mask[2],mask[1]], labels.index(str(rec['label'])), str(rec['split'])))
        print(f'features {k+1}/{len(df)}', end='\r', flush=True)
    cache={'rows':rows,'labels':labels}
    cache_path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(cache,cache_path)
    return cache

def batches(rows, size, shuffle=False):
    indices = np.arange(len(rows))
    if shuffle: np.random.shuffle(indices)
    for start in range(0,len(indices),size):
        sample=[rows[int(i)] for i in indices[start:start+size]]
        yield collate_feature_rows([(f,m,y) for f,m,y,_ in sample])

def forward_logits(model, feats, masks, typ):
    out = model(feats, masks)
    return out[0] if typ=='cmaf' else out

def evaluate(model,rows,typ,device,batch_size,drop_idx=None):
    model.eval()
    ys=[];pred=[]
    with torch.no_grad():
        for features,masks,labels in batches(rows,batch_size):
            features=[x.to(device) for x in features]
            masks=masks.to(device)
            if drop_idx is not None: masks[:,drop_idx]=False
            valid=masks.any(dim=1)
            if not valid.any():continue
            features=[x[valid] for x in features]
            logits=forward_logits(model, features, masks[valid], typ)
            ys.extend(labels[valid.cpu()].tolist());pred.extend(logits.argmax(1).cpu().tolist())
    if not ys:return {'accuracy':None,'macro_f1':None,'n':0}
    return {'accuracy':round(accuracy_score(ys,pred),4), 'macro_f1':round(f1_score(ys,pred,average='macro',zero_division=0),4),'n':len(ys)}

def fit(model,train,val,typ,device,epochs,batch_size,lr,drop):
    model.to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-3)
    best=-1.;state=None
    for epoch in range(epochs):
        model.train();losses=[]
        for feats,masks,labels in batches(train,batch_size,shuffle=True):
            feats=[f.to(device) for f in feats];masks=masks.to(device);labels=labels.to(device)
            if drop: masks=modality_drop(masks,p=.25)
            opt.zero_grad()
            logits=forward_logits(model,feats,masks,typ)
            loss=nn.functional.cross_entropy(logits,labels)
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step();losses.append(loss.item())
        metric=evaluate(model,val,typ,device,batch_size)
        print(f'{typ} epoch {epoch+1}: loss={np.mean(losses):.4f} val={metric}')
        score=metric['macro_f1'] if metric['macro_f1'] is not None else -1
        if score>best:
            best=score;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(state)
    return model

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--epochs',type=int,default=8)
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--lr',type=float,default=0.0005)
    parser.add_argument('--out',default='artifacts')
    parser.add_argument('--cache',default='artifacts/features.pt')
    args=parser.parse_args()
    torch.manual_seed(42);np.random.seed(42)
    manifest=Path(args.manifest).resolve()
    df=read_manifest(manifest)
    cached=extract(df,manifest.parent,Path(args.cache))
    labels=cached['labels'];data=cached['rows'];classes=len(labels)
    splits={key:[row for row in data if row[3]==key] for key in ('train','val','test')}
    device='cuda' if torch.cuda.is_available() else 'cpu'
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    results=[]
    for typ,drop in [('concat',False),('cmaf',False),('cmaf',True)]:
        name='concat' if typ=='concat' else ('cmaf_drop' if drop else 'cmaf_no_drop')
        model=(CMAFNet(input_dims=FEATURE_DIMS,classes=classes) if typ=='cmaf' else ConcatBaseline(input_dims=FEATURE_DIMS,classes=classes))
        model=fit(model,splits['train'],splits['val'],typ,device,args.epochs,args.batch_size,args.lr,drop)
        torch.save({'model':model.cpu().state_dict(),'type':typ,'labels':labels,'input_dims':FEATURE_DIMS,'model_name':name},out/f'{name}.pt')
        model.to(device)
        for setting,index in [('all',None),('no_image',0),('no_text',1),('no_audio',2)]:
            metric=evaluate(model,splits['test'],typ,device,args.batch_size,index)
            results.append({'model':name,'setting':setting,**metric})
    # Unimodal probes trained using the same concatenation model architecture.
    for kept,label in [(0,'image'),(1,'text'),(2,'audio')]:
        masked_splits={k:[(f,[m[i] and i==kept for i in range(3)],y,s) for f,m,y,s in rows if m[kept]] for k,rows in splits.items()}
        if not all(masked_splits.values()): continue
        model=ConcatBaseline(input_dims=FEATURE_DIMS,classes=classes)
        model=fit(model,masked_splits['train'],masked_splits['val'],'concat',device,args.epochs,args.batch_size,args.lr,False)
        metric=evaluate(model,masked_splits['test'],'concat',device,args.batch_size)
        results.append({'model':f'{label}_only','setting':'unimodal',**metric})
    pd.DataFrame(results).to_csv(out/'ablation.csv',index=False)
    print(f'Wrote {out}/ablation.csv and checkpoints')

if __name__=='__main__': main()
