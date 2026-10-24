import io
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import torch
from PIL import Image
from cmaf.model import CMAFNet
from cmaf.features import image_features, text_features, audio_features, missing_feature, audio_from_bytes, FEATURE_DIMS

st.set_page_config(page_title='CMAF-Net | Multimodal Resilience Lab',page_icon='🧠',layout='wide')
st.title('CMAF-Net')
st.caption('Gated cross-modal attention • image, text and audio • modality-drop robustness')

with st.sidebar:
    st.header('Model configuration')
    checkpoint=st.text_input('Trained checkpoint file',value='artifacts/cmaf_drop.pt')
    st.info('No checkpoint = architecture explorer only. Predictions are deliberately disabled until a trained model is loaded.')
    st.markdown('**Sarvam AI (optional)**')
    st.caption('Set SARVAM_API_KEY as an environment variable or Streamlit secret. Never paste keys into source code.')
    try:
        sarvam_key=os.getenv('SARVAM_API_KEY') or st.secrets.get('SARVAM_API_KEY','')
    except Exception:
        sarvam_key=os.getenv('SARVAM_API_KEY','')
    st.write('Sarvam configured:', 'Yes' if sarvam_key else 'No')

@st.cache_resource(show_spinner='Loading pretrained image and language encoders...')
def get_model(path,mtime):
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if payload.get('type')!='cmaf':raise ValueError('App requires a CMAF checkpoint, not a concatenation checkpoint.')
    model=CMAFNet(input_dims=tuple(payload['input_dims']), classes=len(payload['labels']))
    model.load_state_dict(payload['model']);model.eval()
    return model,payload['labels']

ready=False
if Path(checkpoint).is_file():
    try:
        model, labels=get_model(checkpoint,Path(checkpoint).stat().st_mtime)
        ready=True;st.success('Trained checkpoint loaded. Model inference enabled.')
    except Exception as exc: st.error(f'Checkpoint error: {exc}')
else:
    st.warning('No trained checkpoint found. You can explore the workflow and Sarvam transcription, but scores/attention will not be invented.')

tab1,tab2,tab3=st.tabs(['🔬 Live inference','📊 Ablation results','🧩 Architecture & methodology'])
with tab1:
    st.subheader('Supply one or more modalities')
    c1,c2,c3=st.columns(3)
    with c1:
        photo=st.file_uploader('Image (JPG/PNG)',type=['jpg','jpeg','png'])
        if photo: st.image(photo,width=280)
        include_image=st.checkbox('Use image',value=True)
        image_noise=st.slider('Image noise',0.,0.7,0.,.05)
    with c2:
        typed=st.text_area('Text or transcript',height=130,placeholder='Enter speech transcript or review...')
        include_text=st.checkbox('Use text',value=True)
        text_drop=st.slider('Token corruption (approx.)',0.,.8,0.,.1)
    with c3:
        audio=st.file_uploader('Audio (WAV)',type=['wav'])
        if audio:st.audio(audio)
        include_audio=st.checkbox('Use audio',value=True)
        audio_noise=st.slider('Audio waveform noise',0.,0.2,0.,.01)
        if audio and st.button('Transcribe with Sarvam Saaras v4',disabled=not bool(sarvam_key)):
            try:
                from cmaf.sarvam import transcribe
                st.session_state['transcript']=transcribe(audio.getvalue(),audio.name,sarvam_key)
            except Exception as exc:st.error(f'Transcription failed: {exc}')
        if 'transcript' in st.session_state:st.write('Transcript:',st.session_state['transcript'])
    use_transcript=st.checkbox('Use Sarvam transcript instead of typed text',value=False)
    effective_text=st.session_state.get('transcript','') if use_transcript else typed
    if st.button('Run multimodal prediction',type='primary',disabled=not ready):
        avail=[bool(photo and include_image),bool(effective_text.strip() and include_text),bool(audio and include_audio)]
        if not any(avail):st.error('Add at least one enabled modality.')
        else:
            try:
                with st.spinner('Extracting features and running CMAF-Net...'):
                    if avail[0]:
                        img=Image.open(io.BytesIO(photo.getvalue())).convert('RGB')
                        if image_noise:
                            arr=np.asarray(img,dtype=np.float32)/255
                            arr=np.clip(arr+np.random.default_rng(42).normal(0,image_noise,arr.shape),0,1)
                            img=Image.fromarray((arr*255).astype('uint8'))
                        im=image_features(img)
                    else:im=missing_feature(512)
                    if avail[1]:
                        words=effective_text.split()
                        if text_drop:
                            rng=np.random.default_rng(42)
                            words=[w for w in words if rng.random()>text_drop]
                        tx=text_features(' '.join(words) or '[MASK]')
                    else:tx=missing_feature(768)
                    if avail[2]:
                        wave,sr=audio_from_bytes(audio.getvalue())
                        if audio_noise:wave=wave+np.random.default_rng(42).normal(0,audio_noise,wave.shape)
                        au=audio_features(wave,sr)
                    else:au=missing_feature(128)
                    feats=[z.unsqueeze(0) for z in (im,tx,au)]
                    mask=torch.tensor([[avail[0],avail[1],avail[2]]])
                    with torch.no_grad():logits,gates,attn=model(feats,mask,return_attention=True)
                    prob=torch.softmax(logits,dim=-1)[0].numpy()
                st.subheader('Prediction')
                st.metric('Predicted label',labels[int(prob.argmax())]);st.caption('Confidence is a model score, not a calibrated probability of correctness.')
                st.bar_chart(pd.DataFrame({'score':prob},index=labels))
                st.subheader('Gating weights')
                st.bar_chart(pd.DataFrame({'gate':gates[0].numpy()},index=['Image','Text','Audio']))
                st.caption('Absent modalities are masked to zero. Gate weights are not causal explanations.')
                active={k:v for k,v in attn.items() if avail[['image','text','audio'].index(k.split('<-')[0])] and avail[['image','text','audio'].index(k.split('<-')[1])]}
                if active:
                    target=st.selectbox('Attention pair',list(active))
                    fig,ax=plt.subplots(figsize=(7,3))
                    ax.imshow(active[target][0].numpy(),aspect='auto',interpolation='nearest')
                    ax.set(title=f'Cross attention: {target}',xlabel='Key positions',ylabel='Query positions')
                    st.pyplot(fig);plt.close(fig)
            except Exception as exc:st.error(f'Inference failed: {exc}')
with tab2:
    path=Path('artifacts/ablation.csv')
    if path.exists():
        result=pd.read_csv(path)
        st.dataframe(result,use_container_width=True,hide_index=True)
        usable=result.dropna(subset=['macro_f1'])
        if not usable.empty:
            st.bar_chart(usable.pivot(index='model',columns='setting',values='macro_f1'))
        st.download_button('Download ablation CSV',result.to_csv(index=False),file_name='cmaf_ablation.csv')
    else:
        st.info('Train your model with train.py to generate actual experimental results. No benchmark figures are fabricated.')
with tab3:
    st.markdown('''### Processing pipeline
1. **Visual encoder:** pretrained ResNet-18, 7×7 patch grid, 512 dimensions.
2. **Text encoder:** pretrained DistilBERT, up to 64 token states of 768 dimensions.
3. **Audio:** 128-bin log-Mel frames, subsampled to at most 80 frames. This branch uses spectral features rather than a separately pretrained audio encoder.
4. **Attention:** all six directed pairs of cross-modal attention, with residual fusion into each source sequence.
5. **Gates:** masked softmax over three learned scores. Missing modalities get exactly zero weight.
6. **Modality dropout:** during training, 25% chance per sample to remove one available modality.
7. **Class head:** pooled and gated representation → sentiment prediction.

**Academic disclaimer:** This is an implementable original combination, not a claim that cross-attention or modality dropout were invented here. Always use real holdout metrics and disclose dataset/feature assumptions.
''')
    st.markdown('Use `python train.py --manifest data/manifest.csv` from the project root. See README for the CSV schema and evaluation protocol.')
