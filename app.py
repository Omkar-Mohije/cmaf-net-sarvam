import json
import os
import streamlit as st
import pandas as pd
from sarvam_client import transcribe, classify_text, explain_analysis, SarvamError
from signal_metrics import image_metrics, wav_metrics

st.set_page_config(page_title="CMAF-Net | Sarvam Workshop", page_icon="🧠", layout="wide")
st.markdown("""<style>
.block-container{max-width:1200px;padding-top:2rem}
.hero{border:1px solid #d8e0ec;border-radius:18px;padding:24px;background:linear-gradient(120deg,#0b2544,#164c65);color:white}
.hero h1{color:white;margin:0;font-size:2.2rem}.hero p{color:#d4e9ee;margin:5px 0 0}
.smallnote{font-size:.85rem;color:#667085}
</style>""",unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>🧠 CMAF-Net Lab</h1><p>Multilingual sentiment intelligence · Sarvam AI · Image + text + audio diagnostics</p></div>',unsafe_allow_html=True)
st.caption("Live Sarvam-powered workshop demonstrator. Trained tri-modal CMAF predictions require a separate trained checkpoint; this deployment does not invent fusion outputs.")

def api_key():
    try: return os.getenv("SARVAM_API_KEY","") or st.secrets.get("SARVAM_API_KEY","")
    except Exception: return os.getenv("SARVAM_API_KEY","")

with st.sidebar:
    st.header("Workshop settings")
    key=api_key()
    if key:
        st.success("Sarvam API configured")
    else:
        st.warning("SARVAM_API_KEY not configured")
    st.caption("Configure a newly rotated key in Streamlit Cloud → App settings → Secrets.")
    st.divider()
    st.write("**Active Sarvam models**")
    st.code("saaras:v4\nsarvam-105b", language="text")
    st.write("**Privacy**")
    st.caption("Uploaded audio and submitted text go to Sarvam only when you click the API buttons. Image stays in this application runtime. Avoid sensitive data.")
    st.caption("Never place a secret in your public repository.")

tab_live,tab_robust,tab_about=st.tabs(["🎙️ Live workshop", "🧪 Modality-drop simulator", "📐 Research architecture"])
with tab_live:
    st.subheader("Analyze a multimodal example")
    a,b,c=st.columns(3)
    with a:
        st.markdown("**1 · Image**")
        img=st.file_uploader("Upload JPG/PNG", type=["png","jpg","jpeg"])
        if img: st.image(img,caption="Visual context",use_container_width=True)
    with b:
        st.markdown("**2 · Text**")
        typed=st.text_area("Enter a review, opinion or transcript",height=150,
           placeholder="I really enjoyed this experience and learned a lot!")
    with c:
        st.markdown("**3 · Audio**")
        audio=st.file_uploader("Upload WAV/MP3/M4A ≤30 seconds",type=["wav","mp3","m4a"])
        if audio:st.audio(audio)
        mode=st.selectbox("Saaras mode",["transcribe","translate"],help="Translate mode produces English text")
        if st.button("Transcribe with Sarvam",disabled=not bool(audio and key),use_container_width=True):
            try:
                with st.spinner("Calling Saaras v4..."):
                    result=transcribe(audio.getvalue(),audio.name,key,mode)
                st.session_state["sarvam_transcript"]=result["text"]
                st.session_state["speech_language"]=result["language"]
                st.success("Transcribed successfully")
            except Exception as e: st.error(str(e))
    if "sarvam_transcript" in st.session_state:
        st.text_area("Sarvam transcript",st.session_state["sarvam_transcript"],height=100,disabled=True)
        st.caption("Detected language: " + str(st.session_state.get("speech_language", "N/A")))
    prefer_transcript=st.checkbox("Use Sarvam transcript for text sentiment",value=True)
    effective_text=(st.session_state.get("sarvam_transcript","") if prefer_transcript and st.session_state.get("sarvam_transcript") else typed).strip()
    run=st.button("✨ Analyze sentiment with Sarvam-105B",type="primary",disabled=not bool(effective_text and key))
    if run:
        try:
            with st.spinner("Calling Sarvam-105B..."):
                result=classify_text(effective_text,key)
            st.session_state["sentiment_result"]=result
            st.session_state["text_analyzed"]=effective_text
        except Exception as e: st.error(str(e))
    if "sentiment_result" in st.session_state:
        r=st.session_state["sentiment_result"]
        st.markdown("### Sarvam text sentiment result")
        x,y=st.columns(2)
        x.metric("Sentiment",r["label"])
        y.metric("LLM-estimated confidence (uncalibrated)",f'{r["confidence"]:.0%}')
        st.write(r["explanation"])
        if r["cues"]: st.caption("Text cues: " + ", ".join(r["cues"]))
        st.info("This sentiment label comes from Sarvam-105B analyzing text. It is **not** a trained CMAF-Net tri-modal prediction.")
        summary={"source":"Sarvam-105B text-only sentiment","result":r,"transcription":st.session_state.get("sarvam_transcript","")}
        if img:
            try: summary["image_diagnostics"]=image_metrics(img.getvalue())
            except Exception as e: st.warning(f"Image metrics unavailable: {e}")
        if audio and audio.name.lower().endswith(".wav"):
            try: summary["audio_diagnostics"]=wav_metrics(audio.getvalue())
            except Exception as e: st.warning(f"Audio metrics unavailable: {e}")
        st.markdown("#### Modality diagnostics (descriptive, not sentiment weights)")
        m1,m2,m3=st.columns(3)
        m1.metric("Image", "Present" if img else "Missing")
        m2.metric("Text", "Present" if effective_text else "Missing")
        m3.metric("Audio", "Present" if audio else "Missing")
        with st.expander("View measured image/audio statistics"):
            st.json({k:v for k,v in summary.items() if k.endswith("diagnostics")})
        language=st.selectbox("Sarvam explanation language",["English","Hindi","Marathi","Tamil","Telugu","Bengali"])
        if st.button("Generate multilingual explanation",disabled=not bool(key)):
            try:
                with st.spinner("Calling Sarvam-105B..."):
                    st.session_state["explanation"]=explain_analysis(summary,key,language)
            except Exception as e: st.error(str(e))
        if "explanation" in st.session_state:st.write(st.session_state["explanation"])
        st.download_button("Export result JSON",json.dumps(summary,ensure_ascii=False,indent=2),"cmaf_workshop_result.json","application/json")
with tab_robust:
    st.subheader("Missing modality simulator")
    st.write("This is an **availability visualization**, not a trained-model robustness benchmark. A real CMAF-Net gate would zero out absent modalities and renormalize learned weights.")
    i,t,a=st.columns(3)
    with i: keep_image=st.checkbox("Image available",True,key="keep_img")
    with t: keep_text=st.checkbox("Text available",True,key="keep_txt")
    with a: keep_audio=st.checkbox("Audio available",True,key="keep_aud")
    mask={"Image":int(keep_image),"Text":int(keep_text),"Audio":int(keep_audio)}
    st.bar_chart(pd.DataFrame({"Available (1/0)":mask}))
    st.code(f'Modality mask = [{int(keep_image)}, {int(keep_text)}, {int(keep_audio)}]')
    if not keep_text:st.warning("Text sentiment is unavailable without text/transcription. No fake fallback prediction will be made.")
    st.markdown("**Experimental evaluation after training**: clean test macro-F1, each of three missing-modality macro-F1 scores, performance change, and additive-noise sweeps.")
with tab_about:
    st.subheader("CMAF-Net research model")
    st.markdown("""
    **Target model (implemented separately in `research/`):**
    1. **Image:** pretrained ResNet-18, patch-level 512-d features.
    2. **Text:** DistilBERT token embeddings (768 dimensions).
    3. **Audio:** log-Mel spectrogram frame features (128 bins).
    4. **Six cross-modal attention directions:** each available modality queries the other two.
    5. **Gated pooling:** masked softmax gates; absent inputs receive zero contribution.
    6. **Modality-drop training:** 25% probability of dropping one available modality per sample.
    7. **Evaluation:** independent test set and leave-one-modality-out ablation.

    **Current workshop deployment**: lightweight Sarvam transcription and text sentiment, image/audio descriptive diagnostics. It does not claim trained CMAF inference. Launch `research/app.py` after creating a validated checkpoint to show true gated predictions and attention maps.

    **Novelty caution:** Gated fusion, cross-attention and modality dropout are known methods. The contribution is their comparative implementation and evaluation, not their invention.
    """)
    st.markdown("[Sarvam API documentation](https://docs.sarvam.ai/) · [Streamlit deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)")
