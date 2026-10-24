"""Descriptive diagnostics only: not trained image/audio sentiment classifiers."""
import io
import numpy as np
from PIL import Image, ImageFilter
import wave

def image_metrics(content):
    if len(content)>8_000_000: raise ValueError("Image exceeds 8 MB limit")
    with Image.open(io.BytesIO(content)) as original:
        image=original.convert("RGB")
        image.thumbnail((512,512))
    a=np.asarray(image,dtype=np.float32)/255
    luminance=.2126*a[:,:,0]+.7152*a[:,:,1]+.0722*a[:,:,2]
    return {"width":image.width,"height":image.height,"mean_brightness":round(float(luminance.mean()),3),
            "color_variation":round(float(a.std()),3)}

def wav_metrics(content):
    with wave.open(io.BytesIO(content),'rb') as f:
        channels=f.getnchannels(); rate=f.getframerate(); frames=f.getnframes(); width=f.getsampwidth()
        if width not in (1,2,4): raise ValueError("Only PCM WAV audio supported for signal diagnostics")
        if frames / rate > 30: raise ValueError("Use an audio clip up to 30 seconds")
        raw=f.readframes(frames)
    dtype={1:np.uint8,2:np.int16,4:np.int32}[width]
    audio=np.frombuffer(raw,dtype=dtype).astype(np.float64)
    if width==1: audio=(audio-128)/128
    else: audio=audio/float(2**(8*width-1))
    if channels>1: audio=audio.reshape(-1,channels).mean(axis=1)
    rms=float(np.sqrt(np.mean(audio*audio))) if len(audio) else 0.
    return {"duration_seconds":round(frames/rate,2),"sample_rate":rate,"rms":round(rms,4),
            "peak":round(float(np.max(np.abs(audio))),4) if len(audio) else 0.}
