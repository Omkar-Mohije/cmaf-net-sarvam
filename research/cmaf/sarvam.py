import io
import os

def transcribe(audio_bytes, filename='audio.wav', key=None):
    from sarvamai import SarvamAI
    secret = key or os.environ.get('SARVAM_API_KEY')
    if not secret:
        raise ValueError('Set SARVAM_API_KEY in environment or Streamlit secrets')
    api = SarvamAI(api_subscription_key=secret)
    stream = io.BytesIO(audio_bytes)
    stream.name = filename
    result = api.speech_to_text.transcribe(file=stream, model='saaras:v4', mode='transcribe')
    return result.transcript

def translate(text, source='hi-IN', key=None):
    from sarvamai import SarvamAI
    secret = key or os.environ.get('SARVAM_API_KEY')
    if not secret:
        raise ValueError('Set SARVAM_API_KEY in environment or Streamlit secrets')
    api = SarvamAI(api_subscription_key=secret)
    result = api.text.translate(input=text[:2000], source_language_code=source,
                                target_language_code='en-IN', model='sarvam-translate:v1')
    return result.translated_text
