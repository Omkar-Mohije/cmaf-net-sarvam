from unittest.mock import patch, Mock
import json
from sarvam_client import classify_text, transcribe, SarvamError
from signal_metrics import image_metrics, wav_metrics
from PIL import Image
import io
import wave

def fake(payload, status=200):
    r=Mock();r.ok=status==200;r.status_code=status;r.json.return_value=payload;r.text="failure";return r

def test_chat_parsing():
    payload={"choices":[{"message":{"content":json.dumps({"label":"Positive","confidence":.8,"explanation":"Pleasant","cues":["great"]})}}]}
    with patch("requests.post",return_value=fake(payload)) as send:
        res=classify_text("Great workshop", "fake-test-key")
    assert res["label"]=="Positive"
    assert send.call_args.kwargs["headers"]["api-subscription-key"]=="fake-test-key"
    assert send.call_args.kwargs["json"]["model"]=="sarvam-105b"
    assert "reasoning_effort" in send.call_args.kwargs["json"]
    assert send.call_args.kwargs["json"]["reasoning_effort"] is None

def test_transcribe():
    with patch("requests.post",return_value=fake({"transcript":"namaste","language_code":"hi-IN"})) as send:
        result=transcribe(b"audio", "record.wav", "fake-test-key")
    assert result["text"]=="namaste"
    assert send.call_args.kwargs["data"]["model"]=="saaras:v4"

def test_auth_failure():
    with patch("requests.post",return_value=fake({},401)):
        try: classify_text("Hello","bad")
        except SarvamError as e: assert "Authentication" in str(e)
        else: assert False

def test_image_diagnostics():
    buffer=io.BytesIO();Image.new("RGB",(20,10),"red").save(buffer,format="PNG")
    assert image_metrics(buffer.getvalue())["width"]==20

def test_wav_diagnostics():
    buffer=io.BytesIO()
    with wave.open(buffer,'wb') as f:
        f.setnchannels(1);f.setsampwidth(2);f.setframerate(16000);f.writeframes(b"\0\0"*1600)
    assert abs(wav_metrics(buffer.getvalue())["duration_seconds"]-.1)<.001


def test_empty_model_content_is_readable_error():
    payload={"choices":[{"message":{"content":None,"reasoning_content":"reasoning"},"finish_reason":"length"}]}
    with patch("requests.post",return_value=fake(payload)):
        try:
            classify_text("A pleasant evening", "fake-test-key")
        except SarvamError as e:
            assert "empty answer" in str(e)
            assert "finish_reason=length" in str(e)
        else:
            assert False
