"""Minimal documented Sarvam REST integration. No credentials persisted."""
import json
import re
import requests

BASE = "https://api.sarvam.ai"

class SarvamError(RuntimeError):
    pass

def _checked(resp):
    if not resp.ok:
        status = resp.status_code
        if status in (401, 403):
            raise SarvamError("Authentication failed. Check a newly rotated SARVAM_API_KEY and account permissions.")
        if status == 429:
            raise SarvamError("Sarvam rate limit reached. Try again later.")
        raise SarvamError(f"Sarvam returned HTTP {status}: {resp.text[:350]}")
    return resp.json()

def transcribe(audio: bytes, filename: str, key: str, mode="transcribe"):
    if not audio or len(audio) > 15_000_000:
        raise ValueError("Audio is empty or exceeds the 15 MB workshop limit.")
    r = requests.post(BASE + "/speech-to-text", headers={"api-subscription-key":key},
        files={"file": (filename, audio, "application/octet-stream")},
        data={"model":"saaras:v4", "mode":mode}, timeout=90)
    data = _checked(r)
    return {"text": data.get("transcript", ""), "language":data.get("language_code", "Not specified")}

def chat(messages, key, max_tokens=900):
    r = requests.post(BASE + "/v1/chat/completions",
        headers={"api-subscription-key": key, "Content-Type":"application/json"},
        json={"model":"sarvam-105b", "messages":messages, "temperature":0.1, "max_tokens":max_tokens, "reasoning_effort":None}, timeout=90)
    data=_checked(r)
    choices = data.get("choices") or []
    if not choices:
        raise SarvamError("Sarvam returned no completion choices. Please retry.")
    content = (choices[0].get("message") or {}).get("content")
    if not isinstance(content, str) or not content.strip():
        finish_reason = choices[0].get("finish_reason", "unknown")
        raise SarvamError(
            f"Sarvam returned an empty answer (finish_reason={finish_reason}). "
            "Please retry. If this persists, check model permissions and token limits."
        )
    return content

def classify_text(text, key):
    instructions = ("You are evaluating expressed SENTIMENT in a text transcript for a workshop. "
        "Respond with ONLY one JSON object with keys label, confidence, explanation, cues. "
        "label must be Positive, Neutral, or Negative. confidence must be a number between 0 and 1, "
        "and is a subjective model confidence estimate, NOT a calibrated probability. "
        "explanation must be a short sentence. cues must be a list of up to 4 short strings. "
        "Do not infer sentiment from the presence of disaster or protected demographic categories alone. "
        "No code fences. Be cautious with sarcasm and ambiguity.")
    raw = chat([{"role":"system","content":instructions},{"role":"user","content":"Classify this text:\n"+text[:6000]}],key)
    try:
        parsed=json.loads(raw.strip())
    except json.JSONDecodeError:
        match=re.search(r"\{.*\}", raw, re.S)
        if not match: raise SarvamError("The language model did not return parsable JSON. Please retry.")
        try: parsed=json.loads(match.group())
        except ValueError: raise SarvamError("The language model response was invalid JSON. Please retry.")
    label=str(parsed.get("label", "Neutral")).capitalize()
    if label not in {"Positive","Negative","Neutral"}: raise SarvamError("Unexpected sentiment label; please retry.")
    try: confidence=max(0., min(1., float(parsed.get("confidence", .5))))
    except (ValueError,TypeError): confidence=.5
    return {"label":label,"confidence":confidence,"explanation":str(parsed.get("explanation","")),
            "cues": [str(x) for x in parsed.get("cues", [])[:4]] if isinstance(parsed.get("cues"),list) else []}

def explain_analysis(summary, key, language="English"):
    instruction = ("Explain the provided computational results clearly in " + language + ". "
        "Explain that the LLM sentiment is based on text only, not a trained CMAF fusion prediction. "
        "Do not claim the image or original audio affected the sentiment label. Maximum 100 words.")
    return chat([{"role":"system","content":instruction},
                 {"role":"user","content":json.dumps(summary,ensure_ascii=False)}],key, max_tokens=300)
