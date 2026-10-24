"""Pretrained image and text encoders; log-Mel audio patch features (fixed projection)."""
from functools import lru_cache
import io
import numpy as np
import librosa
import torch
from PIL import Image

FEATURE_DIMS = (512, 768, 128)

@lru_cache(maxsize=1)
def load_encoders():
    from torchvision.models import resnet18, ResNet18_Weights
    from transformers import AutoTokenizer, AutoModel
    visual = resnet18(weights=ResNet18_Weights.DEFAULT)
    visual.fc = torch.nn.Identity()
    visual.eval()
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    language = AutoModel.from_pretrained('distilbert-base-uncased')
    language.eval()
    return visual, tokenizer, language

@torch.no_grad()
def image_features(img):
    from torchvision.models import ResNet18_Weights
    from torchvision.models.feature_extraction import create_feature_extractor
    visual, _, _ = load_encoders()
    # Last conv feature map: 7x7 image patches of dimension 512.
    extractor = create_feature_extractor(visual, return_nodes={'layer4': 'features'}).eval()
    from torchvision.models import ResNet18_Weights
    transform = ResNet18_Weights.DEFAULT.transforms()
    t = transform(img.convert('RGB')).unsqueeze(0)
    z = extractor(t)['features']
    return z.flatten(2).transpose(1,2).squeeze(0).float()

@torch.no_grad()
def text_features(text):
    _, tokenizer, model = load_encoders()
    encoded = tokenizer(text or ' ', return_tensors='pt', truncation=True, max_length=64)
    out = model(**encoded).last_hidden_state[0].float()
    return out

def audio_features(data, sr):
    # Log-mel patches; does not use an untrained audio CNN during extraction.
    wave = np.asarray(data, dtype=np.float32)
    if wave.ndim > 1:
        wave = wave.mean(axis=-1)
    if len(wave) == 0:
        raise ValueError('Empty audio input')
    if sr != 16000:
        wave = librosa.resample(wave, orig_sr=sr, target_sr=16000)
    wave = wave[:16000*30]
    mel = librosa.feature.melspectrogram(y=wave, sr=16000, n_fft=400, hop_length=160, n_mels=128)
    db = librosa.power_to_db(mel + 1e-9, ref=np.max)
    # Frame downsampling avoids quadratic attention over long recordings.
    index = np.linspace(0, db.shape[1]-1, min(80, db.shape[1])).astype(int)
    frames = db[:, index].T
    return torch.from_numpy(((frames + 80.) / 80.).astype(np.float32))

def image_from_bytes(content):
    return Image.open(io.BytesIO(content)).convert('RGB')

def audio_from_bytes(content):
    import soundfile as sf
    x, sr = sf.read(io.BytesIO(content),dtype='float32',always_2d=False)
    return x, sr

def collate_feature_rows(rows):
    """Rows are (list of three [length,dim] tensors, bool[3], integer label)."""
    from torch.nn.utils.rnn import pad_sequence
    streams=[]
    for m,dim in enumerate(FEATURE_DIMS):
        streams.append(pad_sequence([row[0][m] for row in rows],batch_first=True))
    masks = torch.tensor([row[1] for row in rows], dtype=torch.bool)
    labels = torch.tensor([row[2] for row in rows], dtype=torch.long)
    return streams,masks,labels

def missing_feature(dim):
    return torch.zeros(1,dim)
