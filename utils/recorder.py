import numpy as np
import soundfile as sf

def _mel_filterbank(n_fft, sr, n_mels=40, fmin=300.0, fmax=None):
    if fmax is None:
        fmax = sr / 2.0
    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)
    def mel_to_hz(m):
        return 700.0 * (10.0**(m / 2595.0) - 1.0)
    mel_points = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    fbanks = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(1, n_mels + 1):
        left = bin_points[i - 1]
        center = bin_points[i]
        right = bin_points[i + 1]
        if center > left:
            fbanks[i - 1, left:center] = (np.arange(left, center) - left) / (center - left)
        if right > center:
            fbanks[i - 1, center:right] = (right - np.arange(center, right)) / (right - center)
    return fbanks

def compute_embedding(filepath):
    x, sr = sf.read(filepath)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x.astype(np.float32)
    if len(x) < sr // 2:
        pad = sr // 2 - len(x)
        x = np.pad(x, (0, pad))
    x[1:] = x[1:] - 0.97 * x[:-1]
    frame_len = int(0.025 * sr)
    hop = int(0.010 * sr)
    n_fft = 512 if sr <= 22050 else 1024
    fb = _mel_filterbank(n_fft, sr, n_mels=40)
    frames = []
    for start in range(0, len(x) - frame_len + 1, hop):
        frame = x[start:start + frame_len]
        frame = frame * np.hamming(frame_len)
        spec = np.fft.rfft(frame, n=n_fft)
        ps = (np.abs(spec) ** 2)
        mel = fb.dot(ps[:fb.shape[1]])
        mel = np.log(mel + 1e-10)
        frames.append(mel)
    if not frames:
        emb = np.zeros(fb.shape[0] * 2, dtype=np.float32)
    else:
        M = np.vstack(frames)
        mu = M.mean(axis=0)
        sigma = M.std(axis=0)
        emb = np.concatenate([mu, sigma]).astype(np.float32)
    np.save("owner_embed.npy", emb)


