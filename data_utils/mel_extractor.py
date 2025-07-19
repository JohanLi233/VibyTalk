import torch
import torch.nn.functional as F
import soundfile as sf
import librosa
import numpy as np
import kaldi_native_fbank as knf
from argparse import ArgumentParser


def get_mel_from_16k_wav(wav_16k_name, num_mel_bins=80):
    """
    Extract mel features from 16kHz wav file

    Args:
        wav_16k_name: 16kHz wav file path
        num_mel_bins: Number of mel filter banks
        normalize: Whether to perform per-instance normalization

    Returns:
        Mel feature tensor with shape (T//4, 4, num_mel_bins)
    """
    speech_16k, sr = sf.read(wav_16k_name)
    if sr != 16000:
        raise ValueError(f"Expected 16kHz audio, got {sr}Hz")
    features = get_mel_from_16k_speech(speech_16k, num_mel_bins)
    return features


def _per_instance_normalize(features):
    """
    Perform per-instance normalization on audio features.

    Args:
        features: Audio feature tensor of arbitrary shape

    Returns:
        Normalized feature tensor with unchanged shape
    """
    # Flatten all dimensions, compute global mean and std
    flattened = features.flatten()
    mean = flattened.mean()
    std = flattened.std()
    # Prevent division by zero
    if std > 1e-8:
        normalized = (features - mean) / std
    else:
        normalized = features - mean
    return normalized


def get_mel_from_16k_speech(
    speech, num_mel_bins=80, frame_length_ms=25, frame_shift_ms=10
):
    """
    Extract mel features from 16kHz audio using kaldi_native_fbank.
    Output feature frame rate is 100Hz (10ms frame shift).

    Args:
        speech: 16kHz audio array
        num_mel_bins: Number of mel filter banks
        frame_length_ms: Frame length (milliseconds)
        frame_shift_ms: Frame shift (milliseconds)
        normalize: Whether to perform per-instance normalization (recommended for MAE tasks)

    Returns:
        Mel feature tensor with shape (T//4, 4, num_mel_bins)
        If normalize=True, features will be normalized to zero mean and unit variance
    """
    if speech.ndim == 2:
        # Convert stereo to mono
        speech = speech.mean(axis=1)

    # kaldi_native_fbank parameter setup
    opts = knf.FbankOptions()
    opts.frame_opts.dither = 0.0
    opts.frame_opts.window_type = "hanning"
    opts.frame_opts.frame_length_ms = frame_length_ms
    opts.frame_opts.frame_shift_ms = frame_shift_ms
    opts.mel_opts.num_bins = num_mel_bins
    opts.energy_floor = 1e-10
    opts.use_energy = False

    # Extract fbank features
    fbank_extractor = knf.OnlineFbank(opts)
    fbank_extractor.accept_waveform(16000, speech)
    fbank_extractor.input_finished()

    # Get all frames
    fbank = np.stack(
        [fbank_extractor.get_frame(i) for i in range(fbank_extractor.num_frames_ready)]
    )

    # Convert to torch tensor
    mel_features = torch.from_numpy(fbank).float()

    # --- Feature processing logic ---
    # Goal: Process 100Hz Mel features to 25Hz features (group every 4 frames)
    mel_features = make_first_dim_divisible(mel_features, 4)
    T, D = mel_features.shape
    mel_features = mel_features.reshape(T // 4, 4, D)

    mel_features = _per_instance_normalize(mel_features)

    return mel_features


def make_first_dim_divisible(tensor, n):
    """
    Make the first dimension of tensor divisible by n through padding.
    """
    if tensor.size(0) % n != 0:
        pad_len = n - (tensor.size(0) % n)
        # F.pad's 'replicate' mode requires 3D or higher, temporarily expand then reduce dimensions
        return F.pad(tensor.unsqueeze(0), (0, 0, 0, pad_len), mode="replicate").squeeze(
            0
        )
    return tensor


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--wav", type=str, required=True, help="wav文件路径")
    parser.add_argument("--num_mel_bins", type=int, default=80, help="mel滤波器组数量")
    args = parser.parse_args()

    wav_name = args.wav
    num_mel_bins = args.num_mel_bins

    print(f"正在读取音频文件: {wav_name} ...")
    speech, sr = sf.read(wav_name)

    # 若采样率不是16kHz，则重采样
    if sr != 16000:
        print(f"重采样: {sr}Hz -> 16000Hz ...")
        speech_16k = librosa.resample(
            speech.astype(np.float32),
            orig_sr=sr,
            target_sr=16000,
            res_type="kaiser_best",
        )
    else:
        speech_16k = speech

    print(f"提取mel特征: {num_mel_bins}个mel bins, 帧率100Hz ...")

    # 得到mel_features，帧率为25Hz（每4个10ms帧分组，40ms）
    mel_features = get_mel_from_16k_speech(speech_16k, num_mel_bins)

    # 现在mel_features为(T//4, 4, D)格式，与音频特征处理一致

    output_path = wav_name.replace(".wav", f"_mel_{num_mel_bins}.npy")
    np.save(output_path, mel_features.numpy())
    print(f"\n已保存mel特征，形状为{mel_features.numpy().shape}，路径: {output_path}")
    print(
        f"特征格式: (T={mel_features.shape[0]}, temporal_group=4, feature_dim={mel_features.shape[2]})"
    )
    print("该格式的有效帧率为25Hz（每个时间步覆盖40ms）")
    print(
        f"标准化特征 - 均值: {mel_features.mean():.6f}, 标准差: {mel_features.std():.6f}"
    )
