import numpy as np
from scipy.io.wavfile import read


def q1(audio_path) -> np.array:
    """
    :param audio_path: path to q1 audio file
    :return: return q1 denoised version
    """
    _, original = read(audio_path)
    fft = np.fft.fft(original)
    magnitude = np.abs(fft)
    clean_fft = magnitude
    clean_fft[2700] = (clean_fft[2699] + clean_fft[2701]) / 2
    clean_fft[6900] = (clean_fft[6899] + clean_fft[6901]) / 2
    clean = np.fft.ifft(clean_fft * np.exp(1j * np.angle(fft))).real
    return clean


def q2(audio_path) -> np. array:
    """
    :param audio_path: path to q2 audio file
    :return: return q2 denoised version
    """
    sr, original = read(audio_path)
    noisy_indexes = np.arange(int(1.5*sr), int(4*sr))
    noisy = original[noisy_indexes]

    fft = np.fft.fft(noisy)
    magnitude = np.abs(fft)
    freqs = np.fft.fftfreq(len(noisy), 1/sr)
    noise_freq = 600

    noise_freq_indexes = np.where(
        (abs(freqs) > noise_freq - 20)
        & (abs(freqs) < noise_freq + 20))
    clean_fft = magnitude.copy()
    clean_fft[noise_freq_indexes] = 0.05 * clean_fft[noise_freq_indexes]
    clean = np.fft.ifft(clean_fft * np.exp(1j * np.angle(fft))).real
    clean = np.concatenate(
        (original[:int(1.5 * sr)], clean, original[int(4*sr):]))
    return clean
