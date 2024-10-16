#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   mfcc.py
@Time    :   2024/10/10 11:33:07
@Author  :   zhanglingling 
@Version :   1.0
@Contact :   None
@License :   None
@Desc    :   None
'''

# here put the import lib
import numpy as np 
import scipy

nSamplesPerSec = 16000 # 采样率
length_DFT = 1024 #傅里叶点数
hop_length = 160  #步长 下一帧取数据相对于这一帧的右偏移量
win_length = 800  #帧长 假设采样率16k, 则取0.1时间长的数据
number_filterbanks = 80 #过滤器个数
preemphasis = 0.97 #预加重系数 
max_db = 100
ref_db = 20
r = 1
pi = 3.14159265358979323846
mel_basis = None
hannWindow = None

def hz_to_mel(frequencies, htk=False):
    if htk:
        return 2595.0 * np.log10(1.0 + frequencies / 700.0)
    f_min = 0.0
    f_sp = 200.0 / 3
    mels = (frequencies - f_min) / f_sp
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    if frequencies >= min_log_hz:
        mels = min_log_mel + np.log(frequencies / min_log_hz) / logstep
    return mels

def mel_to_hz(mels, htk=False):
    # if htk:
    #     return 700.0 * (10.0**(mels / 2595.0) - 1.0)
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = mels * f_sp + f_min
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    log_t = mels >= min_log_mel
    freqs[log_t] = min_log_hz * np.exp((mels[log_t] - min_log_mel) * logstep)
    return freqs

def linspace(min_, max_, length):
    return np.linspace(min_, max_, length)

def mel_spectrogram_create(nps, n_fft, n_mels):
    f_max = nps / 2.0
    f_min = 0.0
    n_fft_2 = 1 + n_fft // 2
    weights = np.zeros((n_mels, n_fft_2), dtype=np.float64)
    fftfreqs = linspace(f_min, f_max, n_fft_2)
    min_mel = hz_to_mel(f_min)
    max_mel = hz_to_mel(f_max)
    mels = linspace(min_mel, max_mel, n_mels + 2)
    mel_f = mel_to_hz(mels)
    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)
    lower = -ramps[:n_mels] / fdiff[:n_mels, np.newaxis] 
    upper = ramps[2:] / fdiff[1:, np.newaxis]
    weights = np.maximum(0, np.minimum(lower, upper))
    enorm = 2.0 / (mel_f[2:n_mels + 2] - mel_f[:n_mels])
    weights *= enorm[:, np.newaxis]
    return weights




def create_hann_window(n_fft, win_length):
    hann_window = np.zeros((1, n_fft), dtype=np.float32)
    if n_fft > win_length:
        insert_cnt = (n_fft - win_length) // 2
    else:
        return np.array([])

    k = np.arange(1, win_length + 1)
    window_values = 0.5 * (1 - np.cos(2 * pi * k / (win_length + 1)))
    hann_window[0, insert_cnt:insert_cnt+win_length] = window_values

    return hann_window



def magnitude_spectrogram(emphasis_data, n_fft=2048, hop_length=0, win_length=0):
    if win_length == 0:
        win_length = n_fft
    if hop_length == 0:
        hop_length = win_length // 4
    pad_length = n_fft // 2
    cv_padbuffer = np.pad(emphasis_data[0, :], (pad_length, pad_length), mode='reflect')[None, :]
    #windowing加窗：将每一帧乘以汉宁窗，以增加帧左端和右端的连续性。
    #生成一个1600长度的hannWindow，并居中到2048长度的
    global hannWindow
    if hannWindow is None:
        hannWindow = create_hann_window(n_fft, win_length)
    
   
    number_feature_vectors = (cv_padbuffer.size - n_fft) // hop_length + 1
    number_coefficients = n_fft // 2 + 1
    feature_vector = np.zeros((number_feature_vectors, number_coefficients), dtype=np.float32)
    for i in range(0, cv_padbuffer.size - n_fft + 1, hop_length):
        framef = cv_padbuffer[0, i:i + n_fft][None, :] * hannWindow
        spectrum = np.fft.rfft(framef, n=n_fft)
        feature_vector[i // hop_length] = np.abs(spectrum)
    return feature_vector.T

def log_mel(ifile_data, nSamples_per_sec):
    if nSamples_per_sec != nSamplesPerSec:
        return -1
    emphasis_data = np.concatenate((np.zeros((1, 1), dtype=np.float32), ifile_data[:, 1:] - preemphasis * ifile_data[:, :-1]), axis=-1)
    mag = magnitude_spectrogram(emphasis_data, length_DFT, hop_length, win_length).astype(np.float64)
  
    mag_power = np.abs(mag) ** 2 
    mag = np.abs(mag) ** 2
    global mel_basis
    if mel_basis is None:
        mel_basis = mel_spectrogram_create(nSamplesPerSec, length_DFT, number_filterbanks)
    mel = np.dot(mel_basis, mag)
    mel_copy = mel.copy() 
    mel = np.log(mel + 1e-5) / 2.3025850929940459 * 10#20 * np.log10(np.maximum(1e-5, mel))
    mel = mel - ref_db #np.clip((mel - ref_db + max_db) / max_db, 1e-8, 1)
    mel = mel.T.astype(np.float32)
    return mel