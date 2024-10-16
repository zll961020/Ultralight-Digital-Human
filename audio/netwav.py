#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   netwav.py
@Time    :   2024/10/10 11:36:42
@Author  :   zhanglingling 
@Version :   1.0
@Contact :   None
@License :   None
@Desc    :   None
'''

# here put the import lib

import numpy as np
import threading
from scipy.io import wavfile 

MFCC_OFFSET=6400
##define MFCC_OFFSET  0
#define MFCC_DEFRMS  0.1f
MFCC_FPS = 25
MFCC_RATE  = 16000
# MFCC_WAVCHUNK  = 960000
MFCC_WAVCHUNK=560000
#define MFCC_WAVCHUNK  512

#define MFCC_MELBASE  6001
MFCC_MELBASE = 3501
MFCC_MELCHUNK = 80
#define MFCC_MELCHUNK  20

#define MFCC_BNFBASE  1499
MFCC_BNFBASE  = 874
MFCC_BNFCHUNK  = 256



class MBufCache:
    def __init__(self, initsec, secw, sech, blockh, in_bufchahe=None):
        self.m_secw = secw
        self.m_sech = sech
        self.m_blockh = blockh
        self.m_lineh = sech - blockh  # 874 - 20 
        self.m_lock = threading.Lock()
        self.vec_buf = [np.zeros((1, self.m_sech+1, self.m_secw), dtype=np.float32) for _ in range(initsec)] \
            if in_bufchahe is None else [in_bufchahe]
        self.m_tagarr = [0] * 512

    def secBuf(self, sec):
        with self.m_lock:
            if sec < len(self.vec_buf):
                return self.vec_buf[sec]
            else:
                mat = np.zeros((1, self.m_sech + 1, self.m_secw), dtype=np.float32)
                self.vec_buf.append(mat)
                return mat

    def inxBuf(self, inx):
        seca = inx // self.m_sech  # 行高 
        secb = inx % self.m_sech  # 列宽
        if secb >= self.m_lineh:
            mat = np.zeros((1, self.m_blockh, self.m_secw), dtype=np.float32)
            sa = self.secBuf(seca)
            la = self.m_sech - secb
            parta = la * self.m_secw 
            # 从sa的secb行拷贝到mat
            src_data = sa[:, secb:]
            assert parta <= len(src_data.flatten())
            src_data_flatted = src_data.flatten()[:parta] # 拷贝该行的parta个数据
            mat_flatten = mat.flatten()
            mat_flatten[:len(src_data_flatted)] = src_data_flatted 
           
            # 从sa的0行接着拷贝到mat中
            sb = self.secBuf(seca + 1)
            lb = self.m_blockh - la 
            partb = lb * self.m_secw 
            src_data_b = sa[:, 0:]
            src_data_b_flatten = src_data_b.flatten()[:partb] # 拷贝该行的partb个数据
            mat_flatten[len(src_data_flatted): len(src_data_flatted) + len(src_data_b_flatten)] = src_data_b_flatten
            # reshape回去
            mat = mat_flatten.reshape(mat.shape)
           
        else:
            src = self.secBuf(seca)
            # 将buf复制给mat 
            mat = np.zeros((1, self.m_blockh, self.m_secw), dtype=np.float32)
            mat = src[:, secb:secb+self.m_blockh, :]


        return mat

    def tagarr(self):
        return self.m_tagarr

class MBnfCache(MBufCache):
    def __init__(self, incache=None):
        # Example values for MFCC_BNFCHUNK and MFCC_BNFBASE
        super().__init__(3, MFCC_BNFCHUNK, MFCC_BNFBASE, 20, in_bufchahe=incache)



class KWav(object):
    """
    读取音频文件到缓冲区 
    """
    def __init__(self, audio_path):
        self.audio_path = audio_path
        fs, x = wavfile.read(audio_path)
        #print(f'fs {fs} len x: {len(x)}')
        self.m_waitcnt = 0 
        self.m_calcnt = 0 
        self.m_resultcnt = 0 
        self.m_leftsample = 0 
        self.m_waitsample = 0 
        #######计算缓冲区相关参数

        # 原始音频样本数
        self.m_pcmsmaple = len(x)
        # 用于计算的音频样本数
        self.m_wavsample = self.m_pcmsmaple + 2 * MFCC_OFFSET
        # 音频时长
        self.m_duration = len(x) * 1.0 / 16000.0
        # 完整的wav块数 块大小是MFCC_WAVCHUNK  
        self.m_seca = int(self.m_wavsample / MFCC_WAVCHUNK)
        # 剩余的样本数（不足一个块大小）
        self.m_secb = int(self.m_wavsample % MFCC_WAVCHUNK)
        # 如果有剩余样本，计算最后一块的mel特征数
        self.m_mellast = int(self.m_secb / 160) + 1 if self.m_secb else 0 
        # 如果有剩余样本，计算最后一块的BNF特征数
        self.m_bnflast = int(self.m_mellast * 0.25 - 0.75) if self.m_secb else 0
        # 计算总的wav大小，包括完整块和剩余样本
        self.m_wavsize = self.m_seca * MFCC_WAVCHUNK + self.m_secb
        # 计算总的mel特征大小，包括完整块和剩余样本
        self.m_melsize = self.m_seca * MFCC_MELBASE + self.m_mellast
        # 计算总的BNF特征大小，包括完整块和剩余样本
        self.m_bnfsize = self.m_seca * MFCC_BNFBASE + self.m_bnflast
        # 计算需要处理的块数 
        self.m_calsize = self.m_seca + 1 if self.m_secb else self.m_seca
        # 初始化 wav矩阵
        self.m_wavmat = np.zeros((1, self.m_calsize, MFCC_WAVCHUNK), dtype=np.float32) # （w, h, c）分别为宽、高和通道 对应numpy (通道, 高， 宽)
        # 初始化mel矩阵 
        self.m_melmat = np.zeros((1, self.m_calsize, MFCC_MELCHUNK * MFCC_MELBASE), dtype=np.float32)
        self.m_bnfblock = int(self.m_duration * MFCC_FPS)
        if self.m_bnfblock > self.m_bnfsize-10:
            self.m_bnfblock = self.m_bnfsize - 10 

        # 初始化 存储wenet输出特征矩阵
        self.m_bnfcache = MBnfCache()
        ##### 
        self.initinx()
        # 将数据读到wavmat中 
        flatted_wavmat = self.m_wavmat.flatten()
        flatted_wavmat[self.m_curwav:self.m_curwav+self.m_pcmsmaple] = x / 32767.0
        self.m_wavmat = flatted_wavmat.reshape(self.m_wavmat.shape)
        # incsample  
        self.incsample(self.m_pcmsmaple)
        self.readyall()
    
    def bnfblock(self):
        return self.m_bnfblock
    def inxBuf(self, index):
        return self.m_bnfcache.inxBuf(index)

    def initinx(self):
        self.m_curwav = 0 + MFCC_OFFSET # 指向m_wavmat 
        self.m_leftsample = self.m_pcmsmaple
        self.m_waitsample = MFCC_OFFSET
        secs = self.m_pcmsmaple * 1.0 / MFCC_RATE  # 秒数
        bnf_block = secs * MFCC_FPS  # 帧数
        if bnf_block > self.m_bnfsize - 10:
            bnf_block = self.m_bnfsize - 10 
        return 0 
    def incsample(self, sample):
        self.m_leftsample -= sample
        self.m_waitsample += sample
        while self.m_waitsample > MFCC_WAVCHUNK:
            self.m_waitsample -= MFCC_WAVCHUNK
            self.m_waitcnt += 1 
        if self.m_leftsample <= 0:
            self.m_waitcnt = self.m_calsize
    def readyall(self):
        self.m_waitcnt = self.m_calsize
        return 0 
    
    def isready(self):
        if self.m_waitcnt > self.m_calcnt:
            self.m_calcnt += 1
            return self.m_calcnt
        else:
            return 0 
    
    def finishone(self, index):
        self.m_resultcnt = index
        return 0 

    def calcbuf(self, calcinx):
        index = calcinx - 1
        if calcinx == self.m_calsize:  # 最后一个块，可能不满wav chunk 大小 
            melcnt = self.m_mellast
            bnfcnt = self.m_bnflast 
        else:
            melcnt = MFCC_MELBASE
            bnfcnt = MFCC_BNFBASE
        # 按照wav chunk进行计算的 
        pwav = self.m_wavmat[:, index, :]  # channel， 行，列 
        pmfcc = self.m_melmat[:, index, :]
        pbnf = self.m_bnfcache.secBuf(index)
        return pwav, pmfcc, pbnf, melcnt, bnfcnt
    