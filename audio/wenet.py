#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   wenet.py
@Time    :   2024/10/10 11:38:50
@Author  :   zhanglingling 
@Version :   1.0
@Contact :   None
@License :   None
@Desc    :   None
'''
from .netwav import * 
from .mfcc import log_mel 
import onnxruntime

# here put the import lib
class OnnxWenetModel:
    def __init__(self, model_path):
        self.session = onnxruntime.InferenceSession(model_path)
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        # # 打印期望的输入形状
        # for inp in self.session.get_inputs():
        #     print(f"输入 '{inp.name}' 期望形状 {inp.shape}")
        # # 打印期望的输入形状
        # for inp in self.session.get_outputs():
        #     print(f"输出 '{inp.name}' 期望形状 {inp.shape}")

    def run_model(self, melbin, melnum, bnfbin, bnfnum):
        #kwav = KWav(audio_path=audio_path)
        
        # 根据模型的期望输入形状调整输入形状
        melbin = np.array(melbin, dtype=np.float32).reshape(1, melnum, -1)  # 根据需要调整 期望输入 (B, T, 80)
        mel_lengths = np.array([melnum], dtype=np.int32)  # 更改为 int32

        # 运行模型
        inputs = {self.input_names[0]: melbin, self.input_names[1]: mel_lengths}
        outputs = self.session.run(self.output_names, inputs)
        #print(f"len of outputs: {len(outputs)}")
        #print(f"shape of output: {outputs[0].shape}")
        # 假设 bnfbin 用于存储输出  # 期望输出形状 (B, T_out, 256)
        bnfbin[:] = outputs[0]

        return 0
    def calcall(self, wavmat, mfcc_spec=None):
        rst = wavmat.readyall()
        cnt = 0
        while cnt < 1000:
            rst = wavmat.isready()
            if not rst:
                break
            self.calcinx(wavmat, rst, mfcc_spec)

    def calcinx(self, wavmat, calcinx, mfcc_spec=None):
        if calcinx > wavmat.m_calsize:
            return -1
        if calcinx < 1:
            return -2
        # index = calcinx - 1
        # if calcinx == wavmat.m_calsize:  # 最后一个块，可能不满wav chunk 大小 
        #     melcnt = wavmat.m_mellast
        #     bnfcnt = wavmat.m_bnflast 
        # else:
        #     melcnt = MFCC_MELBASE
        #     bnfcnt = MFCC_BNFBASE
        # # 按照wav chunk进行计算的 
        # pwav = wavmat.m_wavmat[:, index, :]
        # pmfcc = wavmat.m_melmat[:, index, :]
        # pbnf = wavmat.m_bnfcache.secBuf(index)
        pwav, pmfcc, pbnf, melcnt, bnfcnt = wavmat.calcbuf(calcinx)
        # audio_data = pwav.copy()
        # rst = 0
        # mel_cnt = int(MFCC_WAVCHUNK / 160) + 1
        mel_res = log_mel(pwav,  16000)
        #mel_res_librosa = log_mel_librosa(audio_data, 16000)
        # print(f'pwav shape: {pwav.shape}')
        # print(f'pmfcc shape: {pmfcc.shape}')
        # print(f'mel res shape {mel_res.shape}')  # shape (3501, 80)实际上不一定是3501长度
        if mfcc_spec is not None:
            pmfcc = mfcc_spec
            # 输入mfcc特征 输出wenet特征 
            mel_input =(mfcc_spec.reshape((MFCC_MELBASE, MFCC_MELCHUNK)))[:melcnt][np.newaxis]
        else:
            pmfcc = mel_res.flatten()[None, :]
            # 输入mfcc特征 输出wenet特征 
            mel_input = mel_res[:melcnt][np.newaxis]
        #print(f'mel input: {mel_input.shape}')
       
        pbnf_input = pbnf[:, :bnfcnt]
        #print(f'pbnf input: {pbnf_input.shape}')
        self.run_model(mel_input, melcnt, pbnf_input, bnfcnt)
        wavmat.finishone(calcinx)
        return 0 
