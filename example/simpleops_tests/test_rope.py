# 
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# AscendOpCommonLib is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# 
import unittest
import random
import logging
import numpy as np
import torch
import sys, os
sys.path.append(f"{os.environ['MKI_HOME_PATH']}/tests/pythontest")

import op_test

OP_NAME = "RopeOperation"

class TestRopeOperation(op_test.OpTest):
    def golden_calc(self, in_tensors):
        q = np.array(in_tensors[0].cpu()).astype(np.float16)
        kk = np.array(in_tensors[1].cpu()).astype(np.float16)
        cos = np.array(in_tensors[2].cpu()).astype(np.float16)
        sin = np.array(in_tensors[3].cpu()).astype(np.float16)
        seqlen = np.array(in_tensors[4].cpu()).astype(np.int32)

        batch = q.shape[0]
        rotaryCoeff = self.op_desc["specificParam"]["rotaryCoeff"]
        headDim = cos.shape[-1]
        hiddensizeQ = q.shape[-1]
        hiddensizeK = kk.shape[-1]
        hiddensize = max(hiddensizeQ, hiddensizeK)
        headNumQ = hiddensizeQ // headDim
        headNumK = hiddensizeK // headDim
        headNum = max(headNumQ, headNumK)
        ntokens = np.sum(seqlen)

        rope_q = np.zeros(shape=(ntokens, hiddensizeQ)).astype(np.float16)
        rope_k = np.zeros(shape=(ntokens, hiddensizeK)).astype(np.float16)
        prefix_Ntokens = 0
        cosTable = np.zeros(shape=(ntokens, hiddensize)).astype(np.float16)

        for i in range(ntokens):
            for j in range(headNum):
                cosTable[i][j*headDim:(j+1)*headDim] = cos[i][:]
        for i in range(batch):
            curr_seqLen = seqlen[i][0]
            q1 = np.zeros(shape=(curr_seqLen, hiddensizeQ)).astype(np.float16)
            k1 = np.zeros(shape=(curr_seqLen, hiddensizeK)).astype(np.float16)

            for i in range(prefix_Ntokens, prefix_Ntokens + curr_seqLen):
                q1[i-prefix_Ntokens] = q[i] * cosTable[i][:hiddensizeQ]
                k1[i-prefix_Ntokens] = kk[i] * cosTable[i][:hiddensizeK] 
            q2 = np.zeros(shape=(curr_seqLen, hiddensizeQ)).astype(np.float16)
            k2 = np.zeros(shape=(curr_seqLen, hiddensizeK)).astype(np.float16)        
            for k in range(headNum):
                src_ = k * headDim
                dst_ = (k + 1) * headDim
                strdie = headDim // 2
                rotaryStrdie = headDim // rotaryCoeff
                rotaryTimesPerHead = rotaryCoeff / 2
                for cycle in range(int(rotaryTimesPerHead)):
                    src =  src_ + cycle * rotaryStrdie * 2
                    dst = src + rotaryStrdie * 2
                    for curr_seqLeni in range(curr_seqLen):
                        if k < headNumQ:
                            q2[curr_seqLeni][src:src + rotaryStrdie] = q[prefix_Ntokens + curr_seqLeni][src+ rotaryStrdie:dst] * (-1)
                            q2[curr_seqLeni][src + rotaryStrdie:dst] = q[prefix_Ntokens + curr_seqLeni][src:src+rotaryStrdie]
                            q2[curr_seqLeni][src:dst] = q2[curr_seqLeni][src:dst] * sin[prefix_Ntokens + curr_seqLeni][cycle * rotaryStrdie * 2: (cycle +1) * rotaryStrdie * 2]
                        if k < headNumK:
                            k2[curr_seqLeni][src:src + rotaryStrdie] = kk[prefix_Ntokens + curr_seqLeni][src+ rotaryStrdie:dst] * (-1)
                            k2[curr_seqLeni][src + rotaryStrdie:dst] = kk[prefix_Ntokens + curr_seqLeni][src:src+rotaryStrdie]
                            k2[curr_seqLeni][src:dst] = k2[curr_seqLeni][src:dst] * sin[prefix_Ntokens + curr_seqLeni][cycle * rotaryStrdie * 2: (cycle +1) * rotaryStrdie * 2]
            rope_q[prefix_Ntokens:prefix_Ntokens + curr_seqLen] += q1 + q2
            rope_k[prefix_Ntokens:prefix_Ntokens + curr_seqLen] += k1 + k2      
            
            prefix_Ntokens += curr_seqLen
        return [torch.tensor(rope_q), torch.tensor(rope_k)]

    def golden_compare(self, out_tensors, golden_out_tensors):
        print(torch.allclose(out_tensors[0], golden_out_tensors[0], rtol=0.001, atol=0.001))
        print(torch.allclose(out_tensors[1], golden_out_tensors[1], rtol=0.001, atol=0.001))

        return torch.allclose(out_tensors[0], golden_out_tensors[0], rtol=0.001, atol=0.001) and torch.allclose(out_tensors[1], golden_out_tensors[1], rtol=0.001, atol=0.001)

    def test_rope_larger_qk(self):
        '''
            基础场景
            shape: (320, 16640) / (320, 9216) / (3, 256) / (3, 256)
        '''
        batch = 320
        rotaryCoeff = 2
        headDim = 256
        OP_PARAM0 = {"rotaryCoeff": rotaryCoeff}

        hiddensizeQ = 16640
        hiddensizeK = 9216
        hiddensize = max(hiddensizeQ, hiddensizeK)

        headNumQ = hiddensizeQ // headDim
        headNumK = hiddensizeK // headDim
        headNum = max(headNumQ, headNumK)
        q = np.random.uniform(-1, 1, size=(batch, hiddensizeQ)).astype(np.float16)
        kk = np.random.uniform(-1, 1, size=(batch, hiddensizeK)).astype(np.float16)
        cos = np.random.uniform(-1, 1, size=(batch, headDim)).astype(np.float16)
        sin = np.random.uniform(-1, 1, size=(batch, headDim)).astype(np.float16)
        seqlen =torch.randint(1, 2, size=(batch,1), dtype=torch.int32)
        ntokens = torch.sum(seqlen)
        for i in range(ntokens):
            for j in range(headDim//2):
                cos[i][(2*j+1):(2*j+2)] =  cos[i][(2*j):(2*j+1)]
                sin[i][(2*j+1):(2*j+2)] =  sin[i][(2*j):(2*j+1)]
        
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([torch.tensor(q).half(), torch.tensor(kk).half(), torch.tensor(cos).half(), torch.tensor(sin).half(), torch.tensor(seqlen).int()],
                     [torch.zeros(q.shape).half(), torch.zeros(kk.shape).half()])

    def test_rope_random(self):
        for i in range(50):
            logging.info("index is %d", i)
            batch = np.random.randint(1, 400)
            headDim = random.randint(1, 20) * 16
            rotaryCoeff = random.choice([2, 4, headDim])
            highPrecision = 1
            OP_PARAM0 = {"rotaryCoeff": rotaryCoeff}
            seqlen =np.random.randint(1, 2, size=batch, dtype=np.int32)
            ntokens = np.sum(seqlen)

            hiddensizeQ = random.randint(51, 100) * headDim
            hiddensizeK = random.randint(1, 50) * headDim

            hiddensize = max(hiddensizeQ, hiddensizeK)
            q = np.random.uniform(-1, 1, size=(batch, hiddensizeQ)).astype(np.float16)
            kk = np.random.uniform(-1, 1, size=(batch, hiddensizeK)).astype(np.float16)
            cos = np.random.uniform(-1, 1, size=(batch, headDim)).astype(np.float16)
            sin = np.random.uniform(-1, 1, size=(batch, headDim)).astype(np.float16)
            seqlen =torch.randint(1, 2, size=(batch,1), dtype=torch.int32)

            logging.info("q shape is %s", q.shape)
            logging.info("kk shape is %s", kk.shape)
            logging.info("cos shape is %s", cos.shape)
            logging.info("sin shape is %s", sin.shape)
            logging.info("rotary coefficient is %d", rotaryCoeff)


            ntokens = torch.sum(seqlen)
            for i in range(ntokens):
                for j in range(headDim//2):
                    cos[i][(2*j+1):(2*j+2)] =  cos[i][(2*j):(2*j+1)]
                    sin[i][(2*j+1):(2*j+2)] =  sin[i][(2*j):(2*j+1)]
            
            self.set_param(OP_NAME, OP_PARAM0)
            self.execute([torch.tensor(q).half(), torch.tensor(kk).half(), torch.tensor(cos).half(), torch.tensor(sin).half(), torch.tensor(seqlen).int()],
                        [torch.zeros(q.shape).half(), torch.zeros(kk.shape).half()])

if __name__ == '__main__':
    unittest.main()
