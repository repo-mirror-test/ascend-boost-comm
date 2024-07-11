# Copyright(C) 2023. Huawei Technologies Co.,Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math
import unittest
import numpy as np
import sys, os
sys.path.append(f"{os.environ['MKI_HOME_PATH']}/tests/pythontest")
import op_test
import torch


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def align_up(val: int, align: int) -> int:
    if (align == 0):
        return 0
    return ((val + align - 1) // align) * align


def shape_nd_to_nz(shape, dtype='float16'):
    assert len(shape) >= 2
    batch = shape[:-2]
    a, b = shape[-2], shape[-1]
    a0, b0 = 16, 16
    return list(batch) + [math.ceil(b / b0), math.ceil(a / a0) * a0, b0]


def gen_axes_for_transpose(offset, base):
    return [x for x in range(offset)] + [x + offset for x in base]


def convert_nd_to_nz(x):
    array_trans = gen_axes_for_transpose(len(x.shape) - 2, [1, 0, 2])
    x_shape = shape_nd_to_nz(x.shape, dtype=x.dtype)
    *_, n1, m, n0 = x_shape

    # reshape为3维，对最后2维进行pad
    pad_num1 = align_up(x.shape[-1], 16) - x.shape[-1]
    pad_num2 = align_up(x.shape[-2], 16) - x.shape[-2]
    x = x.reshape([-1] + list(x.shape[-2:]))
    x = np.pad(x, ((0, 0), (0, pad_num2), (0, pad_num1)))

    x = x.reshape(x_shape[:-3] + [m, n1, n0]).transpose(*array_trans)
    x = np.ascontiguousarray(x)
    return x


class TestKvcache(op_test.OpTest):
    def case_param_gen(self, case_params_list):
        self.format = case_params_list['format']
        self.layer = case_params_list['layer']
        self.layer_id = case_params_list['layer_id']
        self.batch = case_params_list['batch']
        self.max_seqlen = case_params_list['max_seqlen']
        self.hidden_size = case_params_list['hidden_size']

    def golden_calc(self, in_tensors):
        newkv = self.newkv
        token_offset = self.token_offset
        seqlen = self.seqlen

        cache_out = np.zeros(shape=(self.layer, self.batch, self.max_seqlen, self.hidden_size)).astype(newkv.dtype)
        prefix_ntokens = 0
        for i in range(self.batch):
            for j in range(seqlen[i]):
                cache_out[self.layer_id][i][token_offset[i] - seqlen[i] + j][:] = newkv[prefix_ntokens + j][:]
            prefix_ntokens += seqlen[i]
        if self.format == self.format_nz:
            cache_out = convert_nd_to_nz(cache_out)
        return [torch.tensor(cache_out)]

    def golden_compare(self, out_tensors, golden_tensors):
        if (out_tensors[0].dtype == torch.bfloat16):
            return torch.allclose(out_tensors[0].bfloat16(), golden_tensors[0].bfloat16(), rtol=0.001, atol=0.001)
        return torch.allclose(out_tensors[0], golden_tensors[0], rtol=0.001, atol=0.001)
    # kvcache nd
    def test_kvcache_case0(self):
        self.set_support_910b_only()

        layer = 28
        layer_id = 0
        batch = 16
        max_seqlen = 384
        hidden_size = 1024

        OP_NAME = "KVCacheOperation"
        OP_PARAM = {"mixType": 0}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 5)
        self.set_output_formats([self.format_nd])

        case_params_list = {'format' : self.format_nd, 'layer': layer, 'layer_id' : layer_id, 'batch': batch,
                            'max_seqlen': max_seqlen, 'hidden_size': hidden_size}
        self.case_param_gen(case_params_list)

        seqlen = np.random.randint(1, max_seqlen // 2, size=batch, dtype=np.int32)
        token_offset = seqlen
        ntokens = np.sum(seqlen)
        newkv = np.random.uniform(-5, 5, size=(ntokens, hidden_size)).astype(np.float16)
        cache_in = np.zeros(shape=(layer, batch, max_seqlen, hidden_size)).astype(np.float16)
        layer_id = np.array([layer_id], dtype=np.int32)

        self.newkv = newkv
        self.token_offset = token_offset
        self.seqlen = seqlen

        cache_out = np.zeros_like(cache_in)
        return self.execute([torch.tensor(newkv).half(), torch.tensor(layer_id), torch.tensor(cache_in).half(),
                             torch.tensor(token_offset), torch.tensor(seqlen)],
                            [2])

    # kvcache nz 对齐
    def test_kvcache_case1(self):
        self.set_support_310p_only()

        layer = 1
        layer_id = 0
        batch = 16
        max_seqlen = 384
        hidden_size = 2048

        OP_NAME = "KVCacheOperation"
        OP_PARAM = {"mixType": 3}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nz, self.format_nd, self.format_nz, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nz])

        case_params_list = {'format' : self.format_nz, 'layer': layer, 'layer_id' : layer_id, 'batch': batch,
                            'max_seqlen': max_seqlen, 'hidden_size': hidden_size}
        self.case_param_gen(case_params_list)

        seqlen = np.random.randint(1, max_seqlen // 2, size=batch, dtype=np.int32)
        token_offset = np.zeros(shape=(batch)).astype(np.int32)
        for i in range(batch):
            token_offset[i] = seqlen[i] + 4
        ntokens = np.sum(seqlen)
        ntokens_ceil = (ntokens + 15) // 16 * 16
        newkv = np.random.uniform(-5, 5, size=(ntokens_ceil, hidden_size)).astype(np.float16)
        for i in range(ntokens, ntokens_ceil):
            newkv[i][:] = [0] * hidden_size
        cache_in = np.zeros(shape=(layer, batch, max_seqlen, hidden_size)).astype(np.float16)
        layer_id = np.array([layer_id], dtype=np.int32)

        # 用于生成golden数据
        self.newkv = newkv
        self.token_offset = token_offset
        self.seqlen = seqlen

        newkv = newkv.reshape([1] + list(newkv.shape[-2:]))
        newkv = convert_nd_to_nz(newkv)
        cache_in = convert_nd_to_nz(cache_in)
        cache_out = np.zeros_like(cache_in)

        return self.execute([torch.tensor(newkv).half(), torch.tensor(layer_id), torch.tensor(cache_in).half(),
                             torch.tensor(token_offset), torch.tensor(seqlen)],
                            [2])

    # kvcache nz 非对齐
    def test_kvcache_case2(self):
        self.set_support_310p_only()

        layer = 1
        layer_id = 0
        batch = 12
        max_seqlen = 1028
        hidden_size = 2047

        OP_NAME = "KVCacheOperation"
        OP_PARAM = {"mixType": 3}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nz, self.format_nd, self.format_nz, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nz])

        case_params_list = {'format' : self.format_nz, 'layer': layer, 'layer_id' : layer_id, 'batch': batch,
                            'max_seqlen': max_seqlen, 'hidden_size': hidden_size}
        self.case_param_gen(case_params_list)

        seqlen = np.random.randint(1, 1024, size=batch, dtype=np.int32)
        token_offset = np.zeros(shape=(batch)).astype(np.int32)
        for i in range(batch):
            token_offset[i] = seqlen[i] + 4
        ntokens = np.sum(seqlen)
        ntokens_ceil = (ntokens + 15) // 16 * 16
        newkv = np.random.uniform(-5, 5, size=(ntokens_ceil, hidden_size)).astype(np.float16)
        for i in range(ntokens, ntokens_ceil):
            newkv[i][:] = [0] * hidden_size
        cache_in = np.zeros(shape=(layer, batch, max_seqlen, hidden_size)).astype(np.float16)
        layer_id = np.array([layer_id], dtype=np.int32)

        # 用于生成golden数据
        self.newkv = newkv
        self.token_offset = token_offset
        self.seqlen = seqlen

        newkv = newkv.reshape([1] + list(newkv.shape[-2:]))
        newkv = convert_nd_to_nz(newkv)
        cache_in = convert_nd_to_nz(cache_in)
        cache_out = np.zeros_like(cache_in)

        return self.execute([torch.tensor(newkv).half(), torch.tensor(layer_id), torch.tensor(cache_in).half(),
                             torch.tensor(token_offset), torch.tensor(seqlen)],
                            [2])

    # kvcache nd
    def test_kvcache_case3(self):
        self.set_support_910b_only()

        layer = 28
        layer_id = 0
        batch = 16
        max_seqlen = 384
        hidden_size = 1024

        OP_NAME = "KVCacheOperation"
        OP_PARAM = {"mixType": 0}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 5)
        self.set_output_formats([self.format_nd])

        case_params_list = {'format' : self.format_nd, 'layer': layer, 'layer_id' : layer_id, 'batch': batch,
                            'max_seqlen': max_seqlen, 'hidden_size': hidden_size}
        self.case_param_gen(case_params_list)

        seqlen = np.random.randint(1, max_seqlen // 2, size=batch, dtype=np.int32)
        token_offset = seqlen
        ntokens = np.sum(seqlen)
        newkv = np.random.uniform(-5, 5, size=(ntokens, hidden_size)).astype(np.float16)
        cache_in = np.zeros(shape=(layer, batch, max_seqlen, hidden_size)).astype(np.float16)
        layer_id = np.array([layer_id], dtype=np.int32)

        self.newkv = newkv
        self.token_offset = token_offset
        self.seqlen = seqlen

        cache_out = np.zeros_like(cache_in)
        return self.execute([torch.tensor(newkv).bfloat16(), torch.tensor(layer_id), torch.tensor(cache_in).bfloat16(),
                             torch.tensor(token_offset), torch.tensor(seqlen)],
                            [2])

    def test_kvcache_nd_int8(self):
        self.set_support_910b_only()

        layer = 28
        layer_id = 0
        batch = 16
        max_seqlen = 384
        hidden_size = 1024

        OP_NAME = "KVCacheOperation"
        OP_PARAM = {"mixType": 0}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 5)
        self.set_output_formats([self.format_nd])

        case_params_list = {'format' : self.format_nd, 'layer': layer, 'layer_id' : layer_id, 'batch': batch,
                            'max_seqlen': max_seqlen, 'hidden_size': hidden_size}
        self.case_param_gen(case_params_list)

        seqlen = np.random.randint(1, max_seqlen // 2, size=batch, dtype=np.int32)
        token_offset = seqlen
        ntokens = np.sum(seqlen)
        newkv = np.random.uniform(-5, 5, size=(ntokens, hidden_size)).astype(np.int8)
        cache_in = np.zeros(shape=(layer, batch, max_seqlen, hidden_size)).astype(np.int8)
        layer_id = np.array([layer_id], dtype=np.int32)

        self.newkv = newkv
        self.token_offset = token_offset
        self.seqlen = seqlen

        return self.execute([torch.tensor(newkv), torch.tensor(layer_id), torch.tensor(cache_in),
                             torch.tensor(token_offset), torch.tensor(seqlen)],
                            [2])

if __name__ == '__main__':
    unittest.main()
