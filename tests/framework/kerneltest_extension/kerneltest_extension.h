/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * MindKernelInfra is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */
#ifndef MKI_KERNELTEST_EXTENSION_H
#define MKI_KERNELTEST_EXTENSION_H

#include <cstdint>

#include "mki/types.h"

extern "C" uint32_t PrintAscendKernel(const void *stream,
    void *dumpAddr, uint64_t dumpSizePerCore, uint32_t blockDim, uint32_t taskType);

#define CALL_ACL(x)                                                                                                    \
    if (auto ret = (x) != 0) {                                                                                         \
        printf("[ERROR] failed to exec acl api %s, result %d\n", #x, ret);                                             \
        return -1;                                                                                                     \
    } else {                                                                                                           \
        printf("[INFO] Succeeded to exec acl api %s\n", #x);                                                           \
    }

#ifdef __cplusplus
extern "C" {
#endif

struct MkiTensor {
    Mki::TensorDType dtype;
    Mki::TensorFormat format;
    uint64_t dimsLen;
    int64_t dims[16];
};

struct MkiOpDesc {
    const char *opName;
    const char *specificParam;
    MkiTensor inTensors[8];
    uint32_t inTensorsLen;
    MkiTensor outTensors[8];
    uint32_t outTensorsLen;
};

struct MkiTensorData {
    uint8_t *hostData;
    uint8_t *deviceData;
    uint64_t dataLen;
};

void *GetStream(uint32_t deviceId);

MkiTensorData GetTilingData(MkiOpDesc *opDesc, uint32_t *blockDim);

MkiTensorData RandF16Tensor(MkiTensor *desc, float min = 0.0, float max = 1.0);
MkiTensorData RandF32Tensor(MkiTensor *desc, float min = 0.0, float max = 1.0);
MkiTensorData RandI8Tensor(MkiTensor *desc, int8_t min = -128, int8_t max = 127);
MkiTensorData RandI32Tensor(MkiTensor *desc, int32_t min = -128, int32_t max = 127);

MkiTensorData ReadFile(MkiTensor *desc, const char *file);
void PrintTensorData(MkiTensor *desc, MkiTensorData *td, uint64_t printMaxCount = 10);

void FreeTensor(MkiTensorData &td);

#ifdef __cplusplus
}
#endif

#endif
