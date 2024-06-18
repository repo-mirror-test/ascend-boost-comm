/**
 * 
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
