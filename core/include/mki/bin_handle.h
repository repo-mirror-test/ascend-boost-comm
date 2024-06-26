/**
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
#ifndef MKI_BIN_HANDLE_H
#define MKI_BIN_HANDLE_H
#include <string>
#include <atomic>
#include "mki/tensor_desc.h"
#include "mki/utils/rt/base/base.h"
#include "mki/utils/noncopyable/noncopyable.h"

namespace Mki {
using KernelHandle = void**;

struct BinaryBasicInfo {
    const uint8_t *binaryBuf = nullptr;
    uint32_t binaryLen = 0;
    std::string targetSoc;
};

struct KernelMetaInfo {
    std::vector<std::string> kernelList;
    std::string compileInfo;
    uint32_t version;
    uint32_t tilingSize = 0;
    uint32_t coreType;
    uint32_t magic = 0x41494343U;
    void *codeBuf = nullptr;
    uint32_t codeBufLen = 0;
};

class BinHandle : public NonCopyable {
public:
    BinHandle(const BinaryBasicInfo *binInfo);
    ~BinHandle();
    KernelHandle GetHandle();
    bool Init(const std::string &kernelName);
    uint32_t GetKernelTilingSize() const;
    int32_t GetKernelCoreType() const;
    const char *GetKernelCompileInfo() const;

private:
    bool CheckBinaryValid() const;
    bool CheckKernelInfo(const std::string &kernelName) const;
    bool RegisterBin(const std::string &kernelName);
    bool RegisterBinWithSingleKernel(const std::string &kernelName, MkiRtModuleInfo &moduleInfo);
    bool RegisterBinWithMultiKernel(const std::string &kernelName, MkiRtModuleInfo &moduleInfo);

private:
    KernelMetaInfo metaInfo_;
    void *handle_ = nullptr;
    void *moduleHandle_ = nullptr;
    BinaryBasicInfo const *basicInfo_ = nullptr;
    std::atomic_bool codeLoadSuccess_ = false;
};
} // namespace Mki

#endif
