/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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

#include "mki/bin_handle.h"
#include "mki/utils/assert/assert.h"
#include "mki/utils/rt/rt.h"

namespace Mki {
constexpr uint32_t BYTE_SIZE = 8;
constexpr uint32_t HEADER_LENGTH = 128;
constexpr uint32_t UINT32_TYPE_LENGTH = sizeof(uint32_t);

struct KernelHeaderInfo {
    uint32_t version;
    uint32_t magic;
    uint32_t tilingSize;
    uint32_t coreType;
    uint32_t kernelNum;
    uint32_t kernelNameOffset;
    uint32_t compileInfoOffset;
    uint32_t kernelBinOffset;
};

BinHandle::BinHandle(const BinaryBasicInfo *binInfo) : basicInfo_(binInfo) {}

BinHandle::~BinHandle() {}

KernelHandle BinHandle::GetHandle() const { return &handle_; }

bool BinHandle::CheckBinaryValid() const
{
    MKI_CHECK(basicInfo_ != nullptr, "basicInfo is nullptr", return false);
    MKI_CHECK(basicInfo_->binaryBuf != nullptr, "binary info is empty!", return false);
    MKI_CHECK(basicInfo_->binaryLen > HEADER_LENGTH, "binary length is less than header length!", return false);
    return true;
}

bool BinHandle::CheckKernelInfo(const std::string &kernelName) const
{
    MKI_CHECK(!metaInfo_.kernelList.empty(), "Get Binary Kernel Num empty, kernel: " << kernelName, return false);
    MKI_CHECK(metaInfo_.magic != 0, "Get Magic empt, kernel: " << kernelName, return false);
    for (const std::string &kernel : metaInfo_.kernelList) {
        MKI_LOG(DEBUG) << kernelName << ".kernelName: " << kernel;
    }
    MKI_LOG(DEBUG) << kernelName << ".devicecodeBufLen: " << metaInfo_.codeBufLen;
    return true;
}

bool BinHandle::Init(const std::string &kernelName)
{
    codeLoadSuccess_ = false;

    MKI_CHECK(CheckBinaryValid(), "basicInfo is invalid", return false);

    const KernelHeaderInfo &header = *reinterpret_cast<const KernelHeaderInfo *>(basicInfo_->binaryBuf);
    const uint8_t *data = basicInfo_->binaryBuf + HEADER_LENGTH;
    const uint8_t *maxAddr = basicInfo_->binaryBuf + basicInfo_->binaryLen;
    metaInfo_.version = header.version;
    metaInfo_.magic = header.magic;
    metaInfo_.tilingSize = header.tilingSize;
    metaInfo_.coreType = header.coreType;
    uint32_t kernelNum = header.kernelNum;
    const uint8_t *kernelNameStart = data + header.kernelNameOffset;
    for (size_t count = 0; count < kernelNum; ++count) {
        MKI_CHECK(kernelNameStart + UINT32_TYPE_LENGTH <= maxAddr, "length error", return false);
        uint32_t kernelNameSize = *reinterpret_cast<const uint32_t *>(kernelNameStart);
        kernelNameStart += UINT32_TYPE_LENGTH;
        MKI_CHECK(kernelNameStart + kernelNameSize <= maxAddr, "length error", return false);
        std::string str = (const char *)kernelNameStart;
        MKI_CHECK(str.length() < kernelNameSize, "length error", return false);
        metaInfo_.kernelList.push_back(str);
        kernelNameStart += kernelNameSize;
    }
    const uint8_t *compileInfoStart = data + header.compileInfoOffset;
    MKI_CHECK(compileInfoStart + UINT32_TYPE_LENGTH <= maxAddr, "length error", return false);
    uint32_t compileInfoSize = *reinterpret_cast<const uint32_t *>(compileInfoStart);
    compileInfoStart += UINT32_TYPE_LENGTH;
    metaInfo_.compileInfo = (const char *)compileInfoStart;
    MKI_CHECK(metaInfo_.compileInfo.length() < compileInfoSize, "length error", return false);

    const uint8_t *kernelBinStart = data + header.kernelBinOffset;
    MKI_CHECK(kernelBinStart + UINT32_TYPE_LENGTH <= maxAddr, "length error", return false);
    uint32_t kernelBinSize = *reinterpret_cast<const uint32_t *>(kernelBinStart);
    MKI_CHECK(kernelBinStart + UINT32_TYPE_LENGTH + kernelBinSize == maxAddr, "length error", return false);
    metaInfo_.codeBuf = const_cast<uint8_t *>(kernelBinStart + UINT32_TYPE_LENGTH);
    metaInfo_.codeBufLen = kernelBinSize;

    MKI_CHECK(CheckKernelInfo(kernelName), kernelName << " check kernel info error", return false);
    MKI_CHECK(RegisterBin(kernelName), kernelName << " register kernel fail", return false);

    codeLoadSuccess_ = true;
    return true;
}

uint32_t BinHandle::GetKernelTilingSize() const
{
    if (!codeLoadSuccess_) {
        return 0;
    }
    uint32_t tilingSize = metaInfo_.tilingSize;
    return ceil(static_cast<double>(tilingSize) / BYTE_SIZE) * BYTE_SIZE;
}

int32_t BinHandle::GetKernelCoreType() const
{
    if (!codeLoadSuccess_) {
        return -1;
    }
    return static_cast<int32_t>(metaInfo_.coreType);
}

const char *BinHandle::GetKernelCompileInfo() const
{
    if (!codeLoadSuccess_) {
        return nullptr;
    }
    return metaInfo_.compileInfo.c_str();
}

bool BinHandle::RegisterBin(const std::string &kernelName)
{
    size_t kernelNum = metaInfo_.kernelList.size();
    MKI_CHECK(kernelNum != 0, "Get Binary Kernel Num empty, op: " << kernelName, return false);

    MkiRtModuleInfo moduleInfo;
    moduleInfo.type = MKIRT_MODULE_OBJECT;
    moduleInfo.version = 0;
    moduleInfo.data = metaInfo_.codeBuf;
    moduleInfo.dataLen = metaInfo_.codeBufLen;
    moduleInfo.magic = metaInfo_.magic;
    if (metaInfo_.kernelList.size() == 1) {
        MKI_CHECK_NO_LOG(RegisterBinWithSingleKernel(kernelName, moduleInfo), return false);
    } else {
        MKI_CHECK_NO_LOG(RegisterBinWithMultiKernel(kernelName, moduleInfo), return false);
    }
    return true;
}

bool BinHandle::RegisterBinWithSingleKernel(const std::string &kernelName, MkiRtModuleInfo &moduleInfo)
{
    MKI_LOG(DEBUG) << "SingleKernel RegisterBin start, opName:" << kernelName;
    int st = MkiRtModuleCreate(&moduleInfo, &moduleHandle_);
    MKI_CHECK(st == MKIRT_SUCCESS, kernelName << " Create RtModule fail, error:" << st, return false);

    MKI_CHECK(moduleHandle_ != nullptr, kernelName << " Create RtModule fail,"
                                                   << " because it return false null handle", return false);

    st = MkiRtModuleBindFunction(moduleHandle_, metaInfo_.kernelList[0].c_str(), &handle_);
    MKI_CHECK(st == MKIRT_SUCCESS, kernelName << " Mki RtModuleGetFunction fail, errCode:" << st
                                              << ", errName:" << MkiRtErrorName(st)
                                              << ", errDesc:" << MkiRtErrorDesc(st), return false);

    MKI_LOG(DEBUG) << "SingleKernel RegisterBin finish, handle:" << handle_;
    return true;
}

bool BinHandle::RegisterBinWithMultiKernel(const std::string &kernelName, MkiRtModuleInfo &moduleInfo)
{
    MKI_LOG(DEBUG) << "MultiKernel RegisterBin start, opName:" << kernelName;
    int st = AstRtRegisterAllFunction(&moduleInfo, &handle_);
    MKI_CHECK(st == MKIRT_SUCCESS, kernelName << " Mki RtRegister AllFunction fail, error:" << st, return false);

    MKI_CHECK(handle_ != nullptr, kernelName << " Mki RtRegister AllFunction fail,"
                                             << " because it return false null handle" << st, return false);
    MKI_LOG(DEBUG) << "MultiKernel RegisterBin finish";
    return true;
}
}
