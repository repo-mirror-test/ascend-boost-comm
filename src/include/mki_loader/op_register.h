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
#ifndef CORE_BASE_KERNEL_REGISTER_H
#define CORE_BASE_KERNEL_REGISTER_H

#include <map>
#include <vector>

#include "mki/base/kernel_base.h"
#include "mki/base/operation_base.h"
#include "mki/utils/assert/assert.h"
#include "mki/utils/log/log.h"

#define MACRO_TO_STR(name) #name
#define MKI_NAMESPACE_LOG(level, spaceName) MKI_LOG(level) << (MACRO_TO_STR(spaceName))

using namespace Mki;
namespace OpSpace {
using NewOperationFunc = Operation*(*)();
using NewKernelFunc = const Kernel*(*)(const BinHandle *);

class OperationRegister {
public:
    OperationRegister(const char *opName, NewOperationFunc func) noexcept
    {
        MKI_CHECK(opName != nullptr, "opName is nullptr", return);
        auto &operationCreators = GetOperationCreators();
        operationCreators.push_back(func);
        MKI_NAMESPACE_LOG(DEBUG, OpSpace) << " register operation " << opName;
    }

    static std::vector<NewOperationFunc> &GetOperationCreators()
    {
        static std::vector<NewOperationFunc> operationCreators;
        return operationCreators;
    }
};

class KernelRegister {
public:
    struct CreatorInfo {
        NewKernelFunc func;
        std::string opName;
        std::string kernelName;
    };

    KernelRegister(const char *opName, const char *kerName, NewKernelFunc func) noexcept
    {
        MKI_CHECK(opName != nullptr, "opName is nullptr", return);
        MKI_CHECK(kerName != nullptr, "kerName is nullptr", return);
        auto &kernelCreators = GetKernelCreators();
        kernelCreators.push_back({ func, opName, kerName });
        MKI_NAMESPACE_LOG(DEBUG, OpSpace) << " register kernel " << kerName << " of operation " << opName;
    }

    static std::vector<CreatorInfo> &GetKernelCreators()
    {
        static std::vector<CreatorInfo> kernelCreators;
        return kernelCreators;
    }
};

class KernelBinaryRegister {
public:
    KernelBinaryRegister(const char *soc, const char *kernelName, const uint8_t *binary, uint32_t binaryLen) noexcept
    {
        MKI_CHECK(kernelName != nullptr, "kernelName is nullptr", return);
        MKI_CHECK(soc != nullptr, "target soc is nullptr", return);
        MKI_CHECK(binary != nullptr, "binary addr is nullptr", return);
        auto &binaryMap = GetKernelBinaryMap();
        binaryMap[std::string(kernelName)].push_back({ binary, binaryLen, soc });
        MKI_NAMESPACE_LOG(DEBUG, OpSpace) << " register kernel binary " << kernelName << "-" << soc << " success";
    }

    static std::map<std::string, std::vector<BinaryBasicInfo>> &GetKernelBinaryMap()
    {
        static std::map<std::string, std::vector<BinaryBasicInfo>> binaryMap;
        return binaryMap;
    }
};

#define REG_OPERATION(opName)                                                                                          \
    Operation *GetOperation##opName()                                                                                  \
    {                                                                                                                  \
        static opName op##opName(#opName);                                                                             \
        return &op##opName;                                                                                            \
    }                                                                                                                  \
    static OperationRegister op##opName##register = OperationRegister(#opName, GetOperation##opName)

#define REG_KERNEL_BASE(kerName)                                                                                       \
    Kernel const *GetKernel##kerName(const BinHandle *binHandle)                                                       \
    {                                                                                                                  \
        static kerName ker##kerName(#kerName, binHandle);                                                              \
        SetKernelSelfCreator(ker##kerName, [=](){ return new kerName(#kerName, binHandle); });                         \
        return &ker##kerName;                                                                                          \
    }                                                                                                                  \
    static KernelRegister ker##kerName##register = KernelRegister(OperationPlaceHolder, #kerName, GetKernel##kerName)

#define REG_KERNEL(soc, kerName, binary)                                                                               \
    static KernelBinaryRegister bin##kerName##soc##register = KernelBinaryRegister(#soc, #kerName, binary, sizeof(binary))
} // namespace OpSpace

#endif
