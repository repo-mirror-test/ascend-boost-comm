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
#include "loader.h"
#include "mki/base/operation_base.h"
#include "mki/base/kernel_base.h"
#include "mki/bin_handle.h"
#include "mki/utils/assert/assert.h"
#include "mki/utils/log/log.h"
#include "schedule/op_register.h"
#include "mki/utils/singleton/singleton.h"
#include <mki/utils/platform/platform_info.h>

namespace OpSpace {
Loader::Loader() { Load(); }

Loader::~Loader() {}

void Loader::GetAllOperations(std::unordered_map<std::string, Operation *> &ops) const
{
    ops = opMap_;
}

void Loader::GetOpKernels(const std::string &opName, KernelMap &kernels) const
{
    auto it = opKernelMap_.find(opName);
    if (it != opKernelMap_.end()) {
        kernels = it->second;
    }
}

bool Loader::IsValid() const
{
    return loadSuccess_;
}

void Loader::CreateOperations()
{
    auto &operationCreators = OperationRegister::GetOperationCreators();
    for (const auto &opCreator : operationCreators) {
        Operation *operation = opCreator();
        opMap_[operation->GetName()] = operation;
        MKI_LOG(DEBUG) << "Create operation " << operation->GetName();
    }
}

void Loader::CreateKernels()
{
    auto &kernelCreators = KernelRegister::GetKernelCreators();
    for (const auto &creatorInfo : kernelCreators) {
        const auto &kernelName = creatorInfo.kernelName;
        auto it = binHandles_.find(kernelName);
        if (it == binHandles_.end()) {
            MKI_LOG(WARN) << kernelName << " find bin handle fail";
            continue;
        }
        auto &handle = it->second;
        MKI_CHECK(handle.Init(kernelName), kernelName << " init handle fail", continue);

        auto kernelCreator = creatorInfo.func;
        MKI_CHECK(kernelCreator, kernelName << " creator function is null", continue);
        const Kernel *kernel = kernelCreator(&handle);
        MKI_CHECK(kernel != nullptr, "Invalid kernel found in op: " << kernelName, return);

        const auto &opName = creatorInfo.opName;
        auto &opKernel = opKernelMap_[opName];
        opKernel[kernelName] = kernel;
    }
}

bool Loader::LoadKernelBinarys()
{
    std::string deviceVersion = PlatformInfo::Instance().GetPlatformName();
    MKI_CHECK(deviceVersion != "unrecognized", "Get device soc version fail: " << deviceVersion, return false);

    const auto &kernelBinaryMap = KernelBinaryRegister::GetKernelBinaryMap();
    for (auto &item : kernelBinaryMap) {
        auto &binaryList = item.second;
        for (auto &binary : binaryList) {
            if (binary.targetSoc == deviceVersion) {
                binHandles_.emplace(item.first, &binary);
                MKI_LOG(DEBUG) << "Kernel " << deviceVersion << " find basic information success";
                break;
            }
        }
    }
    MKI_LOG(DEBUG) << "Loaded kernel Count: " << binHandles_.size();
    return true;
}

void Loader::Load()
{
    loadSuccess_ = false;

    MKI_CHECK(LoadKernelBinarys(), "Load kernel binarys fail", return);
    CreateOperations();
    CreateKernels();
    for (const auto &[opName, op] : opMap_) {
        MKI_LOG(DEBUG) << "mki load operation: " << opName;
        OperationBase *opBase = dynamic_cast<OperationBase *>(op);
        MKI_CHECK(opBase != nullptr, opName << ": opBase is nullptr", return);
        auto it = opKernelMap_.find(opName);
        if (it == opKernelMap_.end()) {
            MKI_LOG(WARN) << opName << ": find kernels map fail ";
            continue;
        }
        auto &nameKernelsMap = it->second;
        for (const auto &[kernelName, kernel] : nameKernelsMap) {
            opBase->AddKernel(kernelName, kernel);
        }
    }
    loadSuccess_ = true;
}
} // namespace OpSpace
