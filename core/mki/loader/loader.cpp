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
#include "mki/utils/assert/assert.h"
#include "mki/utils/log/log.h"
#include "mki/base/op_register.h"
#include "mki/utils/singleton/singleton.h"
#include "mki/loader/device_kernel_loader.h"

namespace Mki {
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
    auto &operationCreators = OperationRegister::OperationCreators();
    for (const auto &opCreator : operationCreators) {
        Operation *operation = opCreator();
        opMap_[operation->GetName()] = operation;
        MKI_LOG(DEBUG) << "Create operation " << operation->GetName();
    }
}

void Loader::CreateKernels()
{
    auto &kernelCreators = KernelRegister::KernelCreators();
    for (const auto &[kernelCreator, opName] : kernelCreators) {
        auto &opKernel = opKernelMap_[opName];
        const Kernel *kernel = kernelCreator();
        MKI_CHECK(kernel != nullptr, "Invalid kernel found in op: " << opName, continue);
        if (GetSingleton<KernelBinaryLoader>().HasKernelMetaInfo(kernel->GetName())) {
            opKernel[kernel->GetName()] = kernel;
        }
    }
}

void Loader::Load()
{
    loadSuccess_ = false;

    CreateOperations();
    CreateKernels();
    for (const auto &[opName, op] : opMap_) {
        MKI_LOG(DEBUG) << "mki load operation: " << opName;
        OperationBase *opBase = reinterpret_cast<OperationBase *>(op);  // TODO: 为什么dynamic_cast不行，什么原因？
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
} // namespace Mki
