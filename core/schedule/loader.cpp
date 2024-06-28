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

namespace {
const std::unordered_map<std::string, std::string> DEVICE_VERSION_MAP {
    // 310p
    {"Ascend310P", "ascend310p"}, {"Ascend310P3", "ascend310p"},
    // 910
    {"Ascend910A", "ascend910"}, {"Ascend910", "ascend910"},
    // 910b
    {"Ascend910B", "ascend910b"}, {"Ascend910B1", "ascend910b"}, {"Ascend910B2", "ascend910b"},
    {"Ascend910B2C", "ascend910b"}, {"Ascend910B3", "ascend910b"}, {"Ascend910B4", "ascend910b"},
    // 910c
    {"Ascend910C", "ascend910b"}, {"Ascend910C1", "ascend910b"}, {"Ascend910C2", "ascend910b"},
    {"Ascend910C3", "ascend910b"}, {"Ascend910C4", "ascend910b"},
    // 310b
    {"Ascend310B1", "ascend310b"}, {"Ascend310B2", "ascend310b"}, {"Ascend310B3", "ascend310b"},
    {"Ascend310B4", "ascend310b"}
};
}

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

bool Loader::GetDeviceSocVersion(std::string &deviceVersion)
{
    const uint32_t deviceVersionLen = 20;
    char socVersion[deviceVersionLen];
    MKI_CHECK(MkiRtDeviceGetSocVersion(socVersion, deviceVersionLen) == MKIRT_SUCCESS,
        "Get device version failed!", return false);
    const auto item = DEVICE_VERSION_MAP.find(socVersion);
    MKI_CHECK(item != DEVICE_VERSION_MAP.end(), "Unrecognized device version!", return false);
    deviceVersion = item->second;
    return true;
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
    std::string deviceVersion;
    MKI_CHECK(GetDeviceSocVersion(deviceVersion), "Get device soc version fail", return false);

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
} // namespace OpSpace
