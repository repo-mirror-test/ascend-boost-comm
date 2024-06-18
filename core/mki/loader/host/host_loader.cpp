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
#include "host_loader.h"
#include <cstring>
#include "mki/kernel.h"
#include "mki/base/kernel_base.h"
#include "mki/base/op_register.h"
#include "mki/utils/assert/assert.h"
#include "mki/utils/log/log.h"
#include "mki/utils/filesystem/filesystem.h"
#include "mki/utils/strings/str_checker.h"
#include "mki/utils/singleton/singleton.h"
#include "mki/loader/device_kernel/device_kernel_loader.h"
// #include "host_so_dl.h"

namespace Mki {

HostLoader::HostLoader()
{
    // const char *mkiHome = std::getenv("MKI_HOME_PATH");
    // MKI_CHECK(mkiHome != nullptr, "env MKI_HOME_PATH not exists", return);

    // soLoadSuccess_ = LoadFromSo(mkiHome, "ops_lib/");
    // MKI_CHECK(soLoadSuccess_, ".so load failed", return);

    CreateOperations();
    CreateKernels();
    MKI_LOG(DEBUG) << "Operation & kernel loaded!";
}

HostLoader::~HostLoader() {}

// bool HostLoader::IsValid() const
// {
//     return soLoadSuccess_;
// }

void HostLoader::GetAllOperations(std::unordered_map<std::string, Operation *> &ops) const
{
    ops = opMap_;
}

void HostLoader::GetOpKernels(const std::string &opName, KernelMap &kernels) const
{
    auto it = opKernelMap_.find(opName);
    if (it != opKernelMap_.end()) {
        kernels = it->second;
    }
}

// bool HostLoader::LoadFromSo(const std::string &mkiHome, const std::string &soPath)
// {
//     std::vector<std::string> hostSoPaths = {mkiHome, soPath};
//     std::string hostSoPath = FileSystem::Join(hostSoPaths);
//     MKI_CHECK(CheckNameValid(hostSoPath), "invalid hostSoPath", return false);
//     MKI_CHECK(FileSystem::Exists(hostSoPath), "hostSoPath is not exit", return false);

//     std::vector<std::string> soFilePaths;
//     FileSystem::GetDirChildFiles(hostSoPath, soFilePaths);

//     bool validSoExists = false;
//     for (auto &filePath: soFilePaths) {
//         const char *fileName = std::strrchr(filePath.c_str(), '/');
//         MKI_CHECK(fileName != nullptr, "Invalid filePath: " << filePath, continue);

//         const char *ext = std::strrchr(fileName, '.');
//         MKI_CHECK(ext != nullptr, "Invalid file extension: " << fileName, continue);

//         std::string extStr(ext);
//         MKI_CHECK(extStr.compare(".so") == 0, "Invalid file extension: " << fileName, continue);

//         MKI_LOG(INFO) << "Load so: " << filePath;

//         auto hostSoDl = std::make_unique<Dl>(filePath);
//         if (!hostSoDl->IsValid()) continue;
//         validSoExists = true;
//         hostSoDls_.push_back(std::move(hostSoDl));
//     }
//     if (!validSoExists) return false;

//     CreateOperations();
//     CreateCreators();

//     return validSoExists;
// }

void HostLoader::CreateOperations()
{
    auto &operationCreators = OperationRegister::OperationCreators();
    for (const auto &opCreator : operationCreators) {
        Operation *operation = opCreator();
        opMap_[operation->GetName()] = operation;
        MKI_LOG(DEBUG) << "Create operation " << operation->GetName();
    }
}

void HostLoader::CreateKernels()
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
} // namespace Mki
