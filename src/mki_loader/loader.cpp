/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "loader.h"
#include <sys/stat.h>
#include "mki/base/operation_base.h"
#include "mki/base/kernel_base.h"
#include "mki/bin_handle.h"
#include "mki/utils/assert/assert.h"
#include "mki/utils/env/env.h"
#include "mki/utils/log/log.h"
#include "mki/utils/platform/platform_info.h"
#include "mki/utils/rt/rt.h"
#include "mki_loader/op_register.h"

namespace Mki {
Loader::Loader(const OperationCreators &operationCreators, const KernelCreators &kernelCreators,
               const BinaryBasicInfoMap &binaryMap)
    : operationCreators_(operationCreators), kernelCreators_(kernelCreators), binaryMap_(binaryMap) { Load(); }

Loader::~Loader() {}

const std::unordered_map<std::string, Operation *> &Loader::GetAllOperations() const
{
    return opMap_;
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

bool Loader::CreateOperations()
{
    for (const auto &opCreator : operationCreators_) {
        Operation *operation = opCreator();
        MKI_CHECK(operation != nullptr, "create operation fail", return false);
        opMap_[operation->GetName()] = operation;
        MKI_LOG(DEBUG) << "Create operation " << operation->GetName();
    }
    return true;
}

bool Loader::CreateKernels()
{
    for (const auto &creatorInfo : kernelCreators_) {
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
        MKI_CHECK(kernel != nullptr, "Invalid kernel found in op: " << kernelName, return false);

        const auto &opName = creatorInfo.opName;
        auto &opKernel = opKernelMap_[opName];
        opKernel[kernelName] = kernel;
    }
    return true;
}

bool Loader::CreateAicpuKernels()
{
    auto &kernelCreators = AicpuKernelRegister::GetKernelCreators();
    for (const auto &creatorInfo : kernelCreators) {
        const auto &kernelName = creatorInfo.kernelName;

        auto kernelCreator = creatorInfo.func;
        MKI_CHECK(kernelCreator, kernelName << " creator function is null", continue);
        const Mki::Kernel *kernel = kernelCreator();
        MKI_CHECK(kernel != nullptr, "Invalid kernel found in op: " << kernelName, return false);

        const auto &opName = creatorInfo.opName;
        auto &opKernel = opKernelMap_[opName];
        opKernel[kernelName] = kernel;
    }

    if (kernelCreators.size() != 0) {
        int ret = 0;
        // Parse FileName
        std::string searchPath = Mki::GetEnv("ASDOPS_HOME_PATH");
        MKI_CHECK(!searchPath.empty(), "ASDOPS_HOME_PATH not exists!", return false);
        searchPath += "/lib/libasdops_aicpu_kernels.so";
        MKI_LOG(INFO) << "serachPath: " << searchPath;

        // Get FileSize
        struct stat buf;
        MKI_CHECK(stat(searchPath.c_str(), &buf) >= 0, "failed to access aicpu kernels binary", return false);
        uint32_t fileSize = (unsigned long)buf.st_size;
        MKI_LOG(DEBUG) << "fileSize: " << fileSize;

        // Load file to HOST
        uint32_t *aicpuKernelSo{nullptr};
        ret = Mki::MkiRtMemMallocHost((void **)&aicpuKernelSo, fileSize);
        MKI_CHECK(ret == 0, "MkiRtMemMallocHost for aicpu kernel so failed", return false);

        FILE *inputFile = nullptr;
        inputFile = fopen(searchPath.c_str(), "rb");
        MKI_CHECK(inputFile != nullptr, "FILE not exists!", return false);
        do{
            fseek(inputFile, 0, SEEK_SET);
            ret = fread(aicpuKernelSo, sizeof(uint8_t), fileSize, inputFile);
        } while(ret != (int)fileSize);
        fclose(inputFile);
        MKI_LOG(DEBUG) << "aicpu kernels binary file stream closed";

        // memcpy .so to DEVICE
        uint32_t *soDevAddr{nullptr};
        ret = Mki::MkiRtMemMallocDevice((void **)&soDevAddr, fileSize, MKIRT_MEM_DEFAULT);
        MKI_CHECK(ret == MKIRT_SUCCESS, "MkiRtMemMallocDevice for soDevAddr fail, errCode:" << ret, return false);
        ret = Mki::MkiRtMemCopy(soDevAddr, fileSize, aicpuKernelSo, fileSize, MKIRT_MEMCOPY_HOST_TO_DEVICE);
        MKI_CHECK(ret == MKIRT_SUCCESS, "MkiRtMemCopy for soDevAddr fail, errCode:" << ret, return false);

        // memcpy .so name to DEVICE
        const char *soName = "libasdops_aicpu_kernels.so";
        uint32_t soNameLen = strlen(soName);
        uint32_t *soNameDevAddr{nullptr};
        ret = Mki::MkiRtMemMallocDevice((void **)&soNameDevAddr, soNameLen, MKIRT_MEM_DEFAULT);
        MKI_CHECK(ret == MKIRT_SUCCESS, "MkiRtMemMallocDevice for soNameDevAddr fail, errCode:" << ret, return false);
        ret = Mki::MkiRtMemCopy(soNameDevAddr, soNameLen, soName, soNameLen, MKIRT_MEMCOPY_HOST_TO_DEVICE);
        MKI_CHECK(ret == MKIRT_SUCCESS, "MkiRtMemCopy for soDevAddr fail, errCode:" << ret, return false);

        // Prep so loading args
        RtLoadOpFromBufArgs opArgs;
        opArgs.kernelSoBuf = (uint64_t)soDevAddr;
        opArgs.kernelSoBufLen = fileSize;
        opArgs.kernelSoName = (uint64_t)soNameDevAddr;
        opArgs.kernelSoNameLen = soNameLen;
        uint32_t argsSize = sizeof(RtLoadOpFromBufArgs);

        // memcpy so loading args to DEVICE
        uint32_t *opArgsHostAddr{nullptr};
        ret = Mki::MkiRtMemMallocHost((void **)&opArgsHostAddr, argsSize);
        MKI_CHECK(ret == MKIRT_SUCCESS, "MkiRtMemMallocDevice for soNameDevAddr fail, errCode:" << ret, return false);
        ret = Mki::MkiRtMemCopy(opArgsHostAddr, argsSize, &opArgs, argsSize, MKIRT_MEMCOPY_HOST_TO_HOST);
        MKI_CHECK(ret == MKIRT_SUCCESS, "MkiRtMemCopy for soDevAddr fail, errCode:" << ret, return false);

        // launch to reg
        MkiRtStream stream = nullptr;
        ret = Mki::MkiRtStreamCreate(&stream, 0);
        MKI_CHECK(ret == MKIRT_SUCCESS && stream != nullptr,
                  "MkiRtStreamCreate for LoadOpFromBuf fail, errCode:" << ret, return false);

        RtArgsExT argsInfo = {};
        argsInfo.args = opArgsHostAddr;
        argsInfo.isNoNeedH2DCopy = 0;
        argsInfo.argsSize = argsSize;

        RtKernelLaunchNamesT launchName = {};
        launchName.soName = nullptr;
        launchName.kernelName = "loadOpFromBuf";
        launchName.opName = "";

        MkiRtAicpuKernelParam kernelParam = {};
        kernelParam.blockDim = 1;
        kernelParam.argsEx = &argsInfo;

        ret = Mki::MkiRtAicpuFunctionLaunchWithFlag(&launchName, &kernelParam, stream);
        MKI_CHECK(ret == MKIRT_SUCCESS, "MkiRtAicpuFunctionLaunchWithFlag for LoadOpFromBuf fail, errCode:" << ret,
                  return false);
        ret = Mki::MkiRtStreamSynchronize(stream);
        MKI_CHECK(ret == MKIRT_SUCCESS, "MkiRtStreamSynchronize for LoadOpFromBuf fail, errCode:" << ret, return false);
        Mki::MkiRtStreamDestroy(stream);
    }

    return true;
}

bool Loader::LoadKernelBinarys()
{
    std::string deviceVersion = PlatformInfo::Instance().GetPlatformName();
    MKI_CHECK(deviceVersion != "unrecognized", "Get device soc version fail: " << deviceVersion, return false);

    for (auto &item : binaryMap_) {
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

bool Loader::OpBaseAddKernels() const
{
    for (const auto &[opName, op] : opMap_) {
        MKI_LOG(DEBUG) << "mki load operation: " << opName;
        OperationBase *opBase = dynamic_cast<OperationBase *>(op);
        MKI_CHECK(opBase != nullptr, opName << ": opBase is nullptr", return false);
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
    return true;
}

void Loader::Load()
{
    loadSuccess_ = false;

    MKI_CHECK(LoadKernelBinarys(), "Load kernel binarys fail", return);
    MKI_CHECK(CreateOperations(), "Load operations fail", return);
    MKI_CHECK(CreateKernels(), "Load kernels fail", return);
    MKI_CHECK(CreateAicpuKernels(), "Load aicpu kernels fail", return);
    MKI_CHECK(OpBaseAddKernels(), "OpBase add kernels fail", return);

    loadSuccess_ = true;
}
} // namespace Mki
