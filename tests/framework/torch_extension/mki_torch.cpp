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
#include "mki_torch.h"
#include <ATen/ATen.h>
#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <nlohmann/json.hpp>
#include "mki/kernel.h"
#include "mki/utils/assert/assert.h"
#include "mki/utils/log/log.h"
#include "mki/utils/status/status.h"
#include "mki/utils/rt/rt.h"
#include "op_desc_json.h"

static std::map<at::ScalarType, Mki::TensorDType> DTYPE_MAP = {
    {at::ScalarType::Byte, Mki::TENSOR_DTYPE_UINT8},   {at::ScalarType::Char, Mki::TENSOR_DTYPE_INT8},
    {at::ScalarType::Half, Mki::TENSOR_DTYPE_FLOAT16}, {at::ScalarType::Float, Mki::TENSOR_DTYPE_FLOAT},
    {at::ScalarType::Int, Mki::TENSOR_DTYPE_INT32},    {at::ScalarType::Long, Mki::TENSOR_DTYPE_INT64},
    {at::ScalarType::BFloat16, Mki::TENSOR_DTYPE_BF16},
};
namespace {
TORCH_LIBRARY(MkiTorch, m)
{
    m.class_<MkiTorch>("MkiTorch")
        .def(torch::init<std::string>())
        .def("execute", &MkiTorch::Execute);
}
}

void *MkiTorch::GetCurrentStream() const
{
    int32_t devId = 0;
    Mki::MkiRtDeviceGetCurrent(&devId);
    void *stream = c10_npu::getCurrentNPUStream(devId).stream();
    return stream;
}

void MkiTorch::ContiguousAtTensor(std::vector<at::Tensor> &atTensors)
{
    for (size_t i = 0; i < atTensors.size(); ++i) {
        if (!atTensors.at(i).is_contiguous()) {
            atTensors.at(i) = atTensors.at(i).contiguous();
        }
    }
}

std::string MkiTorch::AddWorkspace(const Mki::KernelInfo &kernelInfo, Mki::RunInfo &runInfo)
{
    size_t bufferSize = kernelInfo.GetTotalScratchSize();
    if (bufferSize == 0) {
        MKI_LOG(INFO) << "no workspace";
        return "ok";
    }
    MKI_LOG(INFO) << "Workspace size: " << bufferSize;
    uint8_t *deviceBuffer = nullptr;
    int ret = Mki::MkiRtMemMallocDevice(reinterpret_cast<void **>(&deviceBuffer), bufferSize, MKIRT_MEM_DEFAULT);
    if (ret != MKIRT_SUCCESS) {
        MKI_LOG(ERROR) << "MkiRtMemMallocDevice fail, errCode:" << ret << ", errName:" << Mki::MkiRtErrorName(ret)
                       << "errDesc:" << Mki::MkiRtErrorDesc(ret);
        return "error:MkiRtMemMallocDevice fail";
    }
    runInfo.SetScratchDeviceAddr(deviceBuffer);
    return "ok";
}

std::string MkiTorch::FreeWorkspace(const Mki::KernelInfo &kernelInfo, Mki::RunInfo &runInfo)
{
    uint8_t *deviceBuffer = runInfo.GetScratchDeviceAddr();
    size_t bufferSize = kernelInfo.GetTotalScratchSize();
    if (bufferSize == 0) {
        return "ok";
    }
    if (deviceBuffer != nullptr) {
        Mki::MkiRtMemFreeDevice(deviceBuffer);
    }
    return "ok";
}

std::string MkiTorch::RunOp(Mki::LaunchParam &launchParam, const std::vector<Mki::Tensor> &inTensors,
                            std::vector<Mki::Tensor> &outTensors)
{
    Mki::Operation *op = GetOpByName(opName_);
    if (op == nullptr) {
        MKI_LOG(ERROR) << "get operation by name fail, opName:" << opName_;
        return "get operation by name fail";
    }

    for (auto iter : inTensors) {
        launchParam.AddInTensor(iter);
    }
    for (auto iter : outTensors) {
        launchParam.AddOutTensor(iter);
    }

    MKI_LOG(INFO) << "before infershape, launchParam:\n" << launchParam.ToString();
    Mki::Status status = op->InferShape(launchParam);
    if (!status.Ok()) {
        MKI_LOG(ERROR) << opName_ << " infer shape fail, error:" << status.ToString();
        return "infer shape fail";
    }

    MKI_LOG(INFO) << "after infershape, runInfo:\n" << launchParam.ToString();
    Mki::Kernel *kernel = op->GetBestKernel(launchParam);
    if (kernel == nullptr) {
        MKI_LOG(ERROR) << opName_ << " get best kernel fail";
        return "get best kernel fail";
    }

    Mki::RunInfo runInfo;
    MkiRtStream stream = GetCurrentStream();
    MKI_LOG(INFO) << "stream:" << stream;
    runInfo.SetStream(stream);

    uint8_t *deviceLaunchBuffer = nullptr;
    if (launchWithTiling_) {
        kernel->SetLaunchWithTiling(true);
        auto status = kernel->Init(launchParam);
        MKI_CHECK(status.Ok(), "failed to init op", return "failed to init op");
    } else {
        kernel->SetLaunchWithTiling(false);
        uint32_t launchBufferSize = kernel->GetTilingSize(launchParam);
        MKI_CHECK(launchBufferSize > 0, "empty tiling size", return "empty tiling size");

        uint8_t hostLaunchBuffer[launchBufferSize];
        kernel->SetTilingHostAddr(hostLaunchBuffer, launchBufferSize);
        auto status = kernel->Init(launchParam);
        MKI_CHECK(status.Ok(), "failed to init op", return "failed to init op");

        int st = Mki::MkiRtMemMallocDevice(reinterpret_cast<void **>(&deviceLaunchBuffer),
                                              launchBufferSize, MKIRT_MEM_DEFAULT);
        MKI_CHECK(st == MKIRT_SUCCESS, "MkiRtMemMallocDevice error", return "MkiRtMemMallocDevice error");

        st = Mki::MkiRtMemCopy(deviceLaunchBuffer, launchBufferSize,
                                  hostLaunchBuffer, launchBufferSize, MKIRT_MEMCOPY_HOST_TO_DEVICE);
        if (st != MKIRT_SUCCESS) {
            Mki::MkiRtMemFreeDevice(deviceLaunchBuffer);
            deviceLaunchBuffer = nullptr;
            MKI_LOG(ERROR) << "MkiRtMemCopy error";
            return "MkiRtMemCopy error";
        }
        runInfo.SetTilingDeviceAddr(deviceLaunchBuffer);
    }

    const Mki::KernelInfo &kernelInfo = kernel->GetKernelInfo();
    std::string resStr = AddWorkspace(kernelInfo, runInfo);
    if (resStr != "ok") {
        return resStr;
    }

    MKI_LOG(INFO) << kernel->GetName() << " run start, LaunchParam:\n" << launchParam.ToString();
    MKI_LOG(INFO) << "RunInfo:\n" << runInfo.ToString();

    status = kernel->Run(launchParam, runInfo);
    MKI_LOG_IF(!status.Ok(), ERROR) << kernel->GetName() << " run fail, error:" << status.ToString();
    if (!status.Ok()) {
        return "kernel run fail";
    }
    int ret = Mki::MkiRtStreamSynchronize(runInfo.GetStream());
    MKI_LOG_IF(ret != 0, ERROR) << "MkiRtStreamSynchronize fail";
    if (ret != 0) {
        return "MkiRtStreamSynchronize fail";
    }

    if (deviceLaunchBuffer != nullptr) {
        Mki::MkiRtMemFreeDevice(deviceLaunchBuffer);
    }

    std::string retStr = FreeWorkspace(kernelInfo, runInfo);
    delete kernel;
    return retStr;
}

std::string MkiTorch::RunOpPerf(Mki::LaunchParam &launchParam, std::vector<Mki::Tensor> &inTensors,
                                   std::vector<Mki::Tensor> &outTensors, int runTimes)
{
    int st;
    int inTensorIdx = 0;
    std::string retStr;
    char *inTensorTempBufList[20];
    for (auto iter : inTensors) {
        inTensorTempBufList[inTensorIdx] = (char *)malloc(iter.dataSize);
        st = Mki::MkiRtMemCopy((void *)inTensorTempBufList[inTensorIdx], iter.dataSize, iter.data, iter.dataSize,
                                  MKIRT_MEMCOPY_DEVICE_TO_HOST);
        if (st != MKIRT_SUCCESS) {
            MKI_LOG(ERROR) << "MkiRtMemCopy error";
            return "MkiRtMemCopy error";
        }
        inTensorIdx++;
    }
    Mki::Operation *op = GetOpByName(opName_);
    if (op == nullptr) {
        MKI_LOG(ERROR) << "get operation by name fail, opName:" << opName_;
        return "get operation by name fail";
    }

    for (auto iter : inTensors) {
        launchParam.AddInTensor(iter);
    }
    for (auto iter : outTensors) {
        launchParam.AddOutTensor(iter);
    }

    MKI_LOG(INFO) << "before infershape, launchParam:\n" << launchParam.ToString();
    Mki::Status status = op->InferShape(launchParam);
    if (!status.Ok()) {
        MKI_LOG(ERROR) << opName_ << " infer shape fail, error:" << status.ToString();
        return "infer shape fail";
    }

    MKI_LOG(INFO) << "after infershape, runInfo:\n" << launchParam.ToString();
    Mki::Kernel *kernel = op->GetBestKernel(launchParam);
    if (kernel == nullptr) {
        MKI_LOG(ERROR) << opName_ << " get best kernel fail";
        return "get best kernel fail";
    }

    Mki::RunInfo runInfo;
    MkiRtStream stream = GetCurrentStream();
    MKI_LOG(INFO) << "stream:" << stream;
    runInfo.SetStream(stream);


    uint8_t *deviceLaunchBuffer = nullptr;
    if (launchWithTiling_) {
        kernel->SetLaunchWithTiling(true);
        status = kernel->Init(launchParam);
        MKI_CHECK(status.Ok(), "failed to run tiling", return "failed to run tiling");
    } else {
        kernel->SetLaunchWithTiling(false);
        uint32_t launchBufferSize = kernel->GetTilingSize(launchParam);
        MKI_CHECK(launchBufferSize > 0, "empty tiling size", return "empty tiling size");

        uint8_t hostLaunchBuffer[launchBufferSize];
        kernel->SetTilingHostAddr(hostLaunchBuffer, launchBufferSize);
        status = kernel->Init(launchParam);
        MKI_CHECK(status.Ok(), "failed to init op", return "failed to init op");

        int st = Mki::MkiRtMemMallocDevice(reinterpret_cast<void **>(&deviceLaunchBuffer),
                                           launchBufferSize, MKIRT_MEM_DEFAULT);
        MKI_CHECK(st == MKIRT_SUCCESS, "MkiRtMemMallocDevice error", return "MkiRtMemMallocDevice error");

        st = Mki::MkiRtMemCopy(deviceLaunchBuffer, launchBufferSize,
                               hostLaunchBuffer, launchBufferSize, MKIRT_MEMCOPY_HOST_TO_DEVICE);
        if (st != MKIRT_SUCCESS) {
            Mki::MkiRtMemFreeDevice(deviceLaunchBuffer);
            deviceLaunchBuffer = nullptr;
            MKI_LOG(ERROR) << "MkiRtMemCopy error";
            return "MkiRtMemCopy error";
        }
        runInfo.SetTilingDeviceAddr(deviceLaunchBuffer);
    }

    const Mki::KernelInfo &kernelInfo = kernel->GetKernelInfo();
    retStr = AddWorkspace(kernelInfo, runInfo);
    if (retStr != "ok") {
        return retStr;
    }

    MKI_LOG(INFO) << kernel->GetName() << " run start, runInfo:\n" << runInfo.ToString();

    for (int runIdx = 0; runIdx < runTimes; runIdx++) {
        inTensorIdx = 0;
        for (auto iter : inTensors) {
            st = Mki::MkiRtMemCopy(iter.data, iter.dataSize, inTensorTempBufList[inTensorIdx], iter.dataSize,
                                      MKIRT_MEMCOPY_HOST_TO_DEVICE);
            if (st != MKIRT_SUCCESS) {
                MKI_LOG(ERROR) << "MkiRtMemCopy error";
                return "MkiRtMemCopy error";
            }
            inTensorIdx++;
        }
        status = kernel->Run(launchParam, runInfo);
        MKI_LOG_IF(!status.Ok(), ERROR) << kernel->GetName() << " run fail, error:" << status.ToString();
        if (!status.Ok()) {
            if (deviceLaunchBuffer != nullptr) {
                Mki::MkiRtMemFreeDevice(deviceLaunchBuffer);
            }
            retStr = FreeWorkspace(kernelInfo, runInfo);
            return "kernel run fail";
        }
    }
    int ret = Mki::MkiRtStreamSynchronize(runInfo.GetStream());
    MKI_LOG_IF(ret != 0, ERROR) << "MkiRtStreamSynchronize fail";
    if (ret != 0) {
        return "MkiRtStreamSynchronize fail";
    }

    if (deviceLaunchBuffer != nullptr) {
        Mki::MkiRtMemFreeDevice(deviceLaunchBuffer);
    }

    retStr = FreeWorkspace(kernelInfo, runInfo);
    delete kernel;
    return retStr;
}

Mki::Tensor MkiTorch::AtTensor2MkiTensor(const at::Tensor &atTensor)
{
    Mki::Tensor mkiTensor;
    mkiTensor.data = atTensor.data_ptr();
    mkiTensor.dataSize = atTensor.numel() * atTensor.element_size();

    mkiTensor.desc.dims.resize(atTensor.sizes().size());
    for (uint64_t i = 0; i < atTensor.sizes().size(); i++) {
        mkiTensor.desc.dims[i] = atTensor.sizes()[i];
    }

    mkiTensor.desc.format = static_cast<Mki::TensorFormat>(at_npu::native::get_npu_format(atTensor));

    MKI_LOG(INFO) << "At tensor dtype " << atTensor.scalar_type();

    auto it = DTYPE_MAP.find(atTensor.scalar_type());
    if (it != DTYPE_MAP.end()) {
        mkiTensor.desc.dtype = it->second;
    } else {
        MKI_LOG(ERROR) << "not support dtype:" << atTensor.scalar_type();
    }

    return mkiTensor;
}

std::string MkiTorch::ExecuteImpl(std::vector<at::Tensor> &atInTensors, std::vector<at::Tensor> &atOutTensors)
{
    MKI_LOG(INFO) << "Execute start, json:" << opDescJsonStr_ << ", compile time:" << __DATE__ << ":" << __TIME__;
    nlohmann::json opDescJson;
    std::string retStr;
    try {
        opDescJson = nlohmann::json::parse(opDescJsonStr_);
    } catch (nlohmann::json::parse_error &ex) {
        MKI_LOG(ERROR) << "json parse error, CallOp fail";
        return "error:json parse error, CallOp fail";
    }
    Mki::LaunchParam launchParam;
    int runTimes;
    int perfFlag = 0;

    opName_ = opDescJson["opName"];
    if (opDescJson.find("runTimes") != opDescJson.end()) {
        perfFlag = 1;
        runTimes = opDescJson["runTimes"];
    }
    JsonToOpParam(opDescJson, launchParam);

    std::vector<Mki::Tensor> mkiInTensors;
    std::vector<Mki::Tensor> mkiOutTensors;
    if (opDescJson.find("input_formats") !=  opDescJson.end()) {
        for (uint32_t i = 0; i < atInTensors.size(); i++) {
            mkiInTensors.push_back(AtTensor2MkiTensor(atInTensors[i]));
            mkiInTensors[i].desc.format = static_cast<Mki::TensorFormat>(opDescJson["input_formats"][i]);
        }
    } else {
        MKI_LOG(INFO) << "use default format";
        for (at::Tensor &atTensor : atInTensors) {
            mkiInTensors.push_back(AtTensor2MkiTensor(atTensor));
        }
    }
    if (opDescJson.find("output_formats") !=  opDescJson.end()) {
        for (uint32_t i = 0; i < atOutTensors.size(); i++) {
            mkiOutTensors.push_back(AtTensor2MkiTensor(atOutTensors[i]));
            mkiOutTensors[i].desc.format = static_cast<Mki::TensorFormat>(opDescJson["output_formats"][i]);
        }
    } else {
        MKI_LOG(INFO) << "use default format";
        for (at::Tensor &atTensor : atOutTensors) {
            mkiOutTensors.push_back(AtTensor2MkiTensor(atTensor));
        }
    }

    if (perfFlag == 0) {
        retStr = RunOp(launchParam, mkiInTensors, mkiOutTensors);  
    } else if (perfFlag == 1) {
        retStr = RunOpPerf(launchParam, mkiInTensors, mkiOutTensors, runTimes);
    }

    return retStr;
}

std::string MkiTorch::Execute(std::vector<at::Tensor> atInTensors, std::vector<at::Tensor> atOutTensors)
{
    ContiguousAtTensor(atInTensors);
    ContiguousAtTensor(atOutTensors);
    std::string retStr = ExecuteImpl(atInTensors, atOutTensors);
    return retStr;
}
