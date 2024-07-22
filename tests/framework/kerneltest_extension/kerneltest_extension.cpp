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
#include "kerneltest_extension.h"
#include <fstream>
#include <random>
#include <sstream>
#include <nlohmann/json.hpp>
#include "acl/acl.h"
#include "mki/ops.h"
#include "mki/utils/fp16/fp16_t.h"
#include "mki/utils/log/log.h"
#include "op_desc_json.h"

#define CALL_ACL1(x)                                                                                                    \
    if (auto ret = (x) != 0) {                                                                                         \
        MKI_FLOG(ERROR, "Failed to exec acl api %s, result %d", #x, ret);                                              \
        assert(false);                                                                                                 \
    } else {                                                                                                           \
        MKI_FLOG(INFO, "Succeeded to exec acl api %s", #x);                                                            \
    }

void *GetStream(uint32_t deviceId)
{
    CALL_ACL1(aclrtSetDevice(deviceId));
    aclrtStream stream;
    CALL_ACL1(aclrtCreateStream(&stream));
    return stream;
}

void TransToLaunchParam(MkiOpDesc *opDesc, Mki::LaunchParam *launchParam)
{
    for (uint64_t i = 0; i < opDesc->inTensorsLen; i++) {
        const auto &t = opDesc->inTensors[i];
        Mki::Tensor tensor;
        tensor.desc.dtype = t.dtype;
        tensor.desc.format = t.format;
        for (uint64_t j = 0; j < t.dimsLen; j++) {
            tensor.desc.dims.push_back(t.dims[j]);
        }
        launchParam->AddInTensor(tensor);
    }
    for (uint64_t i = 0; i < opDesc->outTensorsLen; i++) {
        const auto &t = opDesc->outTensors[i];
        Mki::Tensor tensor;
        tensor.desc.dtype = t.dtype;
        tensor.desc.format = t.format;
        for (uint64_t j = 0; j < t.dimsLen; j++) {
            tensor.desc.dims.push_back(t.dims[j]);
        }
        launchParam->AddOutTensor(tensor);
    }
    MKI_LOG(INFO) << "Execute start, json:" << opDesc->specificParam;
    nlohmann::json opDescJson;
    std::string retStr;
    try {
        opDescJson = nlohmann::json::parse(opDesc->param);
    } catch (nlohmann::json::parse_error &ex) {
        MKI_LOG(ERROR) << "json parse error, CallOp fail";
    }
    JsonToOpParam(opDescJson, launchParam);
    MKI_LOG(INFO) << "TransToLaunchParam ok, result:\n" << launchParam->ToString();
}

MkiTensorData GetTilingData(MkiOpDesc *opDesc, uint32_t *blockDim)
{
    MkiTensorData td;

    Mki::LaunchParam lp;
    TransToLaunchParam(opDesc, &lp);
    Mki::Kernel *op = Mki::Ops::Instance().GetKernelInstance(opName_);
    td.dataLen = op->GetTilingSize(lp);

    CALL_ACL1(aclrtMallocHost(reinterpret_cast<void **>(&(td.hostData)), td.dataLen));

    op->SetLaunchWithTiling(false);
    op->SetTilingHostAddr(td.hostData, td.dataLen);
    auto status = op->Init(launchParam);
    MKI_CHECK(status.Ok(), "failed to init op", return td);
    // *blockDim = op->GetKernelInfo().GetBlockDim();
    *blockDim = op->GetBlockDim();
    MKI_LOG(INFO) << "get tiling data, block dim " << *blockDim;

    CALL_ACL1(aclrtMalloc(reinterpret_cast<void **>(&(td.deviceData)), td.dataLen, ACL_MEM_MALLOC_HUGE_FIRST));
    CALL_ACL1(aclrtMemcpy(td.deviceData, td.dataLen, td.hostData, td.dataLen, ACL_MEMCPY_HOST_TO_DEVICE));

    return td;
}

uint64_t GetTensorNumel(MkiTensor *desc)
{
    uint64_t num = 1;
    for (size_t i = 0; i < desc->dimsLen; i++) {
        num *= desc->dims[i];
    }
    return num;
}

template <typename T>
void PrintSampleData(T *data, uint64_t len)
{
    std::stringstream ss;
    for (uint64_t i = 0; i < len && i < 10; i++) {
        int value = data[i];
        if (std::is_same<T, int8_t>::value) {
            ss << value << ' ';
        } else {
            ss << data[i] << ' ';
        }
    }
    MKI_LOG(INFO) << "data: " << ss.str();
}

void PrintTensorData(MkiTensor *desc, MkiTensorData *td, uint64_t printMaxCount)
{
    if (td->deviceData) {
        CALL_ACL1(aclrtMemcpy(td->hostData, td->dataLen, td->deviceData, td->dataLen, ACL_MEMCPY_DEVICE_TO_HOST));
    }
    std::ostringstream ss;
    uint64_t numel = GetTensorNumel(desc);
    for (uint64_t i = 0; i < numel; ++i) {
        if (i > 0 && i % 16 == 0) {
            ss << std::endl;
        }
        if (printMaxCount > 0 && i == printMaxCount) {
            ss << "...";
            break;
        }

        if (desc->dtype == Mki::TENSOR_DTYPE_FLOAT16) {
            Mki::fp16_t *tensorData = reinterpret_cast<Mki::fp16_t *>(td->hostData);
            ss << tensorData[i] << ",";
        } else if (desc->dtype == Mki::TENSOR_DTYPE_FLOAT) {
            float *tensorData = reinterpret_cast<float *>(td->hostData);
            ss << tensorData[i] << ",";
        } else if (desc->dtype == Mki::TENSOR_DTYPE_INT32) {
            int32_t *tensorData = reinterpret_cast<int32_t *>(td->hostData);
            ss << tensorData[i] << ",";
        } else if (desc->dtype == Mki::TENSOR_DTYPE_INT64) {
            int64_t *tensorData = reinterpret_cast<int64_t *>(td->hostData);
            ss << tensorData[i] << ",";
        } else if (desc->dtype == Mki::TENSOR_DTYPE_INT8) {
            int8_t *tensorData = reinterpret_cast<int8_t *>(td->hostData);
            ss << static_cast<int>(tensorData[i]) << ",";
        } else if (desc->dtype == Mki::TENSOR_DTYPE_UINT32) {
            uint32_t *tensorData = reinterpret_cast<uint32_t *>(td->hostData);
            ss << tensorData[i] << ",";
        } else {
            ss << "N,";
        }
    }
    MKI_LOG(INFO) << "data: " << ss.str();
}

MkiTensorData RandF16Tensor(MkiTensor *desc, float min, float max)
{
    MkiTensorData td;
    uint64_t numel = GetTensorNumel(desc);
    td.dataLen = numel * sizeof(Mki::fp16_t);
    CALL_ACL1(aclrtMallocHost(reinterpret_cast<void **>(&(td.hostData)), td.dataLen));
    Mki::fp16_t *f16data = reinterpret_cast<Mki::fp16_t *>(td.hostData);

    std::random_device rd;
    std::default_random_engine eng(rd());
    std::uniform_real_distribution<float> distr(min, max);
    for (uint64_t i = 0; i < numel; i++) {
        f16data[i] = static_cast<Mki::fp16_t>(distr(eng));
    }
    MKI_LOG(INFO) << "rand f16 tensor, len: " << numel;
    PrintSampleData(f16data, numel);

    CALL_ACL1(aclrtMalloc(reinterpret_cast<void **>(&(td.deviceData)), td.dataLen, ACL_MEM_MALLOC_HUGE_FIRST));
    CALL_ACL1(aclrtMemcpy(td.deviceData, td.dataLen, td.hostData, td.dataLen, ACL_MEMCPY_HOST_TO_DEVICE));

    return td;
}

MkiTensorData RandF32Tensor(MkiTensor *desc, float min, float max)
{
    MkiTensorData td;
    uint64_t numel = GetTensorNumel(desc);
    td.dataLen = numel * sizeof(float);
    CALL_ACL1(aclrtMallocHost(reinterpret_cast<void **>(&(td.hostData)), td.dataLen));
    float *f32data = reinterpret_cast<float *>(td.hostData);

    std::random_device rd;
    std::default_random_engine eng(rd());
    std::uniform_real_distribution<float> distr(min, max);
    for (uint64_t i = 0; i < numel; i++) {
        f32data[i] = static_cast<float>(distr(eng));
    }
    MKI_LOG(INFO) << "rand f32 tensor, len: " << numel;
    PrintSampleData(f32data, numel);

    CALL_ACL1(aclrtMalloc(reinterpret_cast<void **>(&(td.deviceData)), td.dataLen, ACL_MEM_MALLOC_HUGE_FIRST));
    CALL_ACL1(aclrtMemcpy(td.deviceData, td.dataLen, td.hostData, td.dataLen, ACL_MEMCPY_HOST_TO_DEVICE));

    return td;
}

MkiTensorData RandI8Tensor(MkiTensor *desc, int8_t min, int8_t max)
{
    MkiTensorData td;
    uint64_t numel = GetTensorNumel(desc);
    td.dataLen = numel * sizeof(int8_t);
    CALL_ACL1(aclrtMallocHost(reinterpret_cast<void **>(&(td.hostData)), td.dataLen));
    int8_t *i8data = reinterpret_cast<int8_t *>(td.hostData);

    std::random_device rd;
    std::default_random_engine eng(rd());
    std::uniform_int_distribution<int8_t> distr(min, max);
    for (uint64_t i = 0; i < numel; i++) {
        i8data[i] = static_cast<int8_t>(distr(eng));
    }
    MKI_LOG(INFO) << "rand i8 tensor, len: " << numel;
    PrintSampleData(i8data, numel);

    CALL_ACL1(aclrtMalloc(reinterpret_cast<void **>(&(td.deviceData)), td.dataLen, ACL_MEM_MALLOC_HUGE_FIRST));
    CALL_ACL1(aclrtMemcpy(td.deviceData, td.dataLen, td.hostData, td.dataLen, ACL_MEMCPY_HOST_TO_DEVICE));

    return td;
}

MkiTensorData RandI32Tensor(MkiTensor *desc, int32_t min, int32_t max)
{
    MkiTensorData td;
    uint64_t numel = GetTensorNumel(desc);
    td.dataLen = numel * sizeof(int32_t);
    CALL_ACL1(aclrtMallocHost(reinterpret_cast<void **>(&(td.hostData)), td.dataLen));
    int32_t *i32data = reinterpret_cast<int32_t *>(td.hostData);

    std::random_device rd;
    std::default_random_engine eng(rd());
    std::uniform_int_distribution<int32_t> distr(min, max);
    for (uint64_t i = 0; i < numel; i++) {
        i32data[i] = static_cast<int32_t>(distr(eng));
    }
    MKI_LOG(INFO) << "rand i32 tensor, len: " << numel;
    PrintSampleData(i32data, numel);

    CALL_ACL1(aclrtMalloc(reinterpret_cast<void **>(&(td.deviceData)), td.dataLen, ACL_MEM_MALLOC_HUGE_FIRST));
    CALL_ACL1(aclrtMemcpy(td.deviceData, td.dataLen, td.hostData, td.dataLen, ACL_MEMCPY_HOST_TO_DEVICE));

    return td;
}

MkiTensorData ReadFile(MkiTensor *desc, const char *file)
{
    MkiTensorData td;
    uint64_t numel = GetTensorNumel(desc);
    uint64_t dataLen = numel * GetTensorElementSize(desc->dtype);

    std::ifstream input(std::string(file), std::ios::binary);
    if (!input.is_open()) {
        MKI_LOG(ERROR) << "Failed to open file " << file;
        return td;
    }

    std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(input), {});
    if (buffer.size() != dataLen) {
        MKI_FLOG_ERROR("File size %zu is invalid, expect %zu", buffer.size(), dataLen);
        return td;
    }
    MKI_FLOG_INFO("File %s read, total size %zu", file, buffer.size());
    td.dataLen = dataLen;
    CALL_ACL1(aclrtMallocHost(reinterpret_cast<void **>(&(td.hostData)), td.dataLen));
    auto ret = memcpy(td.hostData, buffer.data(), td.dataLen);
    if (ret != 0) {
        MKI_LOG(ERROR) << "ReadFile fail";
        return td;
    }
    PrintTensorData(desc, &td);
    CALL_ACL1(aclrtMalloc(reinterpret_cast<void **>(&(td.deviceData)), td.dataLen, ACL_MEM_MALLOC_HUGE_FIRST));
    CALL_ACL1(aclrtMemcpy(td.deviceData, td.dataLen, td.hostData, td.dataLen, ACL_MEMCPY_HOST_TO_DEVICE));
    return td;
}

void FreeTensor(MkiTensorData &td)
{
    CALL_ACL1(aclrtFree(td.deviceData));
    CALL_ACL1(aclrtFreeHost(td.hostData));
}
