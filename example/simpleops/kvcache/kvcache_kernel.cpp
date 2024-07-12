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
#include <mki/utils/assert/assert.h>
#include <mki/utils/platform/platform_info.h>
#include <mki_loader/op_register.h>
#include "mki/base/kernel_base.h"
#include "params/params.h"
#include "mki/utils/log/log.h"
#include "mki/utils/math/tensor_utils.h"
#include "mki/utils/math/math.h"
#include "tiling_data.h"

namespace SimpleOps {
constexpr uint64_t TENSOR_BATCH_STATUS_IDX = 5;
constexpr uint32_t MAX_PROCESS_NUM = 98304;

static constexpr uint64_t KVCACHE_TILING_ID_BASE = 2000000000;
static constexpr uint64_t KVCACHE_TILING_ID_TYPE = 100000000; // 0:fp16/bf16,1:int8

Status KVCacheTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    auto tilingDataPtr = reinterpret_cast<KVCacheTilingData *>(kernelInfo.GetTilingHostAddr());
    const auto &inTensor3Dims = launchParam.GetInTensor(DIM_3).desc.dims;
    auto batch = static_cast<uint32_t>(inTensor3Dims[0]);

    const auto &inTensor0Dims = launchParam.GetInTensor(0).desc.dims;
    auto hiddenSize = static_cast<uint32_t>(inTensor0Dims[1]);

    uint32_t maxUbBytes = static_cast<uint32_t>(PlatformInfo::Instance().GetUbSize());
    TensorDType dtype = launchParam.GetInTensor(0).desc.dtype;
    uint32_t maxProcessNum = maxUbBytes / GetTensorElementSize(dtype);
    MKI_CHECK(hiddenSize > 0 && hiddenSize <= maxProcessNum, "invalid hiddenSize: " << hiddenSize,
        return Status::FailStatus(ERROR_INVALID_VALUE, "invalid hiddenSize:" + std::to_string(hiddenSize)));

    const auto &inTensor2Dims = launchParam.GetInTensor(DIM_2).desc.dims;
    uint32_t maxSeqLen = 0;
    if (inTensor2Dims.size() == DIM_4) {
        maxSeqLen = static_cast<uint32_t>(inTensor2Dims[DIM_2]);
    } else if (inTensor2Dims.size() == DIM_3) {          // dyncBatch dims
        maxSeqLen = static_cast<uint32_t>(inTensor2Dims[DIM_1]);
    } else {
        return Status::FailStatus(ERROR_INVALID_VALUE, "invalid tilingData maxSeqLen: " + std::to_string(maxSeqLen) +
            ", maxSeqLen > " + std::to_string(0));
    }

    tilingDataPtr->batch = batch;
    tilingDataPtr->maxSeqLen = maxSeqLen;
    tilingDataPtr->hiddenSize = hiddenSize;
    MKI_LOG(INFO) << "KVCache Tiling Data: batch " << tilingDataPtr->batch << ", maxSeqLen "
        << tilingDataPtr->maxSeqLen << ", hiddenSize: " << tilingDataPtr->hiddenSize;

    kernelInfo.SetBlockDim(batch);
    return Status::OkStatus();
}

Status KVCacheNdTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    Status status = KVCacheTiling(launchParam, kernelInfo);
    if (!status.Ok()) {
        return status;
    }

    uint64_t tilingKey = KVCACHE_TILING_ID_BASE;
    TensorDType dtype = launchParam.GetInTensor(0).desc.dtype;
    tilingKey += (dtype == TENSOR_DTYPE_FLOAT16 || dtype == TENSOR_DTYPE_BF16) ? 0 : KVCACHE_TILING_ID_TYPE;

    kernelInfo.SetTilingId(tilingKey);

    MKI_LOG(INFO) << "KVCacheNd TilingKey: " << tilingKey;

    return Status::OkStatus();
}

Status KVCacheNzTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    auto tilingDataPtr = reinterpret_cast<KVCacheTilingData *>(kernelInfo.GetTilingHostAddr());
    const auto &inTensor3Dims = launchParam.GetInTensor(3).desc.dims;
    auto batch = static_cast<uint32_t>(inTensor3Dims[0]);

    const auto &inTensor0Dims = launchParam.GetInTensor(0).desc.dims;
    int64_t hiddenSize = inTensor0Dims[1] * 16; // 16是因为数据是Nz排布
    MKI_CHECK(hiddenSize <= MAX_PROCESS_NUM, "invalid hiddenSize: " << hiddenSize,
        return Status::FailStatus(ERROR_INVALID_VALUE, "invalid hiddenSize:" + std::to_string(hiddenSize)));

    const auto &inTensor2Dims = launchParam.GetInTensor(2).desc.dims;
    auto maxSeqLen = static_cast<uint32_t>(inTensor2Dims[3]);
    uint32_t coreNum = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR);
    tilingDataPtr->batchPerCore = Utils::CeilDiv(batch, coreNum);
    uint32_t numCore = Utils::CeilDiv(batch, tilingDataPtr->batchPerCore);
    tilingDataPtr->batch = batch;
    tilingDataPtr->maxSeqLen = maxSeqLen;
    tilingDataPtr->hiddenSize = static_cast<uint32_t>(hiddenSize);

    MKI_LOG(INFO) << "KVCacheNz Tiling Data: batch " << tilingDataPtr->batch << ", maxSeqLen "
                  << tilingDataPtr->maxSeqLen << ", hiddenSize: " << tilingDataPtr->hiddenSize << ", batchPerCore: "
                  << tilingDataPtr->batchPerCore;

    kernelInfo.SetBlockDim(numCore);
    return Status::OkStatus();
}

class KVCacheKernel : public KernelBase {
public:
    explicit KVCacheKernel(const std::string &kernelName, const BinHandle *handle) noexcept : KernelBase(kernelName, handle)
    {
        launchBufferSize_ = sizeof(KVCacheTilingData);
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::KVCache),
            "kv_cache: param type invalid", return false);
        MKI_CHECK(launchParam.GetInTensorCount() == 5, "in tensor num invalid", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == 1, "out tensor num invalid", return false);
        return true;
    }
};

class KVCacheNdKernel : public KVCacheKernel {
public:
    explicit KVCacheNdKernel(const std::string &kernelName, const BinHandle *handle) noexcept : KVCacheKernel(kernelName, handle) {}

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(KVCacheKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.format == TENSOR_FORMAT_ND,
            "in tensor 0 format is invalid", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dims.size() == 2,
            "in tensor 0 dims num is invalid", return false);
        MKI_CHECK(launchParam.GetInTensor(3).desc.dims.size() == 1,
            "in tensor 3 dims num is invalid", return false);
        return true;
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        KernelInfo &kernelInfo = GetKernelInfo();
        return KVCacheNdTiling(launchParam, kernelInfo);
    }
};
REG_KERNEL_BASE(KVCacheNdKernel);

class KVCacheNzKernel : public KVCacheKernel {
public:
    explicit KVCacheNzKernel(const std::string &kernelName, const BinHandle *handle) noexcept : KVCacheKernel(kernelName, handle) {}

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(KVCacheKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.format == TENSOR_FORMAT_FRACTAL_NZ,
            "in tensor 0 format is invalid", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dims.size() == 4,
            "in tensor 0 dims num is invalid", return false);
        MKI_CHECK(launchParam.GetInTensor(2).desc.dims.size() == 5,
            "in tensor 2 dims num is invalid", return false);
        MKI_CHECK(launchParam.GetInTensor(3).desc.dims.size() == 1,
            "in tensor 3 dims num is invalid", return false);
        return true;
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        KernelInfo &kernelInfo = GetKernelInfo();
        return KVCacheNzTiling(launchParam, kernelInfo);
    }
};
REG_KERNEL_BASE(KVCacheNzKernel);

class KVCacheNdDynamicBatchKernel : public KVCacheKernel {
public:
    explicit KVCacheNdDynamicBatchKernel(const std::string &tacticName, const BinHandle *handle) noexcept : KVCacheKernel(tacticName, handle) {}

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(KVCacheKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.format == TENSOR_FORMAT_ND,
            "in tensor 0 format is invalid", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dims.size() == 2,
            "in tensor 0 dims num is invalid", return false);
        MKI_CHECK(launchParam.GetInTensor(3).desc.dims.size() == 1,
            "in tensor 3 dims num is invalid", return false);
        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::KVCache),
            "OpParam is invalid", return 0);
        auto param = AnyCast<OpParam::KVCache>(launchParam.GetParam());
        size_t constTensorSize = Utils::GetConstTensorSize<int32_t>(param.batchRunStatus);
        size_t maxVal = std::numeric_limits<uint32_t>::max();
        MKI_CHECK(maxVal - constTensorSize >= launchBufferSize_, "batchRunStatus Size is invalid", return 0);
        return launchBufferSize_ + constTensorSize;
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        KernelInfo &kernelInfo = GetKernelInfo();
        auto status = KVCacheTiling(launchParam, kernelInfo);
        MKI_CHECK_NO_LOG(status.Ok(), return status);

        kernelInfo.SetConstTensorOffset(launchBufferSize_);

        auto &param = AnyCast<OpParam::KVCache>(launchParam.GetParam());
        auto ret = kernelInfo.AddConstTensorData<int32_t>(TENSOR_BATCH_STATUS_IDX, param.batchRunStatus);
        MKI_CHECK_NO_LOG(ret, return Status::FailStatus(1));

        return Status::OkStatus();
    }
};
REG_KERNEL_BASE(KVCacheNdDynamicBatchKernel);
} // namespace SimpleOps