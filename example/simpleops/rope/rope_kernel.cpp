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
#include <schedule/op_register.h>
#include <mki/utils/assert/assert.h>
#include <mki/utils/log/log.h>
#include <mki/utils/platform/platform_info.h>
#include <mki/utils/const/op_const.h>
#include "params/params.h"
#include "tiling_data.h"

static constexpr uint32_t TENSOR_INPUT_NUM = 5;
static constexpr uint32_t TENSOR_OUTPUT_NUM = 2;

using namespace Mki;

namespace SimpleOps {
static constexpr uint32_t BLOCK_SIZE = 16;
static constexpr uint32_t REMAIN_SPACE = 16 * 1024 * 1024;
static constexpr uint32_t NUM_COSIN = 2;
static constexpr uint32_t TOTAL_OFFSET = 6;

static constexpr uint32_t TILING_BF16_BROARD = 32;
static constexpr uint32_t TILING_HIGH_PREC_BOARD = 31;
static constexpr uint32_t TILING_HIGH_PERF_BROARD = 30;
static constexpr uint32_t TILING_BF16 = 22;
static constexpr uint32_t TILING_HIGH_PREC = 21;
static constexpr uint32_t TILING_HIGH_PERF = 20;

void RopeNdProcess(const LaunchParam &launchParam, KernelInfo &kernelInfo, RopeTilingData *tilingDataPtr)
{
    uint32_t hiddenSizeQ  = static_cast<uint32_t>(launchParam.GetInTensor(0).desc.dims[1]);
    uint32_t hiddenSizeK = static_cast<uint32_t>(launchParam.GetInTensor(1).desc.dims[1]);
    auto cosSize = launchParam.GetInTensor(2).desc.dims.size();
    uint32_t headDim = static_cast<uint32_t>(launchParam.GetInTensor(2).desc.dims[cosSize - 1]);
    uint32_t ntokens = static_cast<uint32_t>(launchParam.GetInTensor(0).desc.dims[0]);
    uint32_t batch = static_cast<uint32_t>(launchParam.GetInTensor(4).desc.dims[0]);
    uint32_t maxCore = static_cast<uint32_t>(PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR));
    auto maxUbSize = static_cast<uint32_t>(PlatformInfo::Instance().GetUbSize());
    tilingDataPtr->maxUbSize = maxUbSize;
    uint32_t tempCore = (ntokens + maxCore - 1) / maxCore;
    uint32_t realCore = (ntokens + tempCore - 1) / tempCore;
    tilingDataPtr->realCore = realCore;
    tilingDataPtr->hiddenSizeQ = hiddenSizeQ;
    tilingDataPtr->hiddenSizeK = hiddenSizeK;
    tilingDataPtr->headDim = headDim;
    tilingDataPtr->ntokens = ntokens;
    tilingDataPtr->batch = batch;
    kernelInfo.SetBlockDim(realCore);
}
Status TilingKeyChose(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    auto platformType = PlatformInfo::Instance().GetPlatformType();
    auto cosSize = launchParam.GetInTensor(NUM_COSIN).desc.dims.size();
    if (cosSize == NUM_COSIN) {
        if (launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_BF16) {
            if (platformType == PlatformType::ASCEND_910B) {
                kernelInfo.SetTilingId(TILING_BF16); // first 2 for shape dims of cos
                // second 2 for BF16
            } else {
                MKI_LOG(ERROR) << "BF16 only supports 910B";
                return Status::FailStatus(ERROR_INVALID_VALUE);
            }
        } else if (launchParam.GetInTensor(NUM_COSIN).desc.dtype == TENSOR_DTYPE_FLOAT) {
            kernelInfo.SetTilingId(TILING_HIGH_PREC); // second 1 for FP32
        } else {
            kernelInfo.SetTilingId(TILING_HIGH_PERF); // second 0 for FP16
        }
    } else {
        if (launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_BF16) {
            if (platformType == PlatformType::ASCEND_910B) {
                kernelInfo.SetTilingId(TILING_BF16_BROARD); // first 3 for shape dims of cos
                // second 2 for BF16
            } else {
                MKI_LOG(ERROR) << "BF16 only supports 910B";
                return Status::FailStatus(ERROR_INVALID_VALUE);
            }
        } else if (launchParam.GetInTensor(NUM_COSIN).desc.dtype == TENSOR_DTYPE_FLOAT) {
            kernelInfo.SetTilingId(TILING_HIGH_PREC_BOARD); // second 1 for FP32
        } else {
            kernelInfo.SetTilingId(TILING_HIGH_PERF_BROARD); // second 0 for fp16
        }
    }
    return Status::OkStatus();
}
Status RopeTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    RopeTilingData *tilingDataPtr = reinterpret_cast<RopeTilingData *>(kernelInfo.GetTilingHostAddr());
    MKI_CHECK(tilingDataPtr != nullptr, "tilingDataPtr should not be empty",
                 return Status::FailStatus(ERROR_INVALID_VALUE));
    auto attrs = AnyCast<OpParam::Rope>(launchParam.GetParam());
    uint32_t headNumQ = 1;
    uint32_t headNumK = 1;
    auto ret = TilingKeyChose(launchParam, kernelInfo);
    if (!ret.Ok()) {
        return Status::FailStatus(ERROR_INVALID_VALUE);
    }
    RopeNdProcess(launchParam, kernelInfo, tilingDataPtr);
    if (tilingDataPtr->headDim != 0) {
        headNumQ = tilingDataPtr->hiddenSizeQ / tilingDataPtr->headDim;
        headNumK = tilingDataPtr->hiddenSizeK / tilingDataPtr->headDim;
    } else {
        return Status::FailStatus(ERROR_INVALID_VALUE, "tilingDataPtr->headDim is wrong");
    }
    tilingDataPtr->headNumQ = headNumQ;
    tilingDataPtr->headNumK = headNumK;
    MKI_CHECK(attrs.rotaryCoeff > 0, "attrs.rotaryCoeff is invalid", return Status::FailStatus(ERROR_INVALID_VALUE));
    tilingDataPtr->rotaryCoeff = static_cast<uint32_t>(attrs.rotaryCoeff);
    MKI_CHECK(attrs.cosFormat == 0 || attrs.cosFormat == 1, "wrong cosFormat, cosFormat should be 0 or 1",
                 return Status::FailStatus(ERROR_INVALID_VALUE));
    tilingDataPtr->cosFormat = static_cast<uint32_t>(attrs.cosFormat);
    uint64_t sysWorkspaceSize = static_cast<uint64_t>(
        REMAIN_SPACE + TOTAL_OFFSET * tilingDataPtr->realCore * tilingDataPtr->hiddenSizeQ * sizeof(uint16_t) +
        tilingDataPtr->ntokens * tilingDataPtr->headDim * NUM_COSIN * sizeof(uint16_t));
    MKI_LOG(INFO) << "workspace: " << sysWorkspaceSize;
    kernelInfo.GetScratchSizes() = {sysWorkspaceSize};
    return Status::OkStatus();
}

class RopeKernel : public KernelBase {
public:
    explicit RopeKernel(const std::string &kernelName, KernelHandle handle) noexcept : KernelBase(kernelName, handle) {}

    bool RopeDtypeCheck(const LaunchParam &launchParam, TensorDType dtypeCheck) const
    {
        auto inTensor0 = launchParam.GetInTensor(0);
        auto inTensor1 = launchParam.GetInTensor(1);
        MKI_CHECK(inTensor0.desc.format == TENSOR_FORMAT_ND, "in tensor0 format is invalid", return false);
        MKI_CHECK(inTensor0.desc.dtype == inTensor1.desc.dtype,
            "in tensor0 dtype != tensor1 dtype", return false);
        MKI_CHECK(inTensor1.desc.format == TENSOR_FORMAT_ND, "in tensor1 format is invalid", return false);
        for (size_t i = 0; i < TENSOR_OUTPUT_NUM; i++) {
            auto outTensor = launchParam.GetOutTensor(i);
            MKI_CHECK(outTensor.desc.format == inTensor0.desc.format, "out tensor format is invalid", return false);
            MKI_CHECK(outTensor.desc.dtype == inTensor0.desc.dtype, "out tensor dtype is invalid", return false);
        }
        if (dtypeCheck == TENSOR_DTYPE_BF16) {
            MKI_CHECK(launchParam.GetInTensor(DIM_2).desc.dtype == TENSOR_DTYPE_BF16,
                "in tensor2 dtype is invalid", return false);
            MKI_CHECK(launchParam.GetInTensor(DIM_3).desc.dtype == TENSOR_DTYPE_BF16,
                "in tensor3 dtype is invalid", return false);
        } else {
            if (launchParam.GetInTensor(DIM_2).desc.dtype == dtypeCheck) {
                MKI_CHECK(launchParam.GetInTensor(DIM_3).desc.dtype == dtypeCheck,
                    "in tensor3 dtype is invalid", return false);
            } else if (launchParam.GetInTensor(DIM_2).desc.dtype == TENSOR_DTYPE_FLOAT) {
                MKI_CHECK(launchParam.GetInTensor(DIM_3).desc.dtype == TENSOR_DTYPE_FLOAT,
                    "in tensor3 dtype is invalid", return false);
            } else {
                return false;
            }
        }
        MKI_CHECK(launchParam.GetInTensor(0).desc.dims.size() == 2,
            "in tensor0 dims is invalid", return false);
        MKI_CHECK(launchParam.GetInTensor(1).desc.dims.size() == 2,
            "in tensor1 dims is invalid", return false);
        auto inTensor4 = launchParam.GetInTensor(4);
        MKI_CHECK(inTensor4.desc.format == TENSOR_FORMAT_ND, "in tensor4 format is invalid", return false);
        MKI_CHECK((inTensor4.desc.dtype == TENSOR_DTYPE_INT32) ||
                        (inTensor4.desc.dtype == TENSOR_DTYPE_UINT32), "in tensor4 dtype is invalid", return false);
        return true;
    }
    
    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetInTensorCount() == TENSOR_INPUT_NUM, "in tensor num is invalid", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == TENSOR_OUTPUT_NUM, "out tensor num is invalid", return false);
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Rope),
            "param type is invalid", return false);
        return RopeDtypeCheck(launchParam, TENSOR_DTYPE_FLOAT16) || RopeDtypeCheck(launchParam, TENSOR_DTYPE_BF16);
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        (void)launchParam;
        return sizeof(RopeTilingData);
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        KernelInfo &kernelInfo = GetKernelInfo();
        return RopeTiling(launchParam, kernelInfo);
    }
};

REG_KERNEL_BASE(RopeKernel);
} // namespace SimpleOps
