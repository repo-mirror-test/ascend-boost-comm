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
#ifndef MKI_OPTEST_H
#define MKI_OPTEST_H

#include <array>
#include <float.h>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "mki/launch_param.h"
#include "mki/base/kernel_base.h"
#include "mki/run_info.h"
#include "mki/tensor_desc.h"
#include "mki/utils/any/any.h"
#include "mki/utils/svector/svector.h"

namespace Mki {
const int32_t MAX_INPUT_TENSOR = 16;

struct UtOpDesc {
    std::string opName;
    Any specificParam;
};

struct GoldenContext {
    UtOpDesc opDesc;
    SVector<Tensor> hostInTensors;
    SVector<Tensor> hostOutTensors;
    SVector<Tensor> deviceInTensors;
    SVector<Tensor> deviceOutTensors;
};

using OpTestGolden = std::function<Status(const GoldenContext &context)>;

struct OpTestStatistic {
    uint64_t total = 0;
    uint64_t init = 0;
    uint64_t prepare = 0;
    uint64_t kernelRun = 0;
    uint64_t streamSync = 0;
    uint64_t runGolden = 0;
    std::string ToString() const;
};

class OpTest {
public:
    explicit OpTest();
    ~OpTest();
    void Golden(OpTestGolden golden);
    void FloatRand(float min, float max);
    void Int8Rand(int8_t min, int8_t max);
    void IntRand(int32_t min, int32_t max);
    void LongRand(int64_t min, int64_t max);
    Status Run(const UtOpDesc &opDesc, const TensorDesc &inTensorDesc, const TensorDesc &outTensorDesc,
               const std::string &opName = "");
    Status Run(const UtOpDesc &opDesc, const SVector<TensorDesc> &inTensorDescs,
               const SVector<TensorDesc> &outTensorDescs, const std::string &opName = "");
    Status Run(const UtOpDesc &opDesc, const SVector<Tensor> &inTensorLists, const SVector<Tensor> &outTensorLists,
               const std::string &opName = "");
    Status RunWithDataFile(const UtOpDesc &opDesc, const SVector<TensorDesc> &inTensorDescs,
                           const SVector<TensorDesc> &outTensorDescs, const SVector<std::string> &files,
                           const std::string &opName = "");
    OpTestStatistic GetRunStatistic() const;

    void ReadFile(void *data, size_t dataSize, const std::string &dataFile);

    void SetOutdataUseInputData(size_t outIndex, size_t inIndex);
    void SetOutputNum(int64_t outputNum);

private:
    Status RunImpl(const UtOpDesc &opDesc, const SVector<TensorDesc> &inTensorDescs,
                   const SVector<TensorDesc> &outTensorDescs, const std::string &opName);
    Status RunImpl(const UtOpDesc &opDesc, const SVector<Tensor> &inTensorLists, const SVector<Tensor> &outTensorLists,
                   const std::string &opName);
    Tensor CreateHostRandTensor(const TensorDesc &tensorDesc);
    Tensor CreateHostZeroTensor(const TensorDesc &tensorDesc);
    Tensor CreateHostTensorFromFile(const TensorDesc &tensorDesc, const std::string &dataFile);
    Tensor CreateHostTensor(const Tensor &tensorIn);
    Tensor HostTensor2DeviceTensor(const Tensor &hostTensor);
    Status Prepare(const UtOpDesc &opDesc, const SVector<TensorDesc> &inTensorDescs, const SVector<TensorDesc> &outTensorDescs);
    Status Prepare(const UtOpDesc &opDesc, const SVector<Tensor> &inTensorLists, const SVector<Tensor> &outTensorLists);
    Status RunKernel(const std::string &opName);
    Status CopyDeviceTensorToHostTensor();
    Status RunGolden();
    void Init();
    void AddWorkspace();
    void FreeWorkspace();
    int64_t GetOutputNum();
    void Cleanup();
    void MallocInTensor();
    void MallocInTensor(const SVector<Tensor> &inTensorLists);
    void MallocOutTensor();
    void MallocOutTensor(const SVector<Tensor> &outTensorLists);
    std::string TensorToString(const Tensor &tensor);

private:
    bool launchWithTiling_ = true;
    int deviceId_ = 0;
    OpTestGolden golden_;
    GoldenContext goldenContext_;
    SVector<std::string> dataFiles_;
    std::map<size_t, size_t> outUseInputdata_; // <0, 2> = outtensor(0).data = intensor(2).data
    LaunchParam launchParam_;
    RunInfo runInfo_;
    Kernel *op_ = nullptr;
    int64_t outputNum_ = 0;
    float randFloatMin_ = FLT_MIN;
    float randFloatMax_ = FLT_MAX;
    int8_t randInt8Min_ = 3;
    int8_t randInt8Max_ = 3;
    int32_t randIntMin_ = 2;
    int32_t randIntMax_ = 2;
    int64_t randLongMin_ = 1;
    int64_t randLongMax_ = 1;
    OpTestStatistic statistic_;
};
} // namespace Mki

#endif
