/*
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
#ifndef MKI_TORCH_H
#define MKI_TORCH_H
#include <string>
#include <vector>
#include <torch/script.h>
#include <torch/custom_class.h>
#include "schedule/ops.h"
#include "mki/kernel.h"

class MkiTorch : public torch::CustomClassHolder {
public:
    explicit MkiTorch(std::string opDescJsonStr) : opDescJsonStr_(opDescJsonStr) {}
    ~MkiTorch() {}
    std::string Execute(std::vector<at::Tensor> atInTensors, std::vector<at::Tensor> atOutTensors);
    c10::intrusive_ptr<MkiTorch> clone() const { return c10::make_intrusive<MkiTorch>(opDescJsonStr_); }

private:
    std::string RunOp(Mki::LaunchParam &launchParam, const std::vector<Mki::Tensor> &inTensors,
                      std::vector<Mki::Tensor> &outTensors);
    std::string RunOpPerf(Mki::LaunchParam &launchParam, std::vector<Mki::Tensor> &inTensors,
                          std::vector<Mki::Tensor> &outTensors, int runTimes);
    std::string ExecuteImpl(std::vector<at::Tensor> &atInTensors, std::vector<at::Tensor> &atOutTensors);
    void *GetCurrentStream();
    Mki::Tensor AtTensor2MkiTensor(const at::Tensor &atTensor);
    void ContiguousAtTensor(std::vector<at::Tensor> &atTensors);

    std::string AddWorkspace(const Mki::KernelInfo &kernelInfo, Mki::RunInfo &runInfo);
    std::string FreeWorkspace(const Mki::KernelInfo &kernelInfo, Mki::RunInfo &runInfo);

private:
    bool launchWithTiling_ = true;
    std::string opDescJsonStr_;
    std::string opName_;
};

#endif
