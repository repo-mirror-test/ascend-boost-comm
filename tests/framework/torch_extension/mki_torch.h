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
#ifndef MKI_TORCH_H
#define MKI_TORCH_H
#include <string>
#include <vector>
#include <torch/script.h>
#include <torch/custom_class.h>
#include "mki/ops.h"
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
    void *GetCurrentStream() const;
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
