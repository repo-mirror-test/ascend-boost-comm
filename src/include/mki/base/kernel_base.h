/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
#ifndef CORE_BASE_KERNEL_BASE_H
#define CORE_BASE_KERNEL_BASE_H
#include <string>
#include <vector>
#include <functional>
#include <map>
#include "mki/kernel.h"
#include "mki/kernel_info.h"
#include "mki/run_info.h"
#include "mki/utils/rt/rt.h"
#include "mki/bin_handle.h"

namespace Mki {
class KernelBase : public Kernel {
using KernelSelfCreator = std::function<KernelBase*(void)>;
public:
    KernelBase() = delete;
    KernelBase(const KernelBase &) = delete;
    KernelBase &operator=(const KernelBase &) = delete;
    KernelBase(const std::string &opName, const BinHandle *handle);
    ~KernelBase() override;
    Kernel *Clone() const override;
    void Reset() override;

    bool CanSupport(const LaunchParam &launchParam) const override;
    uint64_t GetTilingSize(const LaunchParam &launchParam) const override;
    Status Init(const LaunchParam &launchParam) override;
    Status Run(const LaunchParam &launchParam, RunInfo &runInfo) override;

    void SetLaunchWithTiling(bool flag) override;
    void SetTilingHostAddr(uint8_t *addr, uint64_t len) override;
    std::string GetName() const override;
    const KernelInfo &GetKernelInfo() const override;
    KernelType GetType() const override;

protected:
    virtual Status InitImpl(const LaunchParam &launchParam);
    const BinHandle *GetBinHandle() const;
    void Copy(const KernelBase &other);
    uint32_t launchBufferSize_ = 0;

private:
    uint64_t GetKernelParamNum(const LaunchParam &launchParam);

    void UpdateKernelType();
    Status UpdateHwsyncArgs(void **args, uint64_t argsNum);
    Status UpdateConstTensorArgs(void **args, uint64_t argsNum,
                                 RtHostInputInfoT *info, uint64_t infoNum);
    Status UpdateConstTensorArgs(void **args, uint64_t argsNum, const RunInfo &runInfo);
    Status UpdateInOutWkspArgs(void **args, uint64_t argsNum, const LaunchParam &launchParam, const RunInfo &runInfo);
    Status UpdateTilingArgs(RtArgsExT &argsEx, uint64_t argsNum, RunInfo &runInfo) const;
    Status UpdateTilingArgs(void **args, uint64_t argsNum, RunInfo &runInfo) const;
    Status MemsetTensorArgs(void **args, uint64_t argsNum, const RunInfo &runInfo);

    std::string kernelName_;
    KernelType kernelType_{KernelType::KERNEL_TYPE_INVALID};
    const BinHandle *handle_{nullptr};
    KernelInfo kernelInfo_;
    bool initFlag_{false};
    KernelSelfCreator creator_{nullptr};
    friend void SetKernelSelfCreator(KernelBase &kernel, KernelSelfCreator func);
};

void SetKernelSelfCreator(KernelBase &kernel, KernelBase::KernelSelfCreator func);
} // namespace Mki

#endif
