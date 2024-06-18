/*
 * Copyright(C) 2023. Huawei Technologies Co.,Ltd. All rights reserved.
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
#ifndef FLOATUTIL_H
#define FLOATUTIL_H

#include <vector>

#include <ATen/ATen.h>

#include "mki/utils/log/log.h"
#include "mki/utils/status/status.h"

class FloatUtil {
public:
    /**
     * @brief 指定精度下判断两float数据是否相同
     * @param excepted      期望数据
     * @param actual        真实计算数据
     * @param atol          绝对误差的阈值，例如0.001
     * @param rtol          相对误差的阈值，例如0.001
     * @return 数据相同返回true，不同返回false
    */
    static bool FloatJudgeEqual(float expected, float actual, float atol, float rtol);

    /**
     * @brief 生成指定大小范围的float类型随机数
     * @param lower         随机数大小下限
     * @param upper         随机数大小上限
     * @param code          输出，生成的随机数vector，数据类型转为T
    */
    template <class T> static void GenerateCode(float lower, float upper, std::vector<T> &code);

    template <typename T>
    static Mki::Status MatchAtTensorFloat(at::Tensor &out, at::Tensor &gt, float atol, float rtol)
    {
        T *result = static_cast<T *>(out.storage().data_ptr().get());
        T *expect = static_cast<T *>(gt.storage().data_ptr().get());
        MKI_LOG(INFO) << "MatchAtTensorFloat";
        for (int i = 0; i < out.numel(); i++) {
            if (i < 10) {
                MKI_LOG(INFO) << "Index " << i << ", Expect " << expect[i] << ", Actual " << result[i];
            }
            if (!FloatJudgeEqual(expect[i], result[i], atol, rtol)) {
                std::string msg = "pos " + std::to_string(i) +
                                  ", expect: " + std::to_string(static_cast<float>(expect[i])) +
                                  ", result: " + std::to_string(static_cast<float>(result[i]));
                return Mki::Status::FailStatus(-1, msg);
            }
        }
        return Mki::Status::OkStatus();
    }
};

#endif
