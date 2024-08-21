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
#ifndef MKI_PARAM_TO_STRING_H
#define MKI_PARAM_TO_STRING_H

#include <functional>
#include <map>
#include <string>
#include "mki/utils/any/any.h"

namespace Mki {
using ToStringFunc = std::function<std::string(const Any &)>;

class ParamToString {
public:
    ParamToString(size_t typeHashCode, ToStringFunc func);
    static std::string ToString(const Any &param);
private:
    static std::map<size_t, ToStringFunc> converterMap_;
};
} // namespace Mki

#define REG_PARAM_TO_STRING(typename, func) \
    static Mki::ParamToString g_register_##func(typeid(typename).hash_code(), func)

#endif
