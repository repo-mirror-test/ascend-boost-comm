/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * AscendOpCommonLib is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 * http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#ifndef INCLUDE_COMMON_FUNC_H
#define INCLUDE_COMMON_FUNC_H

#include <type_traits>
#include <limits>
#include "kernel_operator.h"

template <uint32_t ALIGN, typename T = uint32_t> inline __aicore__ T RoundUp(const T val)
{
    static_assert(ALIGN != 0, "align must not be zero");
    static_assert(std::is_arithmetic<T>::value, "T must be an arithmetic type");
    T align = ALIGN;
    return (val + align - 1) / align * align;
}

template <typename T> inline __aicore__ T RoundUp(const T val, const T align)
{
    static_assert(std::is_arithmetic<T>::value, "T must be an arithmetic type");
    if (align == 0) {
        return 0;
    } else {
        return (val + align - 1) / align * align;
    }
}

template <uint32_t DIVISOR, typename T = uint32_t> inline __aicore__ T CeilDiv(const T dividend)
{
    static_assert(DIVISOR != 0, "align must not be zero");
    static_assert(std::is_arithmetic<T>::value, "T must be an arithmetic type");
    T divisor = DIVISOR;
    return (dividend + divisor - 1) / divisor;
}

template <typename T> constexpr T T_MAX = std::numeric_limits<T>::max();

template <typename T> inline __aicore__ T CeilDiv(const T dividend, const T divisor)
{
    static_assert(std::is_arithmetic<T>::value, "T must be an arithmetic type");
    if (divisor == 0) {
        return T_MAX<T>;
    } else {
        return (dividend + divisor - 1) / divisor;
    }
}

template <typename T>
__aicore__ inline T Min(const T lhs, const T rhs)
{
    return lhs < rhs ? lhs : rhs;
}

#endif
