/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * AscendOpCommonLib is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */
#ifndef ASCEND_UNIT_TEST_DEVICE_VERSION_CHECK_H
#define ASCEND_UNIT_TEST_DEVICE_VERSION_CHECK_H

#include "mki/utils/log/log.h"

#define CHECK_DEVICE_VERSION_ASCEND910B()                                                                              \
    do {                                                                                                               \
        if (!(IsAscend910B())) {                                                                                       \
            MKI_LOG(WARN) << "check device, not in 910b, skip testcase";                                               \
            return;                                                                                                    \
        }                                                                                                              \
    } while (0)

#define CHECK_DEVICE_VERSION_NOT_ASCEND910B()                                                                          \
    do {                                                                                                               \
        if ((IsAscend910B())) {                                                                                        \
            MKI_LOG(WARN) << "check device, in 910b, skip testcase";                                                   \
            return;                                                                                                    \
        }                                                                                                              \
    } while (0)

#define CHECK_DEVICE_VERSION_ASCEND910A()                                                                              \
    do {                                                                                                               \
        if (!(IsAscend910A())) {                                                                                       \
            MKI_LOG(WARN) << "check device, not in 910a, skip testcase";                                               \
            return;                                                                                                    \
        }                                                                                                              \
    } while (0)

#define CHECK_DEVICE_VERSION_NOT_ASCEND910A()                                                                          \
    do {                                                                                                               \
        if ((IsAscend910A())) {                                                                                        \
            MKI_LOG(WARN) << "check device, in 910a, skip testcase";                                                   \
            return;                                                                                                    \
        }                                                                                                              \
    } while (0)

#define CHECK_DEVICE_VERSION_ASCEND310P()                                                                              \
    do {                                                                                                               \
        if (!(IsAscend310P())) {                                                                                       \
            MKI_LOG(WARN) << "check device, not in 310p, skip testcase";                                               \
            return;                                                                                                    \
        }                                                                                                              \
    } while (0)

#define CHECK_DEVICE_VERSION_NOT_ASCEND310P()                                                                          \
    do {                                                                                                               \
        if ((IsAscend310P())) {                                                                                        \
            MKI_LOG(WARN) << "check device, in 310p, skip testcase";                                                   \
            return;                                                                                                    \
        }                                                                                                              \
    } while (0)

bool IsAscend310P();
bool IsAscend910A();
bool IsAscend910B();
#endif // ASCEND_UNIT_TEST_DEVICE_VERSION_CHECK_H