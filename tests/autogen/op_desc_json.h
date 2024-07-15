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
#ifndef MKI_DESC_JSON_H
#define MKI_DESC_JSON_H

#include <nlohmann/json.hpp>
#include "mki/launch_param.h"

#ifdef __cplusplus
extern "C" {
#endif

void JsonToOpParam(const nlohmann::json &opDescJson, Mki::LaunchParam &launchParam);

#ifdef __cplusplus
}
#endif

#endif
