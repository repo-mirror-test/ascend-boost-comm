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

#ifndef UTILS_PLATFORM_PLATFORM_INFOS_IMPL_H
#define UTILS_PLATFORM_PLATFORM_INFOS_IMPL_H

#include <map>
#include <string>
#include <vector>
#include <memory>
#include "platform/platform_infos_def.h"

namespace fe {
class PlatFormInfosImpl {
public:
    std::map<std::string, std::vector<std::string>> GetAICoreIntrinsicDtype();
    std::map<std::string, std::vector<std::string>> GetVectorCoreIntrinsicDtype();
    bool GetPlatformRes(const std::string &label, const std::string &key, std::string &val);
    bool GetPlatformRes(const std::string &label, std::map<std::string, std::string> &res);

    void SetAICoreIntrinsicDtype(std::map<std::string, std::vector<std::string>> &intrinsicDtypes);
    void SetVectorCoreIntrinsicDtype(std::map<std::string, std::vector<std::string>> &intrinsicDtypes);
    void SetPlatformRes(const std::string &label, std::map<std::string, std::string> &res);
    std::map<std::string, std::vector<std::string>> GetFixPipeDtypeMap();
    void SetFixPipeDtypeMap(const std::map<std::string, std::vector<std::string>> &fixpipeDtypeMap);

private:
    std::map<std::string, std::vector<std::string>> aiCoreIntrinsicDtypeMap_;
    std::map<std::string, std::vector<std::string>> vectorCoreIntrinsicDtypeMap_;
    std::map<std::string, std::map<std::string, std::string>> platformResMap_;
    std::map<std::string, std::vector<std::string>> fixpipeDtypeMap_;
};
} // namespace fe

#endif
