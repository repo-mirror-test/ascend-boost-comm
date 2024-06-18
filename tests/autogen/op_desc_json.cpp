
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
#include <functional>
#include <map>
#include "aki/utils/log/log.h"
#include "aki/params/params.h"
#include "op_desc_json.h"

void CumsumJson(const nlohmann::json &opDescJson, Aki::LaunchParam &launchParam)
{
    Aki::OpParam::Cumsum param;
    try {
        std::vector<int64_t> axisValues = opDescJson["specificParam"].at("axis");
        const int axisSizes = int(axisValues.size());
        param.axis.resize(axisSizes);
        for(int i = 0; i < axisSizes; ++i){
            param.axis[i] = axisValues[i];
        }
    } catch(...) {
        AKI_LOG(INFO) << "axis not set, use default";
    }
    try {
        param.exclusive = opDescJson["specificParam"].at("exclusive");
    } catch (...) {
        AKI_LOG(INFO) << "exclusive not set, use default";
    }
    try {
        param.reverse = opDescJson["specificParam"].at("reverse");
    } catch (...) {
        AKI_LOG(INFO) << "reverse not set, use default";
    }
    try {
        param.deterministic = opDescJson["specificParam"].at("deterministic");
    } catch (...) {
        AKI_LOG(INFO) << "deterministic not set, use default";
    }
    launchParam.SetParam(param);
}

void SortJson(const nlohmann::json &opDescJson, Aki::LaunchParam &launchParam)
{
    Aki::OpParam::Sort param;
    try {
        std::vector<int32_t> numValues = opDescJson["specificParam"].at("num");
        const int numSizes = int(numValues.size());
        param.num.resize(numSizes);
        for(int i = 0; i < numSizes; ++i){
            param.num[i] = numValues[i];
        }
    } catch(...) {
        AKI_LOG(INFO) << "num not set, use default";
    }
    launchParam.SetParam(param);
}

using LaunchParamSetFunc = std::function<void(const nlohmann::json &, Aki::LaunchParam &)>;

static const std::map<std::string, LaunchParamSetFunc> OP_DESC_JSON_FUNC_MAP = {
    {"Cumsum", CumsumJson},
    {"Sort", SortJson},
};

void JsonToOpParam(const nlohmann::json &opDescJson, Aki::LaunchParam &launchParam)
{
    AKI_LOG(INFO) << opDescJson.dump(4);
    std::string category = opDescJson["category"];
    auto paramFunc = OP_DESC_JSON_FUNC_MAP.find(category);
    if (paramFunc == OP_DESC_JSON_FUNC_MAP.end()) {
        AKI_LOG(ERROR) << "no category " << category;
        return;
    }
    try {
        paramFunc->second(opDescJson, launchParam);
    } catch(const std::exception &e) {
        AKI_LOG(ERROR) << "convert json fail, error:" << e.what();
    }
}
