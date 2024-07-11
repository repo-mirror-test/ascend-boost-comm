
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
#include "mki/utils/log/log.h"
#include "op_desc_json.h"

void JsonToOpParam(const nlohmann::json &opDescJson, Mki::LaunchParam &launchParam)
{
    MKI_LOG(ERROR) << "Please replace the libmki_test_autogen.so with your own version!";
}

Mki::Operation *GetOpByName(const std::string &operationName)
{
    MKI_LOG(ERROR) << "Please replace the libmki_test_autogen.so with your own version!";
    return nullptr;
}
