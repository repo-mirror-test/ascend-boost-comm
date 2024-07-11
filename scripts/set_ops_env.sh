#!/bin/bash
# Copyright(C) 2023. Huawei Technologies Co.,Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
path="${BASH_SOURCE[0]}"
if [[ -f "$path" ]] && [[ "$path" =~ 'set_ops_env.sh' ]];then
    mki_ops_path=$(cd $(dirname $path); pwd )
    export MKI_OPS_HOME_PATH="${mki_ops_path}"
    export LD_LIBRARY_PATH=$MKI_OPS_HOME_PATH/ops_lib:$MKI_OPS_HOME_PATH/tests:$LD_LIBRARY_PATH
else
    echo "There is no 'set_ops_env.sh' to import"
fi
