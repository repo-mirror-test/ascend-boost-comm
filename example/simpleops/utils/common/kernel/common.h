/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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
#ifndef INCLUDE_COMMON_H
#define INCLUDE_COMMON_H

#define CONST_2 2

#ifdef USE_ASCENDC
#define SET_FLAG(trigger, waiter, e) AscendC::SetFlag<AscendC::HardEvent::trigger##_##waiter>((e))
#define WAIT_FLAG(trigger, waiter, e) AscendC::WaitFlag<AscendC::HardEvent::trigger##_##waiter>((e))
#define PIPE_BARRIER(pipe) AscendC::PipeBarrier<PIPE_##pipe>()
#else
#define SET_FLAG(trigger, waiter, e) set_flag(PIPE_##trigger, PIPE_##waiter, (e))
#define WAIT_FLAG(trigger, waiter, e) wait_flag(PIPE_##trigger, PIPE_##waiter, (e))
#define PIPE_BARRIER(pipe) pipe_barrier(PIPE_##pipe)
#endif

#endif
