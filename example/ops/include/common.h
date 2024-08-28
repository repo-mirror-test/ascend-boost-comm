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

#ifndef __force_inline__
#define __force_inline__ inline __attribute__((always_inline))
#endif

#endif
