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
#ifndef COMMON_VAL_H
#define COMMON_VAL_H
const constexpr uint32_t NUM_TWO = 2; // 2
const constexpr uint32_t BLK_SIZE = 32; // 一个block字节数
const constexpr uint32_t ELE_NUM_FP16 = 16; // 一个block fp16元素个数
const constexpr uint32_t ELE_NUM_FP32 = 8; // 一个block字节数 fp32元素个数
const constexpr uint32_t MAX_LEN_FP16 = 8192; // 非fp16情况下最大长度（hiddensize）
const constexpr uint8_t DEFAULT_REPEAT_STRIDE = 8; // 默认stride, 8 * 32 = 256
const constexpr int64_t REG_910B = 48; // 饱和模式寄存器位置
const constexpr int64_t REG_310P = 53; // 饱和模式寄存器位置
const constexpr int64_t SLICE_SIZE = 4096; // 切片大小
const constexpr int64_t SLICE_SIZE_FP16 = 12288;
#endif