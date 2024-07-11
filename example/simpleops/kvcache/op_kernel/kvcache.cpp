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

#include "kvcache_nd_init_tiling.h"

// tiling hiddensize、maxseqlen

namespace {
constexpr uint32_t MAX_UB_SIZE = 192 * 1024; // UB size
constexpr uint32_t BLOCK_SIZE = 32;
} // namespace

template <typename T>
class KvCacheNd {
public:
    __aicore__ inline KvCacheNd(uint32_t batch, uint32_t hiddenSize, uint32_t maxSeqLen)
    {
        this->hiddenSize = hiddenSize;
        this->maxSeqLen = maxSeqLen;
        this->batch = batch;
        this->blockElemNum = BLOCK_SIZE / sizeof(T);
        this->maxElemNumPerTime = MAX_UB_SIZE / sizeof(T);
        this->blockIdx = AscendC::GetBlockIdx();
    }

    __aicore__ inline void Init(__gm__ uint8_t *newKV, __gm__ uint8_t *layerId, __gm__ uint8_t *cacheIn,
                                __gm__ uint8_t *tokenOffset, __gm__ uint8_t *seqLen, __gm__ uint8_t *cacheOut,
                                __gm__ uint8_t *tiling)
    {
        newKVGm = (__gm__ T *)newKV;
        layerIdGm = (__gm__ uint32_t *)layerId;
        cacheInGm = (__gm__ T *)cacheIn;
        tokenOffsetGm = (__gm__ uint32_t *)tokenOffset;
        seqLenGm = (__gm__ uint32_t *)seqLen;
        cacheOutGm = (__gm__ T *)cacheOut;
        pipe.InitBuffer(outQueue, 1, MAX_UB_SIZE);
        AscendC::LocalTensor<T> cachePerloopUb = outQueue.AllocTensor<T>();
        this->tokenOffset = *(this->tokenOffsetGm + blockIdx);
        this->seqLen = *(this->seqLenGm + blockIdx);
        this->tokenOffset -= this->seqLen;
        this->layerId = *(layerIdGm);
        for (uint32_t i = 0; i < blockIdx; ++i) {
            prefixOfNtokens += *(this->seqLenGm + i);
        }

        cacheBuff = cacheInGm;
        cacheBuff = cacheBuff + this->layerId * this->batch * this->maxSeqLen * this->hiddenSize;
        WriteKvCache(newKVGm, cacheBuff, 0, cachePerloopUb);
        outQueue.FreeTensor(cachePerloopUb);
    }

    __aicore__ inline void WriteKvCache(__gm__ T *oldCache, __gm__ T *newCache, uint64_t kvOffset,
                                        AscendC::LocalTensor<T> &cachePerloopUb)
    {
        if (g_coreType == AscendC::AIC) {
            return;
        }
        AscendC::GlobalTensor<T> oldCacheGm;
        AscendC::GlobalTensor<T> newCacheGm;
        oldCacheGm.SetGlobalBuffer((__gm__ T *)oldCache);
        newCacheGm.SetGlobalBuffer((__gm__ T *)newCache);

        uint64_t tokensPerLoop = maxElemNumPerTime / hiddenSize;
        uint64_t loopTimes = seqLen / tokensPerLoop;
        uint64_t tailTokens = seqLen % tokensPerLoop;
        uint64_t tailLoopTimes = tailTokens == 0 ? 0 : 1;

        uint64_t qkvSplitBatchOffset = prefixOfNtokens * hiddenSize;
        uint64_t kvCacheBatchOffset = blockIdx * maxSeqLen * hiddenSize + tokenOffset * hiddenSize;

        AscendC::DataCopyParams copyParams = {static_cast<uint16_t>(tokensPerLoop),
                                              static_cast<uint16_t>(hiddenSize / blockElemNum), 0, 0};

        for (uint64_t loop = 0; loop < loopTimes; ++loop) {
            DataCopy(cachePerloopUb, oldCacheGm[qkvSplitBatchOffset + kvOffset + loop * tokensPerLoop * hiddenSize],
                     copyParams);
            AscendC::PipeBarrier<PIPE_ALL>();
            DataCopy(newCacheGm[kvCacheBatchOffset + loop * tokensPerLoop * hiddenSize], cachePerloopUb, copyParams);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        // process tail
        copyParams = {static_cast<uint16_t>(tailTokens), static_cast<uint16_t>(hiddenSize / blockElemNum), 0, 0};
        for (uint64_t loop = 0; loop < tailLoopTimes; ++loop) {
            DataCopy(cachePerloopUb,
                     oldCacheGm[qkvSplitBatchOffset + kvOffset + loopTimes * tokensPerLoop * hiddenSize], copyParams);
            AscendC::PipeBarrier<PIPE_ALL>();
            DataCopy(newCacheGm[kvCacheBatchOffset + loopTimes * tokensPerLoop * hiddenSize], cachePerloopUb,
                     copyParams);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

private:
    /* data */
    __gm__ T *newKVGm{nullptr};
    __gm__ uint32_t *layerIdGm{nullptr};
    __gm__ T *cacheInGm{nullptr};
    __gm__ uint32_t *tokenOffsetGm{nullptr};
    __gm__ uint32_t *seqLenGm{nullptr};
    __gm__ T *cacheOutGm{nullptr};
    __gm__ T *cacheBuff{nullptr};
    uint64_t batch{0};
    uint64_t hiddenSize{0};
    uint64_t maxSeqLen{0};
    uint64_t layerId{0};
    uint64_t prefixOfNtokens{0};
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> outQueue;
    uint64_t tokenOffset{0};
    uint64_t seqLen{0};
    uint32_t blockElemNum{16};
    uint32_t maxElemNumPerTime{};
    uint64_t blockIdx{0};
};

extern "C" __global__ __aicore__ void kvcache(GM_ADDR newKV, GM_ADDR layerId, GM_ADDR cacheIn, GM_ADDR tokenOffset,
                                              GM_ADDR seqLen, GM_ADDR cacheOut, GM_ADDR tiling)
{
    SimpleOps::KVCacheTilingData tilingData;
    InitTilingData(tiling, &(tilingData));
    if (TILING_KEY_IS(2000000000)) {
        KvCacheNd<half> op(tilingData.batch, tilingData.hiddenSize, tilingData.maxSeqLen);
        op.Init(newKV, layerId, cacheIn, tokenOffset, seqLen, cacheOut, tiling);
    }
    if (TILING_KEY_IS(2100000000)) {
        KvCacheNd<int8_t> op(tilingData.batch, tilingData.hiddenSize, tilingData.maxSeqLen);
        op.Init(newKV, layerId, cacheIn, tokenOffset, seqLen, cacheOut, tiling);
    }
}