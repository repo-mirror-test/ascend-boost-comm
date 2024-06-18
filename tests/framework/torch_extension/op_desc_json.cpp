
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
#include "asdops/utils/log/log.h"
#include "asdops/params/params.h"
#include "op_desc_json.h"

void OnehotJson(const nlohmann::json &opDescJson, AsdOps::OpDesc &opDesc)
{
    AsdOps::OpParam::Onehot param;
    try {
        param.axis = opDescJson["specificParam"].at("axis");
    } catch (...) {
        ASD_LOG(INFO) << "axis not set, use default";
    }
    try {
        std::vector<int64_t> depthValues = opDescJson["specificParam"].at("depth");
        const int depthSizes = int(depthValues.size());
        param.depth.resize(depthSizes);
        for(int i = 0; i < depthSizes; ++i){
            param.depth[i] = depthValues[i];
        }
    } catch(...) {
        ASD_LOG(INFO) << "depth not set, use default";
    }
    opDesc.specificParam = param;
}

void SliceJson(const nlohmann::json &opDescJson, AsdOps::OpDesc &opDesc)
{
    AsdOps::OpParam::Slice param;
    try {
        std::vector<int64_t> offsetsValues = opDescJson["specificParam"].at("offsets");
        const int offsetsSizes = int(offsetsValues.size());
        param.offsets.resize(offsetsSizes);
        for(int i = 0; i < offsetsSizes; ++i){
            param.offsets[i] = offsetsValues[i];
        }
    } catch(...) {
        ASD_LOG(INFO) << "offsets not set, use default";
    }
    try {
        std::vector<int64_t> sizeValues = opDescJson["specificParam"].at("size");
        const int sizeSizes = int(sizeValues.size());
        param.size.resize(sizeSizes);
        for(int i = 0; i < sizeSizes; ++i){
            param.size[i] = sizeValues[i];
        }
    } catch(...) {
        ASD_LOG(INFO) << "size not set, use default";
    }
    opDesc.specificParam = param;
}

void ConcatJson(const nlohmann::json &opDescJson, AsdOps::OpDesc &opDesc)
{
    AsdOps::OpParam::Concat param;
    try {
        param.concatDim = opDescJson["specificParam"].at("concatDim");
    } catch (...) {
        ASD_LOG(INFO) << "concatDim not set, use default";
    }
    opDesc.specificParam = param;
}

void NormJson(const nlohmann::json &opDescJson, AsdOps::OpDesc &opDesc)
{
    AsdOps::OpParam::Norm param;
    try {
        param.normType = static_cast<AsdOps::OpParam::Norm::NormType>(
            opDescJson["specificParam"].at("normType"));
    } catch (...) {
        ASD_LOG(INFO) << "normType not set, use default";
    }
    try {
        param.beginNormAxis = opDescJson["specificParam"].at("beginNormAxis");
    } catch (...) {
        ASD_LOG(INFO) << "beginNormAxis not set, use default";
    }
    try {
        param.beginParamsAxis = opDescJson["specificParam"].at("beginParamsAxis");
    } catch (...) {
        ASD_LOG(INFO) << "beginParamsAxis not set, use default";
    }
    try {
        param.opsMode = opDescJson["specificParam"].at("opsMode");
    } catch (...) {
        ASD_LOG(INFO) << "opsMode not set, use default";
    }
    try {
        param.epsilon = opDescJson["specificParam"].at("epsilon");
    } catch (...) {
        ASD_LOG(INFO) << "epsilon not set, use default";
    }
    try {
        param.zoomScaleValue = opDescJson["specificParam"].at("zoomScaleValue");
    } catch (...) {
        ASD_LOG(INFO) << "zoomScaleValue not set, use default";
    }
    try {
        param.precisionMode = opDescJson["specificParam"].at("precisionMode");
    } catch (...) {
        ASD_LOG(INFO) << "precisionMode not set, use default";
    }
    try {
        param.gemmaMode = opDescJson["specificParam"].at("gemmaMode");
    } catch (...) {
        ASD_LOG(INFO) << "gemmaMode not set, use default";
    }
    try {
        param.inGamma = opDescJson["specificParam"].at("inGamma");
    } catch (...) {
        ASD_LOG(INFO) << "inGamma not set, use default";
    }
    try {
        param.inBeta = opDescJson["specificParam"].at("inBeta");
    } catch (...) {
        ASD_LOG(INFO) << "inBeta not set, use default";
    }
    try {
        param.inRes = opDescJson["specificParam"].at("inRes");
    } catch (...) {
        ASD_LOG(INFO) << "inRes not set, use default";
    }
    try {
        param.inNormBias = opDescJson["specificParam"].at("inNormBias");
    } catch (...) {
        ASD_LOG(INFO) << "inNormBias not set, use default";
    }
    try {
        param.outMean = opDescJson["specificParam"].at("outMean");
    } catch (...) {
        ASD_LOG(INFO) << "outMean not set, use default";
    }
    try {
        param.outVarience = opDescJson["specificParam"].at("outVarience");
    } catch (...) {
        ASD_LOG(INFO) << "outVarience not set, use default";
    }
    try {
        param.outResQuant = opDescJson["specificParam"].at("outResQuant");
    } catch (...) {
        ASD_LOG(INFO) << "outResQuant not set, use default";
    }
    try {
        param.outRes = opDescJson["specificParam"].at("outRes");
    } catch (...) {
        ASD_LOG(INFO) << "outRes not set, use default";
    }
    opDesc.specificParam = param;
}

void CumsumJson(const nlohmann::json &opDescJson, AsdOps::OpDesc &opDesc)
{
    AsdOps::OpParam::Cumsum param;
    try {
        std::vector<int64_t> axisValues = opDescJson["specificParam"].at("axis");
        const int axisSizes = int(axisValues.size());
        param.axis.resize(axisSizes);
        for(int i = 0; i < axisSizes; ++i){
            param.axis[i] = axisValues[i];
        }
    } catch(...) {
        ASD_LOG(INFO) << "axis not set, use default";
    }
    try {
        param.exclusive = opDescJson["specificParam"].at("exclusive");
    } catch (...) {
        ASD_LOG(INFO) << "exclusive not set, use default";
    }
    try {
        param.reverse = opDescJson["specificParam"].at("reverse");
    } catch (...) {
        ASD_LOG(INFO) << "reverse not set, use default";
    }
    try {
        param.deterministic = opDescJson["specificParam"].at("deterministic");
    } catch (...) {
        ASD_LOG(INFO) << "deterministic not set, use default";
    }
    opDesc.specificParam = param;
}

void TransposeJson(const nlohmann::json &opDescJson, AsdOps::OpDesc &opDesc)
{
    AsdOps::OpParam::Transpose param;
    try {
        std::vector<int32_t> permValues = opDescJson["specificParam"].at("perm");
        const int permSizes = int(permValues.size());
        param.perm.resize(permSizes);
        for(int i = 0; i < permSizes; ++i){
            param.perm[i] = permValues[i];
        }
    } catch(...) {
        ASD_LOG(INFO) << "perm not set, use default";
    }
    opDesc.specificParam = param;
}

void FillJson(const nlohmann::json &opDescJson, AsdOps::OpDesc &opDesc)
{
    AsdOps::OpParam::Fill param;
    try {
        param.withMask = opDescJson["specificParam"].at("withMask");
    } catch (...) {
        ASD_LOG(INFO) << "withMask not set, use default";
    }
    try {
        std::vector<float> valueValues = opDescJson["specificParam"].at("value");
        const int valueSizes = int(valueValues.size());
        param.value.resize(valueSizes);
        for(int i = 0; i < valueSizes; ++i){
            param.value[i] = valueValues[i];
        }
    } catch(...) {
        ASD_LOG(INFO) << "value not set, use default";
    }
    try {
        std::vector<int64_t> outDimValues = opDescJson["specificParam"].at("outDim");
        const int outDimSizes = int(outDimValues.size());
        param.outDim.resize(outDimSizes);
        for(int i = 0; i < outDimSizes; ++i){
            param.outDim[i] = outDimValues[i];
        }
    } catch(...) {
        ASD_LOG(INFO) << "outDim not set, use default";
    }
    opDesc.specificParam = param;
}

void TransdataJson(const nlohmann::json &opDescJson, AsdOps::OpDesc &opDesc)
{
    AsdOps::OpParam::Transdata param;
    try {
        param.transdataType = static_cast<AsdOps::OpParam::Transdata::TransdataType>(
            opDescJson["specificParam"].at("transdataType"));
    } catch (...) {
        ASD_LOG(INFO) << "transdataType not set, use default";
    }
    try {
        std::vector<int64_t> outCropsValues = opDescJson["specificParam"].at("outCrops");
        const int outCropsSizes = int(outCropsValues.size());
        param.outCrops.resize(outCropsSizes);
        for(int i = 0; i < outCropsSizes; ++i){
            param.outCrops[i] = outCropsValues[i];
        }
    } catch(...) {
        ASD_LOG(INFO) << "outCrops not set, use default";
    }
    opDesc.specificParam = param;
}

void SortJson(const nlohmann::json &opDescJson, AsdOps::OpDesc &opDesc)
{
    AsdOps::OpParam::Sort param;
    try {
        std::vector<int32_t> numValues = opDescJson["specificParam"].at("num");
        const int numSizes = int(numValues.size());
        param.num.resize(numSizes);
        for(int i = 0; i < numSizes; ++i){
            param.num[i] = numValues[i];
        }
    } catch(...) {
        ASD_LOG(INFO) << "num not set, use default";
    }
    opDesc.specificParam = param;
}

void ElewiseJson(const nlohmann::json &opDescJson, AsdOps::OpDesc &opDesc)
{
    AsdOps::OpParam::Elewise param;
    try {
        param.elewiseType = static_cast<AsdOps::OpParam::Elewise::ElewiseType>(
            opDescJson["specificParam"].at("elewiseType"));
    } catch (...) {
        ASD_LOG(INFO) << "elewiseType not set, use default";
    }
    try {
        param.varAttr = opDescJson["specificParam"].at("varAttr");
    } catch (...) {
        ASD_LOG(INFO) << "varAttr not set, use default";
    }
    try {
        param.inputScale = opDescJson["specificParam"].at("inputScale");
    } catch (...) {
        ASD_LOG(INFO) << "inputScale not set, use default";
    }
    try {
        param.inputOffset = opDescJson["specificParam"].at("inputOffset");
    } catch (...) {
        ASD_LOG(INFO) << "inputOffset not set, use default";
    }
    try {
        param.asymmetric = opDescJson["specificParam"].at("asymmetric");
    } catch (...) {
        ASD_LOG(INFO) << "asymmetric not set, use default";
    }
    try {
        param.outTensorType = static_cast<AsdOps::TensorDType>(
            opDescJson["specificParam"].at("outTensorType"));
    } catch (...) {
        ASD_LOG(INFO) << "outTensorType not set, use default";
    }
    opDesc.specificParam = param;
}

void ActivationJson(const nlohmann::json &opDescJson, AsdOps::OpDesc &opDesc)
{
    AsdOps::OpParam::Activation param;
    try {
        param.activationType = static_cast<AsdOps::OpParam::Activation::ActivationType>(
            opDescJson["specificParam"].at("activationType"));
    } catch (...) {
        ASD_LOG(INFO) << "activationType not set, use default";
    }
    try {
        param.scale = opDescJson["specificParam"].at("scale");
    } catch (...) {
        ASD_LOG(INFO) << "scale not set, use default";
    }
    try {
        param.dim = opDescJson["specificParam"].at("dim");
    } catch (...) {
        ASD_LOG(INFO) << "dim not set, use default";
    }
    opDesc.specificParam = param;
}

void IndexJson(const nlohmann::json &opDescJson, AsdOps::OpDesc &opDesc)
{
    AsdOps::OpParam::Index param;
    try {
        param.indexType = static_cast<AsdOps::OpParam::Index::IndexType>(
            opDescJson["specificParam"].at("indexType"));
    } catch (...) {
        ASD_LOG(INFO) << "indexType not set, use default";
    }
    try {
        param.axis = opDescJson["specificParam"].at("axis");
    } catch (...) {
        ASD_LOG(INFO) << "axis not set, use default";
    }
    opDesc.specificParam = param;
}

void AsStridedJson(const nlohmann::json &opDescJson, AsdOps::OpDesc &opDesc)
{
    AsdOps::OpParam::AsStrided param;
    try {
        std::vector<int64_t> sizeValues = opDescJson["specificParam"].at("size");
        const int sizeSizes = int(sizeValues.size());
        param.size.resize(sizeSizes);
        for(int i = 0; i < sizeSizes; ++i){
            param.size[i] = sizeValues[i];
        }
    } catch(...) {
        ASD_LOG(INFO) << "size not set, use default";
    }
    try {
        std::vector<int64_t> strideValues = opDescJson["specificParam"].at("stride");
        const int strideSizes = int(strideValues.size());
        param.stride.resize(strideSizes);
        for(int i = 0; i < strideSizes; ++i){
            param.stride[i] = strideValues[i];
        }
    } catch(...) {
        ASD_LOG(INFO) << "stride not set, use default";
    }
    try {
        std::vector<int64_t> offsetValues = opDescJson["specificParam"].at("offset");
        const int offsetSizes = int(offsetValues.size());
        param.offset.resize(offsetSizes);
        for(int i = 0; i < offsetSizes; ++i){
            param.offset[i] = offsetValues[i];
        }
    } catch(...) {
        ASD_LOG(INFO) << "offset not set, use default";
    }
    opDesc.specificParam = param;
}

void ReverseJson(const nlohmann::json &opDescJson, AsdOps::OpDesc &opDesc)
{
    AsdOps::OpParam::Reverse param;
    try {
        std::vector<int32_t> axisValues = opDescJson["specificParam"].at("axis");
        const int axisSizes = int(axisValues.size());
        param.axis.resize(axisSizes);
        for(int i = 0; i < axisSizes; ++i){
            param.axis[i] = axisValues[i];
        }
    } catch(...) {
        ASD_LOG(INFO) << "axis not set, use default";
    }
    opDesc.specificParam = param;
}

void GatherJson(const nlohmann::json &opDescJson, AsdOps::OpDesc &opDesc)
{
    AsdOps::OpParam::Gather param;
    try {
        param.batchDims = opDescJson["specificParam"].at("batchDims");
    } catch (...) {
        ASD_LOG(INFO) << "batchDims not set, use default";
    }
    try {
        std::vector<int64_t> axisValues = opDescJson["specificParam"].at("axis");
        const int axisSizes = int(axisValues.size());
        param.axis.resize(axisSizes);
        for(int i = 0; i < axisSizes; ++i){
            param.axis[i] = axisValues[i];
        }
    } catch(...) {
        ASD_LOG(INFO) << "axis not set, use default";
    }
    opDesc.specificParam = param;
}

void MatMulJson(const nlohmann::json &opDescJson, AsdOps::OpDesc &opDesc)
{
    AsdOps::OpParam::MatMul param;
    try {
        param.transposeA = opDescJson["specificParam"].at("transposeA");
    } catch (...) {
        ASD_LOG(INFO) << "transposeA not set, use default";
    }
    try {
        param.transposeB = opDescJson["specificParam"].at("transposeB");
    } catch (...) {
        ASD_LOG(INFO) << "transposeB not set, use default";
    }
    try {
        std::vector<int64_t> oriShapeValues = opDescJson["specificParam"].at("oriShape");
        const int oriShapeSizes = int(oriShapeValues.size());
        param.oriShape.resize(oriShapeSizes);
        for(int i = 0; i < oriShapeSizes; ++i){
            param.oriShape[i] = oriShapeValues[i];
        }
    } catch(...) {
        ASD_LOG(INFO) << "oriShape not set, use default";
    }
    try {
        param.withBias = opDescJson["specificParam"].at("withBias");
    } catch (...) {
        ASD_LOG(INFO) << "withBias not set, use default";
    }
    try {
        param.enDequant = opDescJson["specificParam"].at("enDequant");
    } catch (...) {
        ASD_LOG(INFO) << "enDequant not set, use default";
    }
    try {
        param.tilingN = opDescJson["specificParam"].at("tilingN");
    } catch (...) {
        ASD_LOG(INFO) << "tilingN not set, use default";
    }
    try {
        param.tilingK = opDescJson["specificParam"].at("tilingK");
    } catch (...) {
        ASD_LOG(INFO) << "tilingK not set, use default";
    }
    try {
        param.enShuffleK = opDescJson["specificParam"].at("enShuffleK");
    } catch (...) {
        ASD_LOG(INFO) << "enShuffleK not set, use default";
    }
    try {
        param.outDtype = static_cast<AsdOps::TensorDType>(
            opDescJson["specificParam"].at("outDtype"));
    } catch (...) {
        ASD_LOG(INFO) << "outDtype not set, use default";
    }
    opDesc.specificParam = param;
}

void CopyJson(const nlohmann::json &opDescJson, AsdOps::OpDesc &opDesc)
{
    AsdOps::OpParam::Copy param;
    try {
        std::vector<int64_t> dstSizeValues = opDescJson["specificParam"].at("dstSize");
        const int dstSizeSizes = int(dstSizeValues.size());
        param.dstSize.resize(dstSizeSizes);
        for(int i = 0; i < dstSizeSizes; ++i){
            param.dstSize[i] = dstSizeValues[i];
        }
    } catch(...) {
        ASD_LOG(INFO) << "dstSize not set, use default";
    }
    try {
        std::vector<int64_t> dstStrideValues = opDescJson["specificParam"].at("dstStride");
        const int dstStrideSizes = int(dstStrideValues.size());
        param.dstStride.resize(dstStrideSizes);
        for(int i = 0; i < dstStrideSizes; ++i){
            param.dstStride[i] = dstStrideValues[i];
        }
    } catch(...) {
        ASD_LOG(INFO) << "dstStride not set, use default";
    }
    try {
        std::vector<int64_t> dstOffsetValues = opDescJson["specificParam"].at("dstOffset");
        const int dstOffsetSizes = int(dstOffsetValues.size());
        param.dstOffset.resize(dstOffsetSizes);
        for(int i = 0; i < dstOffsetSizes; ++i){
            param.dstOffset[i] = dstOffsetValues[i];
        }
    } catch(...) {
        ASD_LOG(INFO) << "dstOffset not set, use default";
    }
    opDesc.specificParam = param;
}

void ReduceJson(const nlohmann::json &opDescJson, AsdOps::OpDesc &opDesc)
{
    AsdOps::OpParam::Reduce param;
    try {
        param.reduceType = static_cast<AsdOps::OpParam::Reduce::ReduceType>(
            opDescJson["specificParam"].at("reduceType"));
    } catch (...) {
        ASD_LOG(INFO) << "reduceType not set, use default";
    }
    try {
        std::vector<int64_t> axisValues = opDescJson["specificParam"].at("axis");
        const int axisSizes = int(axisValues.size());
        param.axis.resize(axisSizes);
        for(int i = 0; i < axisSizes; ++i){
            param.axis[i] = axisValues[i];
        }
    } catch(...) {
        ASD_LOG(INFO) << "axis not set, use default";
    }
    opDesc.specificParam = param;
}

void ExpandJson(const nlohmann::json &opDescJson, AsdOps::OpDesc &opDesc)
{
    AsdOps::OpParam::Expand param;
    try {
        std::vector<int64_t> shapeValues = opDescJson["specificParam"].at("shape");
        const int shapeSizes = int(shapeValues.size());
        param.shape.resize(shapeSizes);
        for(int i = 0; i < shapeSizes; ++i){
            param.shape[i] = shapeValues[i];
        }
    } catch(...) {
        ASD_LOG(INFO) << "shape not set, use default";
    }
    opDesc.specificParam = param;
}

void MultinomialJson(const nlohmann::json &opDescJson, AsdOps::OpDesc &opDesc)
{
    AsdOps::OpParam::Multinomial param;
    try {
        param.numSamples = opDescJson["specificParam"].at("numSamples");
    } catch (...) {
        ASD_LOG(INFO) << "numSamples not set, use default";
    }
    try {
        param.randSeed = opDescJson["specificParam"].at("randSeed");
    } catch (...) {
        ASD_LOG(INFO) << "randSeed not set, use default";
    }
    opDesc.specificParam = param;
}

void SoftmaxJson(const nlohmann::json &opDescJson, AsdOps::OpDesc &opDesc)
{
    AsdOps::OpParam::Softmax param;
    try {
        std::vector<int64_t> axesValues = opDescJson["specificParam"].at("axes");
        const int axesSizes = int(axesValues.size());
        param.axes.resize(axesSizes);
        for(int i = 0; i < axesSizes; ++i){
            param.axes[i] = axesValues[i];
        }
    } catch(...) {
        ASD_LOG(INFO) << "axes not set, use default";
    }
    opDesc.specificParam = param;
}

void MixJson(const nlohmann::json &opDescJson, AsdOps::OpDesc &opDesc)
{
    AsdOps::OpParam::Mix param;
    try {
        param.mixType = static_cast<AsdOps::OpParam::Mix::MixType>(
            opDescJson["specificParam"].at("mixType"));
    } catch (...) {
        ASD_LOG(INFO) << "mixType not set, use default";
    }
    try {
        param.headSize = opDescJson["specificParam"].at("headSize");
    } catch (...) {
        ASD_LOG(INFO) << "headSize not set, use default";
    }
    try {
        std::vector<int32_t> qSeqLenValues = opDescJson["specificParam"].at("qSeqLen");
        const int qSeqLenSizes = int(qSeqLenValues.size());
        param.qSeqLen.resize(qSeqLenSizes);
        for(int i = 0; i < qSeqLenSizes; ++i){
            param.qSeqLen[i] = qSeqLenValues[i];
        }
    } catch(...) {
        ASD_LOG(INFO) << "qSeqLen not set, use default";
    }
    try {
        std::vector<int32_t> kvSeqLenValues = opDescJson["specificParam"].at("kvSeqLen");
        const int kvSeqLenSizes = int(kvSeqLenValues.size());
        param.kvSeqLen.resize(kvSeqLenSizes);
        for(int i = 0; i < kvSeqLenSizes; ++i){
            param.kvSeqLen[i] = kvSeqLenValues[i];
        }
    } catch(...) {
        ASD_LOG(INFO) << "kvSeqLen not set, use default";
    }
    try {
        param.tor = opDescJson["specificParam"].at("tor");
    } catch (...) {
        ASD_LOG(INFO) << "tor not set, use default";
    }
    try {
        param.kvHead = opDescJson["specificParam"].at("kvHead");
    } catch (...) {
        ASD_LOG(INFO) << "kvHead not set, use default";
    }
    try {
        std::vector<int32_t> batchRunStatusValues = opDescJson["specificParam"].at("batchRunStatus");
        const int batchRunStatusSizes = int(batchRunStatusValues.size());
        param.batchRunStatus.resize(batchRunStatusSizes);
        for(int i = 0; i < batchRunStatusSizes; ++i){
            param.batchRunStatus[i] = batchRunStatusValues[i];
        }
    } catch(...) {
        ASD_LOG(INFO) << "batchRunStatus not set, use default";
    }
    try {
        param.isClamp = opDescJson["specificParam"].at("isClamp");
    } catch (...) {
        ASD_LOG(INFO) << "isClamp not set, use default";
    }
    try {
        param.clampMin = opDescJson["specificParam"].at("clampMin");
    } catch (...) {
        ASD_LOG(INFO) << "clampMin not set, use default";
    }
    try {
        param.clampMax = opDescJson["specificParam"].at("clampMax");
    } catch (...) {
        ASD_LOG(INFO) << "clampMax not set, use default";
    }
    try {
        param.isTriuMask = opDescJson["specificParam"].at("isTriuMask");
    } catch (...) {
        ASD_LOG(INFO) << "isTriuMask not set, use default";
    }
    try {
        param.rotaryCoeff = opDescJson["specificParam"].at("rotaryCoeff");
    } catch (...) {
        ASD_LOG(INFO) << "rotaryCoeff not set, use default";
    }
    try {
        param.cosFormat = opDescJson["specificParam"].at("cosFormat");
    } catch (...) {
        ASD_LOG(INFO) << "cosFormat not set, use default";
    }
    try {
        param.headNum = opDescJson["specificParam"].at("headNum");
    } catch (...) {
        ASD_LOG(INFO) << "headNum not set, use default";
    }
    try {
        param.batch = opDescJson["specificParam"].at("batch");
    } catch (...) {
        ASD_LOG(INFO) << "batch not set, use default";
    }
    try {
        param.transA = opDescJson["specificParam"].at("transA");
    } catch (...) {
        ASD_LOG(INFO) << "transA not set, use default";
    }
    try {
        param.transB = opDescJson["specificParam"].at("transB");
    } catch (...) {
        ASD_LOG(INFO) << "transB not set, use default";
    }
    try {
        std::vector<int32_t> mValues = opDescJson["specificParam"].at("m");
        const int mSizes = int(mValues.size());
        param.m.resize(mSizes);
        for(int i = 0; i < mSizes; ++i){
            param.m[i] = mValues[i];
        }
    } catch(...) {
        ASD_LOG(INFO) << "m not set, use default";
    }
    try {
        std::vector<int32_t> kValues = opDescJson["specificParam"].at("k");
        const int kSizes = int(kValues.size());
        param.k.resize(kSizes);
        for(int i = 0; i < kSizes; ++i){
            param.k[i] = kValues[i];
        }
    } catch(...) {
        ASD_LOG(INFO) << "k not set, use default";
    }
    try {
        std::vector<int32_t> nValues = opDescJson["specificParam"].at("n");
        const int nSizes = int(nValues.size());
        param.n.resize(nSizes);
        for(int i = 0; i < nSizes; ++i){
            param.n[i] = nValues[i];
        }
    } catch(...) {
        ASD_LOG(INFO) << "n not set, use default";
    }
    try {
        std::vector<int32_t> ldaValues = opDescJson["specificParam"].at("lda");
        const int ldaSizes = int(ldaValues.size());
        param.lda.resize(ldaSizes);
        for(int i = 0; i < ldaSizes; ++i){
            param.lda[i] = ldaValues[i];
        }
    } catch(...) {
        ASD_LOG(INFO) << "lda not set, use default";
    }
    try {
        std::vector<int32_t> ldbValues = opDescJson["specificParam"].at("ldb");
        const int ldbSizes = int(ldbValues.size());
        param.ldb.resize(ldbSizes);
        for(int i = 0; i < ldbSizes; ++i){
            param.ldb[i] = ldbValues[i];
        }
    } catch(...) {
        ASD_LOG(INFO) << "ldb not set, use default";
    }
    try {
        std::vector<int32_t> ldcValues = opDescJson["specificParam"].at("ldc");
        const int ldcSizes = int(ldcValues.size());
        param.ldc.resize(ldcSizes);
        for(int i = 0; i < ldcSizes; ++i){
            param.ldc[i] = ldcValues[i];
        }
    } catch(...) {
        ASD_LOG(INFO) << "ldc not set, use default";
    }
    try {
        std::vector<int32_t> strideAValues = opDescJson["specificParam"].at("strideA");
        const int strideASizes = int(strideAValues.size());
        param.strideA.resize(strideASizes);
        for(int i = 0; i < strideASizes; ++i){
            param.strideA[i] = strideAValues[i];
        }
    } catch(...) {
        ASD_LOG(INFO) << "strideA not set, use default";
    }
    try {
        std::vector<int32_t> strideBValues = opDescJson["specificParam"].at("strideB");
        const int strideBSizes = int(strideBValues.size());
        param.strideB.resize(strideBSizes);
        for(int i = 0; i < strideBSizes; ++i){
            param.strideB[i] = strideBValues[i];
        }
    } catch(...) {
        ASD_LOG(INFO) << "strideB not set, use default";
    }
    try {
        std::vector<int32_t> strideCValues = opDescJson["specificParam"].at("strideC");
        const int strideCSizes = int(strideCValues.size());
        param.strideC.resize(strideCSizes);
        for(int i = 0; i < strideCSizes; ++i){
            param.strideC[i] = strideCValues[i];
        }
    } catch(...) {
        ASD_LOG(INFO) << "strideC not set, use default";
    }
    try {
        std::vector<int8_t> identityMValues = opDescJson["specificParam"].at("identityM");
        const int identityMSizes = int(identityMValues.size());
        param.identityM.resize(identityMSizes);
        for(int i = 0; i < identityMSizes; ++i){
            param.identityM[i] = identityMValues[i];
        }
    } catch(...) {
        ASD_LOG(INFO) << "identityM not set, use default";
    }
    try {
        param.maskType = static_cast<AsdOps::OpParam::Mix::MaskType>(
            opDescJson["specificParam"].at("maskType"));
    } catch (...) {
        ASD_LOG(INFO) << "maskType not set, use default";
    }
    try {
        param.isAlibiMaskSqrt = opDescJson["specificParam"].at("isAlibiMaskSqrt");
    } catch (...) {
        ASD_LOG(INFO) << "isAlibiMaskSqrt not set, use default";
    }
    try {
        param.alibiLeftAlign = opDescJson["specificParam"].at("alibiLeftAlign");
    } catch (...) {
        ASD_LOG(INFO) << "alibiLeftAlign not set, use default";
    }
    try {
        param.scale = opDescJson["specificParam"].at("scale");
    } catch (...) {
        ASD_LOG(INFO) << "scale not set, use default";
    }
    try {
        param.ioLayout = static_cast<AsdOps::OpParam::Mix::IoLayout>(
            opDescJson["specificParam"].at("ioLayout"));
    } catch (...) {
        ASD_LOG(INFO) << "ioLayout not set, use default";
    }
    try {
        param.keepProb = opDescJson["specificParam"].at("keepProb");
    } catch (...) {
        ASD_LOG(INFO) << "keepProb not set, use default";
    }
    try {
        param.preTokens = opDescJson["specificParam"].at("preTokens");
    } catch (...) {
        ASD_LOG(INFO) << "preTokens not set, use default";
    }
    try {
        param.nextTokens = opDescJson["specificParam"].at("nextTokens");
    } catch (...) {
        ASD_LOG(INFO) << "nextTokens not set, use default";
    }
    try {
        param.maxSeqLen = opDescJson["specificParam"].at("maxSeqLen");
    } catch (...) {
        ASD_LOG(INFO) << "maxSeqLen not set, use default";
    }
    try {
        std::vector<uint32_t> randSeedValues = opDescJson["specificParam"].at("randSeed");
        const int randSeedSizes = int(randSeedValues.size());
        param.randSeed.resize(randSeedSizes);
        for(int i = 0; i < randSeedSizes; ++i){
            param.randSeed[i] = randSeedValues[i];
        }
    } catch(...) {
        ASD_LOG(INFO) << "randSeed not set, use default";
    }
    try {
        param.compressHead = opDescJson["specificParam"].at("compressHead");
    } catch (...) {
        ASD_LOG(INFO) << "compressHead not set, use default";
    }
    opDesc.specificParam = param;
}

void SplitJson(const nlohmann::json &opDescJson, AsdOps::OpDesc &opDesc)
{
    AsdOps::OpParam::Split param;
    try {
        param.splitDim = opDescJson["specificParam"].at("splitDim");
    } catch (...) {
        ASD_LOG(INFO) << "splitDim not set, use default";
    }
    try {
        param.splitNum = opDescJson["specificParam"].at("splitNum");
    } catch (...) {
        ASD_LOG(INFO) << "splitNum not set, use default";
    }
    opDesc.specificParam = param;
}

using OpDescSetFunc = std::function<void(const nlohmann::json &, AsdOps::OpDesc &)>;

static const std::map<std::string, OpDescSetFunc> OP_DESC_JSON_FUNC_MAP = {
    {"OnehotOperation", OnehotJson},
    {"SliceOperation", SliceJson},
    {"ConcatOperation", ConcatJson},
    {"NormOperation", NormJson},
    {"CumsumOperation", CumsumJson},
    {"TransposeOperation", TransposeJson},
    {"FillOperation", FillJson},
    {"TransdataOperation", TransdataJson},
    {"SortOperation", SortJson},
    {"ElewiseOperation", ElewiseJson},
    {"ActivationOperation", ActivationJson},
    {"IndexOperation", IndexJson},
    {"AsStridedOperation", AsStridedJson},
    {"ReverseOperation", ReverseJson},
    {"GatherOperation", GatherJson},
    {"MatMulOperation", MatMulJson},
    {"CopyOperation", CopyJson},
    {"ReduceOperation", ReduceJson},
    {"ExpandOperation", ExpandJson},
    {"MultinomialOperation", MultinomialJson},
    {"SoftmaxOperation", SoftmaxJson},
    {"MixOperation", MixJson},
    {"SplitOperation", SplitJson},
};

void JsonToOpDesc(const nlohmann::json &opDescJson, AsdOps::OpDesc &opDesc)
{
    ASD_LOG(INFO) << opDescJson.dump(4);
    std::string opName = opDescJson["opName"];
    auto paramFunc = OP_DESC_JSON_FUNC_MAP.find(opName);
    if (paramFunc == OP_DESC_JSON_FUNC_MAP.end()) {
        ASD_LOG(ERROR) << "no opName " << opName;
        return;
    }
    try {
        paramFunc->second(opDescJson, opDesc);
    } catch(const std::exception &e) {
        ASD_LOG(ERROR) << "convert json fail, error:" << e.what();
    }
}
