#include <mki/base/operation_base.h>
#include <mki/utils/log/log.h>
#include <mki/utils/const/op_const.h>
#include <mki_loader/op_register.h>

namespace SimpleOps {
using namespace Mki;
class KVCacheOperation : public OperationBase {
public:
    explicit KVCacheOperation(const std::string &opName) noexcept : OperationBase(opName) {}

    int64_t GetInputNum(const Any &specificParam) const override
    {
        (void)specificParam;
        return 5;
    }

    int64_t GetOutputNum(const Any &specificParam) const override
    {
        (void)specificParam;
        return 1;
    }

    bool CheckKVCache(const LaunchParam &launchParam) const
    {
        auto newKV = launchParam.GetInTensor(DIM_0);
        bool nzFlag = (newKV.desc.format == TENSOR_FORMAT_FRACTAL_NZ);
        TensorDType dtype = newKV.desc.dtype;
        MKI_CHECK(dtype == TENSOR_DTYPE_FLOAT16 || dtype == TENSOR_DTYPE_BF16 || dtype == TENSOR_DTYPE_INT8,
            "input new kv data type error", return false);
        MKI_CHECK(newKV.desc.format == TENSOR_FORMAT_ND || nzFlag, "input new kv data format error", return false);
        MKI_CHECK(newKV.desc.dims.size() == (nzFlag ? DIM_4 : DIM_2), "input new kv data dims error: " <<
            newKV.desc.dims.size(), return false);
        auto layerId = launchParam.GetInTensor(DIM_1);
        MKI_CHECK(layerId.desc.format == TENSOR_FORMAT_ND, "input layer id data format error", return false);
        MKI_CHECK(layerId.desc.dims.size() == DIM_1, "input layer id data dims error", return false);

        auto cacheIn = launchParam.GetInTensor(DIM_2);
        nzFlag = (nzFlag && (cacheIn.desc.format == TENSOR_FORMAT_FRACTAL_NZ));
        bool dyncBatchFlag = (cacheIn.desc.dims.size() == DIM_3);
        MKI_CHECK(cacheIn.desc.dtype == dtype, "cacheIn and newKV are of different dtype", return false);
        MKI_CHECK(cacheIn.desc.format == TENSOR_FORMAT_ND || nzFlag, "input cache data format error", return false);
        MKI_CHECK(cacheIn.desc.dims.size() == (nzFlag ? DIM_5 : (dyncBatchFlag ? DIM_3 : DIM_4)),
            "input cache data dims error", return false);

        auto tokenOffset = launchParam.GetInTensor(DIM_3);
        MKI_CHECK(tokenOffset.desc.format == TENSOR_FORMAT_ND, "token offset data format error", return false);
        MKI_CHECK(tokenOffset.desc.dims.size() == DIM_1,  "input token offset data dims error", return false);

        auto seqLen = launchParam.GetInTensor(DIM_4);
        MKI_CHECK(seqLen.desc.format == TENSOR_FORMAT_ND, "input seq len data format error", return false);
        MKI_CHECK(seqLen.desc.dims.size() == DIM_1, "input seq len data dims error", return false);

        MKI_CHECK((nzFlag ? (newKV.desc.dims[DIM_1] == cacheIn.desc.dims[DIM_2]) :
            (newKV.desc.dims[DIM_1] == (dyncBatchFlag ? cacheIn.desc.dims[DIM_2] : cacheIn.desc.dims[DIM_3]))),
            "hiddenSize in newKv and cacheIn are not equal", return false);
        MKI_CHECK(seqLen.desc.dims[DIM_0] == tokenOffset.desc.dims[DIM_0],
            "batch in seq len and token are not equal", return false);
        MKI_CHECK(dyncBatchFlag ? (cacheIn.desc.dims[DIM_0] >= tokenOffset.desc.dims[DIM_0]) :
                                     (cacheIn.desc.dims[DIM_1] == tokenOffset.desc.dims[DIM_0]),
            "batch in cache should be greater or equal to in seq len and token", return false);

        return true;
    }

    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        MKI_CHECK(CheckKVCache(launchParam), "Failed to check launch param",
            return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "Failed to check launch param"));
        auto &tensorcacheIn = launchParam.GetInTensor(DIM_2);
        outTensors[DIM_0].desc = tensorcacheIn.desc;

        return Status::OkStatus();
    }

    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        return GetKernelByName("KVCacheKernel");
    }
};

REG_OPERATION(KVCacheOperation);
} // namespace SimpleOps
