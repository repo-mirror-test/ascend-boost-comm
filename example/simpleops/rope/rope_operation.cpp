#include <mki/base/operation_base.h>
#include <mki/utils/log/log.h>
#include <mki/utils/const/op_const.h>
#include <mki_loader/op_register.h>

namespace SimpleOps {
using namespace Mki;
class RopeOperation : public OperationBase {
public:
    explicit RopeOperation(const std::string &opName) noexcept : OperationBase(opName) {}

    int64_t GetInputNum(const Any &specificParam) const override
    {
        (void)specificParam;
        return 5;
    }

    int64_t GetOutputNum(const Any &specificParam) const override
    {
        (void)specificParam;
        return 2;
    }

    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        MKI_LOG(INFO) << "RopeInferShape enter";
        const SVector<int64_t> &inputQ = launchParam.GetInTensor(DIM_0).desc.dims;
        const SVector<int64_t> &inputK = launchParam.GetInTensor(DIM_1).desc.dims;
        const SVector<int64_t> &inputCos = launchParam.GetInTensor(DIM_2).desc.dims;
        const SVector<int64_t> &inputSin = launchParam.GetInTensor(DIM_3).desc.dims;
        if (inputQ.size() != inputK.size() || inputQ.size() == 0) {
            return Status::FailStatus(ERROR_INVALID_VALUE, "dim size of inputQ is wrong");
        }
        for (size_t i = 0; i < inputQ.size() - 1; i++) {
            if (inputQ[i] != inputK[i]) {
                return Status::FailStatus(ERROR_INVALID_VALUE, "inputQ are not equal to inputK");
            }
        }
        if (inputCos != inputSin) {
            return Status::FailStatus(ERROR_INVALID_VALUE, "inputCos are not equal to inputSin");
        }
        outTensors[DIM_0].desc = launchParam.GetInTensor(DIM_0).desc;
        outTensors[DIM_1].desc = launchParam.GetInTensor(DIM_1).desc;

        return Status::OkStatus();
    }

    Kernel *GetBestKernel(const LaunchParam &launchParam) const override
    {
        return GetKernelByName("RopeKernel");
    }
};

REG_OPERATION(RopeOperation);
} // namespace SimpleOps
