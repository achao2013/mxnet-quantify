/*!
 * Copyright (c) 2015 by Contributors
 * \file activation.cu
 * \brief
 * \author Bing Xu
*/
#include "./activation-inl.h"
#include "./mshadow_op.h"
#if MXNET_USE_CUDNN == 1
#include "./cudnn_activation-inl.h"
#endif

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<gpu>(ActivationParam param, int dtype) {
  Operator *op = NULL;
  // SoftReLU not supported by CUDNN yet
  if (param.act_type == activation::kSoftReLU) {
    MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
      op = new ActivationOp<gpu, mshadow_op::softrelu, mshadow_op::softrelu_grad, DType>();
    })
    op->set_q_method(param.q_method);//add by jzc
	op->set_act_type(param.act_type);
    return op;
  }
  else if(param.act_type == activation::kSign){
	MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
	  op = new ActivationOp<gpu, mshadow_op::binary, mshadow_op::binary_grad, DType>();
	})
	op->set_q_method(param.q_method);//add by jzc
	op->set_act_type(param.act_type);
	return op;
  }
#if MXNET_USE_CUDNN == 1
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new CuDNNActivationOp<DType>(param);
  })
#else
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    switch (param.act_type) {
      case activation::kReLU:
        op = new ActivationOp<gpu, mshadow_op::relu, mshadow_op::relu_grad, DType>();
        break;
      case activation::kSigmoid:
        op = new ActivationOp<gpu, mshadow_op::sigmoid, mshadow_op::sigmoid_grad, DType>();
        break;
      case activation::kTanh:
        op = new ActivationOp<gpu, mshadow_op::tanh, mshadow_op::tanh_grad, DType>();
        break;
      default:
        LOG(FATAL) << "unknown activation";
    }
  })
  op->set_act_type(param.act_type);//add by jzc
#endif  // MXNET_USE_CUDNN
  op->set_q_method(param.q_method);//add by jzc

  return op;
}
}  // namespace op
}  // namespace mxnet

