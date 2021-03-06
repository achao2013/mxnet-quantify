/*!
 * Copyright (c) 2015 by Contributors
 * \file activation-inl.h
 * \brief Activation operator
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_ACTIVATION_INL_H_
#define MXNET_OPERATOR_ACTIVATION_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <utility>
#include "./operator_common.h"
#include "../code_jzc/weight_project-inl.h"
#include "../code_jzc/quant_op.h"
namespace mxnet {
namespace op {
// Declare enumeration of input order to make code more intuitive.
// // These enums are only visible within this header
namespace activation {
enum ActivationOpInputs {kData};
enum ActivationOpOutputs {kOut};
enum ActivationOpType {kReLU, kSigmoid, kTanh, kSoftReLU,kSign};
enum ActivationOpQuantify  {None,Binary,TwoBit,FourBit};
}  // activation

struct ActivationParam : public dmlc::Parameter<ActivationParam> {
  // use int for enumeration
  int act_type;
  int q_method;
  DMLC_DECLARE_PARAMETER(ActivationParam) {
    DMLC_DECLARE_FIELD(act_type)
    .add_enum("relu", activation::kReLU)
    .add_enum("sigmoid", activation::kSigmoid)
    .add_enum("tanh", activation::kTanh)
    .add_enum("softrelu", activation::kSoftReLU)
    .add_enum("sign", activation::kSign)
    .describe("Activation function to be applied.");
    DMLC_DECLARE_FIELD(q_method).set_default(activation::None)
        .describe("Quantify method.");
  }
};

/**
 * \brief This is the implementation of activation operator.
 * \tparam xpu The device that the op will be executed on.
 */
template<typename xpu, typename ForwardOp, typename BackwardOp, typename DType>
class ActivationOp : public Operator {
 public:
	int q_method;
	int act_type;
	  virtual void set_q_method(int i){this->q_method=i;}
	  virtual void set_act_type(int i){this->act_type=i;}
  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 1);
    CHECK_EQ(out_data.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2, DType> data = in_data[activation::kData].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> out = out_data[activation::kOut].FlatTo2D<xpu, DType>(s);
    switch(this->q_method)
    {
      case activation::None:
    	  Assign(out, req[activation::kOut], F<ForwardOp>(data));
    	  //LOG(INFO)<<"activation quantify of none";
    	  break;
      case activation::TwoBit:
    	  Assign(out, req[activation::kOut], F<mshadow_op::twobit>(F<ForwardOp>(data)));
    	  break;
      case activation::FourBit:
          Assign(out, req[activation::kOut], F<mshadow_op::fourbit>(F<ForwardOp>(data)));
          //LOG(INFO)<<"activation quantify of fourbit";
          break;
    }

  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(out_grad.size(), 1);
    CHECK(in_data.size() == 1 && in_grad.size() == 1);
    CHECK_EQ(req.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2, DType> m_out_grad = out_grad[activation::kOut].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> m_out_data = out_data[activation::kOut].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> m_in_grad = in_grad[activation::kData].FlatTo2D<xpu, DType>(s);

    //edit by jzc
    Tensor<xpu, 2, DType> m_in_data = in_data[activation::kData].FlatTo2D<xpu, DType>(s);
    if(act_type==activation::kSign)
    {
    	Assign(m_in_grad, req[activation::kData],  F<BackwardOp>(m_in_data)*m_out_grad);
    	/*Tensor<xpu, 2,DType> iter = aux_args[0].get<xpu, 2, DType>(s);
    	DType cpu_i=DType(0);
    	get_iter<xpu>(iter.dptr_,cpu_i);

    	if(int(cpu_i)%1000==0)
    	{
    		//LOG(INFO)<<"backward sign nbatch: "<<cpu_i;
			print_check<xpu>(m_in_data.dptr_,"m_in_data");
			cudaDeviceSynchronize();
			print_check<xpu>(m_out_grad.dptr_,"m_out_grad");
			cudaDeviceSynchronize();
    	}
*/
    }
    else
    {
        Assign(m_in_grad, req[activation::kData], F<BackwardOp>(m_out_data) * m_out_grad);//original

    }
  }
};  // class ActivationOp

// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(ActivationParam type, int dtype);

#if DMLC_USE_CXX11
class ActivationProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 1) << "Input:[data]";
    const TShape &dshape = in_shape->at(activation::kData);
    if (dshape.ndim() == 0) return false;
    out_shape->clear();
    out_shape->push_back(dshape);
    aux_shape->clear();
    aux_shape->push_back(Shape2(1,1));
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_GE(in_type->size(), 1);
    int dtype = (*in_type)[0];
    CHECK_NE(dtype, -1) << "First input must have specified type";
    for (index_t i = 0; i < in_type->size(); ++i) {
      if ((*in_type)[i] == -1) {
          (*in_type)[i] = dtype;
      } else {
        CHECK_EQ((*in_type)[i], dtype) << "This layer requires uniform type. "
                                       << "Expected " << dtype << " v.s. given "
                                       << (*in_type)[i] << " at " << ListArguments()[i];
      }
    }
    out_type->clear();
    out_type->push_back(dtype);
    aux_type->clear();
    aux_type->push_back(dtype);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new ActivationProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "Activation";
  }

  // decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
#if MXNET_USE_CUDNN == 1
    return {out_grad[activation::kOut], out_data[activation::kOut], in_data[activation::kData]};
#else
    return {out_grad[activation::kOut], out_data[activation::kOut], in_data[activation::kData]};//edit by jzc
#endif  // MXNET_USE_CUDNN
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {{out_grad[activation::kOut], in_grad[activation::kData]}};
  }

  std::vector<std::pair<int, void*> > ForwardInplaceOption(
    const std::vector<int> &in_data,
    const std::vector<void*> &out_data) const override {
    return {{in_data[activation::kData], out_data[activation::kOut]}};
  }
  //add by jzc
  std::vector<std::string> ListAuxiliaryStates() const override {
    return {"nbatch"};
  }
  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  ActivationParam param_;
};
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_ACTIVATION_INL_H_
