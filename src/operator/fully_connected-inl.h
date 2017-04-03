/*!
 * Copyright (c) 2015 by Contributors
 * \file fully_connect_op-inl.h
 * \brief fully connect operator and symbol
*/
#ifndef MXNET_OPERATOR_FULLY_CONNECTED_INL_H_
#define MXNET_OPERATOR_FULLY_CONNECTED_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./operator_common.h"
#include "../code_jzc/weight_project-inl.h"

namespace mxnet {
namespace op {
template<typename xpu,typename DType>
void mx_xpu_asum(mshadow::Stream<xpu>* s,const int n, const DType* x, DType* y);
template <typename xpu,typename DType>
void mx_xpu_scal(mshadow::Stream<xpu>* s,const int N, const DType alpha, DType *X) ;
// Declare enumeration of input order to make code more intuitive.
// These enums are only visible within this header
namespace fullc {
enum FullyConnectedOpInputs {kData, kWeight, kBias};
enum FullyConnectedOpOutputs {kOut};
//add by jzc
enum FullyConnectedOpProjectWeights {None,Binary,SimpleScaleBinary,Sign,Round,Power,Stoch,StochM,AddMorm,MultUnif,FourBit};
}  // fullc

struct FullyConnectedParam : public dmlc::Parameter<FullyConnectedParam> {
  int num_hidden;
  bool no_bias;
  //edit by jzc
  //typedef fullc::FullyConnectedOpProjectWeights OpProjectWeights;
  int w_project_method;
  DMLC_DECLARE_PARAMETER(FullyConnectedParam) {
    // TODO(bing) add support for boolean
    DMLC_DECLARE_FIELD(num_hidden).set_lower_bound(1)
    .describe("Number of hidden nodes of the output.");
    DMLC_DECLARE_FIELD(no_bias).set_default(false)
    .describe("Whether to disable bias parameter.");
    DMLC_DECLARE_FIELD(w_project_method).set_default(fullc::None)
    .describe("weight project method.");
  }
};

/**
 * \brief This is the implementation of fully connected operator.
 * \tparam xpu The device that the op will be executed on.
 */
template<typename xpu, typename DType>
class FullyConnectedOp : public Operator {
 public:
  explicit FullyConnectedOp(FullyConnectedParam p) {
    this->param_ = p;
  }
  //add by jzc
  virtual void init_projection(TBlob& tmp_w)
  {
	  //LOG(INFO)<<"init_projection in FullyConnectedOp";
	  projected_weight = tmp_w;
	  //LOG(INFO)<<"FullyConnected w_project_method:"<<get_project_method();
  }
  virtual void modify_project_method(int i)
  {
	  param_.w_project_method=i;
  }
  virtual int get_project_method()
  {
	  return param_.w_project_method;
  }
  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    if (req[fullc::kOut] == kNullOp) return;
    CHECK_EQ(req[fullc::kOut], kWriteTo);
    size_t expected = param_.no_bias ? 2 : 3;
    CHECK_EQ(in_data.size(), expected);
    CHECK_EQ(out_data.size(), 1);
    // TODO(bing): check the BLAS Handle, be careful
    // maybe need blas handle from context
    // TODO(bing): judge shape to remove flatten op
    Stream<xpu> *s = ctx.get_stream<xpu>();
#if defined(__CUDACC__)
    CHECK_EQ(s->blas_handle_ownership_, Stream<xpu>::OwnHandle)
        << "Must init CuBLAS handle in stream";
#endif  // __CUDACC__
    const TShape& ishape = in_data[fullc::kData].shape_;
    const TShape& oshape = out_data[fullc::kOut].shape_;

    Tensor<xpu, 2, DType> data = in_data[fullc::kData].get_with_shape<xpu, 2, DType>(
        Shape2(ishape[0], ishape.ProdShape(1, ishape.ndim())), s);
    Tensor<xpu, 2, DType> wmat = in_data[fullc::kWeight].get<xpu, 2, DType>(s);
    Tensor<xpu, 2, DType> out = out_data[fullc::kOut].get_with_shape<xpu, 2, DType>(
        Shape2(oshape[0], oshape.ProdShape(1, oshape.ndim())), s);
    //add by jzc
    Tensor<xpu, 2, DType> pwmat = projected_weight.get<xpu, 2, DType>(s);
    //Tensor<xpu, 2, DType> pwmat=mshadow::NewTensor<xpu,DType,2>(wmat.shape_,DType(0),false,s);
    DType tmp;
    switch(this->param_.w_project_method)
    {
    case fullc::None:
    	out = dot(data, wmat.T());//original
    	break;
    case fullc::Binary:
    	//pwmat=F<mshadow_op::binary>(wmat);
    	Assign(pwmat,kWriteTo,F<mshadow_op::binary>(wmat));
    	//LOG(INFO)<<"fc forward of binary";
    	out = dot(data, pwmat.T());
    	break;
    case fullc::SimpleScaleBinary:
		mx_xpu_asum<xpu,DType>(s, wmat.shape_.Size(), wmat.dptr_, &tmp);
		tmp=tmp/wmat.shape_.Size();
		pwmat=F<mshadow_op::binary>(wmat);
		mx_xpu_scal<xpu,DType>(s,pwmat.shape_.Size(),tmp,pwmat.dptr_);
    	out = dot(data, pwmat.T());
    	break;
    case fullc::Sign:
    	break;
    case fullc::FourBit:
    	//pwmat=F<mshadow_op::fourbit>(wmat);
    	Assign(pwmat,kWriteTo,F<mshadow_op::fourbit>(wmat));
    	//LOG(INFO)<<"fc forward of fourbit";
    	out = dot(data, pwmat.T());
    	break;
    }

    if (!param_.no_bias) {
      Tensor<xpu, 1, DType> bias = in_data[fullc::kBias].get<xpu, 1, DType>(s);
      out += repmat(bias, data.size(0));
    }
    //FreeSpace<2,DType>(&pwmat);
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
    size_t expected = param_.no_bias ? 2 : 3;
    CHECK(in_data.size() == expected && in_grad.size() == expected);
    CHECK_EQ(req.size(), expected);
    // TODO(bing): check the BLAS Handle, be careful
    //  maybe need blas handle from context
    Stream<xpu> *s = ctx.get_stream<xpu>();
    const TShape& ishape = in_data[fullc::kData].shape_;
    const TShape& oshape = out_grad[fullc::kOut].shape_;

    Tensor<xpu, 2, DType> data = in_data[fullc::kData].get_with_shape<xpu, 2, DType>(
        Shape2(ishape[0], ishape.ProdShape(1, ishape.ndim())), s);
    Tensor<xpu, 2, DType> wmat = in_data[fullc::kWeight].get<xpu, 2, DType>(s);
    Tensor<xpu, 2, DType> grad = out_grad[fullc::kOut].get_with_shape<xpu, 2, DType>(
        Shape2(oshape[0], oshape.ProdShape(1, oshape.ndim())), s);

#if defined(__CUDACC__)
    CHECK_EQ(s->blas_handle_ownership_, Stream<xpu>::OwnHandle)
        << "Must init CuBLAS handle in stream";
#endif
    //  backprop
    CHECK_NE(req[fullc::kWeight], kWriteInplace) << "cannot write weight inplace";
    // gradient of weight
    Tensor<xpu, 2, DType> gwmat = in_grad[fullc::kWeight].get<xpu, 2, DType>(s);
    //edit by jzc
    /*switch(this->param_.w_project_method)
        {
        case fullc::None:
        	Assign(gwmat,req[fullc::kWeight],dot(grad.T(), data));
        	break;
        case fullc::Binary:
        	{
				Tensor<xpu, 2, DType> tmp_gwmat=mshadow::NewTensor(gwmat.shape_,DType(0),true,s);
				Assign(tmp_gwmat,kWriteTo,dot(grad.T(), data));
				Assign(gwmat, req[fullc::kWeight], F<mshadow_op::trunc>(wmat)*tmp_gwmat);
				FreeSpace<2,DType>(&tmp_gwmat);
            }
        	break;
        }
*/
    Assign(gwmat,req[fullc::kWeight],dot(grad.T(), data));

    switch(this->param_.w_project_method)
    {
    case fullc::None:
  	  break;
    case fullc::Binary:
    	Assign(gwmat, kWriteInplace, F<mshadow_op::binary_grad>(wmat)*gwmat);
  	  //Assign(gwmat, kWriteInplace, F<mshadow_op::clip50>(gwmat));
  	  break;
    }
    // gradient of bias
    if (!param_.no_bias) {
      Tensor<xpu, 1, DType> gbias = in_grad[fullc::kBias].get<xpu, 1, DType>(s);
      Assign(gbias, req[fullc::kBias], sum_rows(grad));
    }
    // gradient of data
    Tensor<xpu, 2, DType> gdata = in_grad[fullc::kData].get_with_shape<xpu, 2, DType>(
        Shape2(ishape[0], ishape.ProdShape(1, ishape.ndim())), s);
    //add by jzc
    Tensor<xpu, 2, DType> pwmat = projected_weight.get<xpu, 2, DType>(s);
    //Tensor<xpu, 2, DType> pwmat=mshadow::NewTensor<xpu,DType,2>(wmat.shape_,DType(0),false,s);
    switch(this->param_.w_project_method)
        {
        case fullc::None:
        	Assign(gdata, req[fullc::kData], dot(grad, wmat));//original
        	break;
        case fullc::Binary:
        	//pwmat=F<mshadow_op::binary>(wmat);
        	Assign(pwmat,kWriteTo,F<mshadow_op::binary>(wmat));
        	//LOG(INFO)<<"fc backward of binary";
        	Assign(gdata, req[fullc::kData], dot(grad, pwmat));
        	break;
        case fullc::Sign:
        	break;
        case fullc::FourBit:
        	Assign(pwmat,kWriteTo,F<mshadow_op::fourbit>(wmat));
        	//LOG(INFO)<<"fc backward of fourbit";
        	Assign(gdata, req[fullc::kData], dot(grad, pwmat));
        	break;
        }
    //FreeSpace<2,DType>(&pwmat);
  }

 private:
  FullyConnectedParam param_;
  //add by jzc
  TBlob projected_weight;
};  // class FullyConnectedOp

// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(FullyConnectedParam param, int dtype);

#if DMLC_USE_CXX11
class FullyConnectedProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    if (!param_.no_bias) {
      return {"data", "weight", "bias"};
    } else {
      return {"data", "weight"};
    }
  }

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
    if (!param_.no_bias) {
      CHECK_EQ(in_shape->size(), 3) << "Input:[data, weight, bias]";
    } else {
      CHECK_EQ(in_shape->size(), 2) << "Input:[data, weight]";
    }
    const TShape &dshape = (*in_shape)[fullc::kData];
    // require data to be known
    if (dshape.ndim() ==  0) return false;

    index_t num_input = dshape.ProdShape(1, dshape.ndim());
    SHAPE_ASSIGN_CHECK(*in_shape, fullc::kWeight, Shape2(param_.num_hidden, num_input));
    if (!param_.no_bias) {
      SHAPE_ASSIGN_CHECK(*in_shape, fullc::kBias, Shape1(param_.num_hidden));
    }
    out_shape->clear();
    out_shape->push_back(Shape2(dshape[0], param_.num_hidden));
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
    return true;
  }

  OperatorProperty* Copy() const override {
    FullyConnectedProp* fc_sym = new FullyConnectedProp();
    fc_sym->param_ = this->param_;
    return fc_sym;
  }

  std::string TypeString() const override {
    return "FullyConnected";
  }

  // decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[fullc::kOut], in_data[fullc::kData], in_data[fullc::kWeight]};
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {{in_data[fullc::kData], in_grad[fullc::kData]}};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  FullyConnectedParam param_;
};  // class FullyConnectedSymbol
#endif
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_FULLY_CONNECTED_INL_H_
