/*!
 * Copyright (c) 2015 by Contributors
 * \file convolution.cu
 * \brief
 * \author Bing Xu
*/

#include "./convolution-inl.h"
#include <vector>
#if MXNET_USE_CUDNN == 1
#include "./cudnn_convolution-inl.h"
#endif  // MXNET_USE_CUDNN

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<gpu>(ConvolutionParam param, int dtype,
                        std::vector<TShape> *in_shape,
                        std::vector<TShape> *out_shape,
                        Context ctx) {
  Operator *op = NULL;
#if MXNET_USE_CUDNN == 1
  if (param.dilate[0] == 1 && param.dilate[1] == 1) {
    MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
      op = new CuDNNConvolutionOp<DType>(param, in_shape, out_shape, ctx);
    })
  } else {
    MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
      op = new ConvolutionOp<gpu, DType>(param);
    })
  }
#else
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new ConvolutionOp<gpu, DType>(param);
  })
#endif  // MXNET_USE_CUDNN
  return op;
}



template<>
void mx_xpu_asum<mshadow::gpu,float>(mshadow::Stream<mshadow::gpu>* s,const int n, const float* x, float* y)
{
    cublasSasum(mshadow::Stream<mshadow::gpu>::GetBlasHandle(s),n,x,1,y);
}
template<>
void mx_xpu_asum<mshadow::gpu,double>(mshadow::Stream<mshadow::gpu>* s,const int n, const double* x, double* y)
{
	cublasDasum(mshadow::Stream<mshadow::gpu>::GetBlasHandle(s),n,x,1,y);
}
template<>
void mx_xpu_asum<mshadow::cpu,float>(mshadow::Stream<mshadow::cpu>* s,const int n, const float* x, float* y)
{
    *y=cblas_sasum(n,x,1);
}
template<>
void mx_xpu_asum<mshadow::cpu,double>(mshadow::Stream<mshadow::cpu>* s,const int n, const double* x, double* y)
{
	*y=cblas_dasum(n,x,1);
}
template<>
void mx_xpu_asum<mshadow::cpu,mshadow::half::half_t>(mshadow::Stream<mshadow::cpu>* s,const int n, const mshadow::half::half_t* x, mshadow::half::half_t* y){}
template<>
void mx_xpu_asum<mshadow::gpu,mshadow::half::half_t>(mshadow::Stream<mshadow::gpu>* s,const int n, const mshadow::half::half_t* x, mshadow::half::half_t* y){}

template <>
void mx_xpu_scal<gpu,double>(mshadow::Stream<mshadow::gpu>* s,const int N, const double alpha, double *X) {
  cublasDscal(mshadow::Stream<mshadow::gpu>::GetBlasHandle(s), N, &alpha, X, 1);
}
template <>
void mx_xpu_scal<gpu,float>(mshadow::Stream<mshadow::gpu>* s,const int N, const float alpha, float *X) {
  cublasSscal(mshadow::Stream<mshadow::gpu>::GetBlasHandle(s), N, &alpha, X, 1);
}
template <>
void mx_xpu_scal<cpu,double>(mshadow::Stream<mshadow::cpu>* s,const int N, const double alpha, double *X) {
  cblas_dscal(N, alpha, X, 1);
}
template <>
void mx_xpu_scal<cpu,float>(mshadow::Stream<mshadow::cpu>* s,const int N, const float alpha, float *X) {
  cblas_sscal(N, alpha, X, 1);
}
template<>
void mx_xpu_scal<mshadow::cpu,mshadow::half::half_t>(mshadow::Stream<mshadow::cpu>* s,const int N, const mshadow::half::half_t alpha, mshadow::half::half_t* X){}
template<>
void mx_xpu_scal<mshadow::gpu,mshadow::half::half_t>(mshadow::Stream<mshadow::gpu>* s,const int N, const mshadow::half::half_t alpha,  mshadow::half::half_t* X){}

}  // namespace op
}  // namespace mxnet

