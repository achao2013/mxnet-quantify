#ifndef QUANT_OP_H
#define QUANT_OP_H
#include "../operator/operator_common.h"

namespace mxnet {
namespace op {

		template<typename DType>
		__global__ void print_kernal(DType* data);

		template<typename xpu,typename DType>
		void print_check(DType * data, std::string s);

		template<typename xpu,typename DType>
		void get_iter(DType* data,DType& cpu_i);

	}
}

#endif
