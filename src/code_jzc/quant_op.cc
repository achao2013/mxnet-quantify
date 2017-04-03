#include "quant_op.h"
#include <iostream>
namespace mxnet {
namespace op {
		template<>
		void print_check<cpu>(float * data, std::string s)
		{
			LOG(INFO)<<s<<": "<<data[0]<<" "<<data[1]<<" "<<data[2]<<" "<<data[3]<<" "<<data[4]<<" "<<data[5]<<" "<<data[6];
		}

		template<>
		void print_check<cpu>(double * data, std::string s)
		{
			LOG(INFO)<<s<<": "<<data[0]<<" "<<data[1]<<" "<<data[2]<<" "<<data[3]<<" "<<data[4]<<" "<<data[5]<<" "<<data[6];
		}

		template<>
		void print_check<cpu>(mshadow::half::half_t * data, std::string s)
		{
			LOG(INFO)<<s<<": "<<data[0]<<" "<<data[1]<<" "<<data[2]<<" "<<data[3]<<" "<<data[4]<<" "<<data[5]<<" "<<data[6];
		}

		template<>
		void get_iter<cpu>(float* data, float& cpu_i)
		{
			cpu_i=*data;
		}
		template<>
		void get_iter<cpu>(double* data,double& cpu_i)
		{
			cpu_i=*data;
		}
		template<>
		void get_iter<cpu>(mshadow::half::half_t* data, mshadow::half::half_t& cpu_i)
		{
			cpu_i=*data;
		}
	}
}
