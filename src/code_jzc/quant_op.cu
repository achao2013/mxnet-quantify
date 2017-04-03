#include "quant_op.h"


namespace mxnet {
namespace op {
		template<>
		__global__ void print_kernal(float* data)
		{
			printf(" %f %f %f %f %f %f %f\n",data[0],data[1],data[2],data[3],data[4],data[5],data[6]);
		}

		template<>
		__global__ void print_kernal(mshadow::half::half_t* data)
		{
			printf(" %f %f %f %f %f %f %f\n",data[0],data[1],data[2],data[3],data[4],data[5],data[6]);
		}

		template<>
		__global__ void print_kernal(double* data)
		{
			printf(" %lf %lf %lf %lf %lf %lf %lf\n",data[0],data[1],data[2],data[3],data[4],data[5],data[6]);
		}

		template<>
		void print_check<gpu>(float * data,std::string s)
		{
			const char* str=s.c_str();
			printf("%s",str);
			print_kernal<<< 1,1 >>>(data);
		}

		template<>
		void print_check<gpu>(double * data,std::string s)
		{
			const char* str=s.c_str();
			printf("%s",str);
			print_kernal<<< 1,1 >>>(data);
		}

		template<>
		void print_check<gpu>(mshadow::half::half_t * data,std::string s)
		{
			const char* str=s.c_str();
			printf("%s",str);
			print_kernal<<< 1,1 >>>(data);
		}

		template<>
		void get_iter<gpu>(float* data, float& cpu_i)
		{
			cudaMemcpy(&cpu_i,data,sizeof(float),cudaMemcpyDeviceToHost);

		}

		template<>
		void get_iter<gpu>(double* data, double& cpu_i)
		{
			cudaMemcpy(&cpu_i,data,sizeof(double),cudaMemcpyDeviceToHost);

		}

		template<>
		void get_iter<gpu>(mshadow::half::half_t* data, mshadow::half::half_t& cpu_i)
		{
			cudaMemcpy(&cpu_i,data,sizeof(mshadow::half::half_t),cudaMemcpyDeviceToHost);

		}
	}
}
