/*
 * weight_project-inl.h
 *
 *  Created on: 2016年7月14日
 *      Author: jiangzhichao
 */

#ifndef WEIGHT_PROJECT_INL_H_
#define WEIGHT_PROJECT_INL_H_
#include "../operator/mshadow_op.h"
namespace mxnet {

	namespace op {
		namespace mshadow_op{
				struct binary_{
				  /*! \brief map a to result using defined operation */
				  template<typename DType>
				  MSHADOW_XINLINE static DType Map(DType a) {
					return a>(DType)0?(DType)1:(DType)-1;
				  }
				};

				struct fourbit{
								  /*! \brief map a to result using defined operation */
				  template<typename DType>
				  MSHADOW_XINLINE static DType Map(DType a) {
					  if((a>(DType)1/128 && a<(DType)1) || (a<(DType)-1/128 && a>(DType)-1))
						  return (DType)3*powf((DType)2,floorf(logf(fabsf(float(128*a)))/logf(2)))/256*(a>(DType)0?(DType)1:(DType)-1);
					  else if(a>(DType)0 && a<=(DType)1/128)
						  return (DType)1/256;
					  else if(a<(DType)0 && a>=(DType)(-1/128))
						  return (DType)(-1/256);
					  else if(a>=(DType)1)
						  return 1;
					  else if(a<=(DType)-1)
						  return -1;
					  else
						  return 0;
					  //1/16 step size
					//return ((std::abs((DType)128*a)>(DType)0 && std::abs((DType)128*a)<=(DType)1/128)) ? (a>(DType)0?(DType)1:(DType)-1)*1/16 :  (a>(DType)0?(DType)1:(DType)-1)*(1+2*std::ceil(std::log(std::abs((DType)128*a))))/(16);
				  }
				};
				struct twobit{
								  /*! \brief map a to result using defined operation */
				  template<typename DType>
				  MSHADOW_XINLINE static DType Map(DType x) {
					  if(x>(DType)0.5 )
						  return (DType)1;
					  else if(x<(DType)-0.5)
						  return (DType)(-1);
					  else
						  return (DType)0;

				  }
				};
				struct twobit_grad{
								  /*! \brief map a to result using defined operation */
				  template<typename DType>
				  MSHADOW_XINLINE static DType Map(DType a) {

				  }
				};
				struct binary{
				  /*! \brief map a to result using defined operation */
				  template<typename DType>
				  MSHADOW_XINLINE static DType Map(DType a) {
					return a>=(DType)0?(DType)1:(DType)-1;
				  }
				};
				struct binary_grad {
				  template<typename DType>
				  MSHADOW_XINLINE static DType Map(DType a) {
				    return DType((a >= DType(-1) && a<=DType(1)) ? DType(1): DType(0.0f));
				  }
				};

				struct trunc{
				  /*! \brief map a to result using defined operation */
				  template<typename DType>
				  MSHADOW_XINLINE static DType Map(DType a) {
					return (a>=DType(-1) && a<=DType(1))?(DType)1:(DType)0;
				  }
				};
				struct trunc_grad{
				  /*! \brief map a to result using defined operation */
				  template<typename DType>
				  MSHADOW_XINLINE static DType Map(DType a) {
					return (a>=DType(-1) && a<=DType(1))?(DType)1:(DType)0;
				  }
				};

				struct clip01{
				  /*! \brief map a to result using defined operation */
				  template<typename DType>
				  MSHADOW_XINLINE static DType Map(DType a) {
						  return (a>=DType(0) && a<=DType(1))?a:(a>DType(1)?DType(1):DType(0));
				  }
				};

				struct clip1{
				  /*! \brief map a to result using defined operation */
				  template<typename DType>
				  MSHADOW_XINLINE static DType Map(DType a) {
						  return (a>=DType(-1) && a<=DType(1))?a:(a>DType(1)?DType(1):DType(-1));
				  }
				};
				struct clip5{
				  /*! \brief map a to result using defined operation */
				  template<typename DType>
				  MSHADOW_XINLINE static DType Map(DType a) {
						  return (a>=DType(-5) && a<=DType(5))?a:(a>DType(5)?DType(5):DType(-5));
				  }
				};

				struct clip20{
				  /*! \brief map a to result using defined operation */
				  template<typename DType>
				  MSHADOW_XINLINE static DType Map(DType a) {
						  return (a>=DType(-20) && a<=DType(20))?a:(a>DType(20)?DType(20):DType(-20));
				  }
				};
				struct clip50{
				  /*! \brief map a to result using defined operation */
				  template<typename DType>
				  MSHADOW_XINLINE static DType Map(DType a) {
						  return (a>=DType(-50) && a<=DType(50))?a:(a>DType(50)?DType(50):DType(-50));
				  }
				};

				struct clip200{
				  /*! \brief map a to result using defined operation */
				  template<typename DType>
				  MSHADOW_XINLINE static DType Map(DType a) {
						  return (a>=DType(-200) && a<=DType(200))?a:(a>DType(200)?DType(200):DType(-200));
				  }
				};
		}
	}
}



#endif /* WEIGHT_PROJECT_INL_H_ */
