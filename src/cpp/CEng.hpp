//
// Created by egrzrbr on 2019-04-06.
//

#ifndef CPP_EXAMPLE_CENG_HPP
#define CPP_EXAMPLE_CENG_HPP

namespace CEng {

	template<typename T, typename P, size_t mB, size_t N, size_t M>
	void mat_vec_mul_simple(std::array<T, M * N> W, std::array<P, N * mB> V, std::array<T, M * mB> &b);


	template<typename T, size_t M, size_t N>
	void matvec_mul_simple_crefs(const std::array<T, M * N> &mat, const std::array<T, N> &vec,
	                             std::array<T, M> &out);


	template<typename T, size_t M, size_t N>
	void mat_vec_mul_stl(const std::array<T, M * N> &mat, const std::array<T, N> &vec,
	                     std::array<T, M> &out);



	class Engine {




	};

}

#include "CEng.hxx"

#endif //CPP_EXAMPLE_CENG_HPP
