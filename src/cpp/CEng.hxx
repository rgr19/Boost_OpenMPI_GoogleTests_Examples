//
// Created by egrzrbr on 2019-04-06.
//


namespace CEng {

	template<typename T, typename P, size_t mB, size_t N, size_t M>
	void mat_vec_mul_simple(std::array<T, M * N> W, std::array<P, N * mB> V, std::array<T, M * mB> &b) {

		for (int k = 0; k < mB; ++k) {
			for (int i = 0; i < M; ++i) {
				T sum = 0;
				for (int j = 0; j < N; ++j) {
					sum += W[N * i + j] * V[mB * j + k];
				}
				b[mB * i + k] = sum;
			}
		}

		for (auto a : b) {
			std::cout << a << std::endl;
		}

	}


	template<typename T, size_t M, size_t N>
	void matvec_mul_simple_crefs(const std::array<T, M * N> &mat, const std::array<T, N> &vec,
	                             std::array<T, M> &out) {

		for (int i = 0; i < M; ++i) {
			T sum = 0;
			for (int j = 0; j < N; ++j) {
				sum += mat[N * i + j] * vec[j];
			}
			out[i] = sum;
		}
	}


	template<typename T, size_t M, size_t N>
	void mat_vec_mul_stl(const std::array<T, M * N> &mat, const std::array<T, N> &vec,
	                     std::array<T, M> &out) {

		for (int i = 0; i < M; ++i) {
			auto sum = std::transform(mat.cbegin() + N * i, mat.cbegin() + N * i + N, vec.cbegin(), out.cbegin(),
			                          [](auto lhs, auto rhs) {
				                          std::cout << lhs << ", " << rhs << std::endl;
				                          return 1;
			                          }
			);
			std::cout << sum << std::endl;
		}

	}

template<typename Iterator>
typename std::iterator_traits<Iterator>::value_type
sum(Iterator begin, Iterator end) {

	using value_type = typename std::iterator_traits<Iterator>::value_type;
	value_type s = value_type();
	for (Iterator it = begin; it != end; it++) {
		s += *it;
	}
	return s;
}

}

