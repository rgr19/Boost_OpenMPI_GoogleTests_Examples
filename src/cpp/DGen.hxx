//
// Created by egrzrbr on 2019-04-06.
//
#ifndef CPP_EXAMPLE_DGEN_HXX
#define CPP_EXAMPLE_DGEN_HXX


#include "DGen.hpp"


namespace DGen {
	// random number generator from Stroustrup:
	// http://www.stroustrup.com/C++11FAQ.html#std-random
	static std::default_random_engine randEng;
	using DIST_UI = std::uniform_int_distribution<int>;
	using DIST_UR = std::uniform_real_distribution<float>;
	using DIST_RN = std::normal_distribution<float>;

	inline const auto rand_uniform_int(const int low, const int high) {
		static DIST_UI distUI{};
		return distUI(randEng, DIST_UI::param_type{low, high});
	}

	inline const auto rand_uniform_real(const float low, const float high) {
		static DIST_UR distUR{};
		return distUR(randEng, DIST_UR::param_type{low, high});
	}

	inline const auto rand_normal(const float mean, const float stdev) {
		static DIST_RN distRN{};
		return distRN(randEng, DIST_RN::param_type{mean, stdev});
	}

	template<typename T, gen_e GEN_E>
	template<typename std::enable_if_t<GEN_E == RANDOM_NORMAL> *>
	constexpr inline T Datagen<T, GEN_E>::operator()(const T mean, const T stdev) {
		return static_cast<T>(rand_normal(mean, stdev));
	}

	template<typename T, gen_e GEN_E>
	template<typename std::enable_if_t<GEN_E == RANDOM_UNIFORM> *>
	constexpr inline T Datagen<T, GEN_E>::operator()(const T min, const T max) {
		if constexpr (std::is_integral_v<T>) {
			return static_cast<T>(rand_uniform_int(min, max));
		} else {
			return static_cast<T>(rand_uniform_real(min, max));
		}
	}

	template<typename T, gen_e GEN_E>
	template<class Iterator, typename std::enable_if_t<GEN_E < DATA_GEN_SWITCH> *>
	inline auto Datagen<T, GEN_E>::generate(Iterator begin, Iterator end, const T a, const T b) {

		std::generate(begin, end, [&] {
			return operator()(a, b);
		});

		return *this;
	}

	template<typename T, gen_e GEN_E>
	template<class Iterator, typename std::enable_if_t<GEN_E == RANGE> *>
	inline auto Datagen<T, GEN_E>::generate(Iterator begin, Iterator end, const T start) {

		std::iota(begin, end, start);
		return *this;
	}

	template<typename T, gen_e GEN_E>
	template<class Iterator, typename std::enable_if_t<GEN_E == RANDOM_NORMAL> *>
	inline auto Datagen<T, GEN_E>::validate(Iterator begin, Iterator end, const T mean_, const T stdev_) {

		T min = 1e6;
		T max = 0;

		size_t n = std::distance(begin, end);

		auto sum = std::accumulate(begin, end, T{0},
		                           [&](T in, T x) {
			                           min = std::min(x, min);
			                           max = std::max(x, max);
			                           return in + x;
		                           }
		);

		auto mean = sum / (T) n;

		auto var = std::accumulate(begin, end, T{0},
		                           [&](T in, T x) {
			                           return in + std::pow(x - mean, 2.);
		                           });

		var /= (T) n;

		T stdev = std::sqrt(var);

		std::cout << "RANDOM NORMAL : VALIDATE : no items = " << n <<
		          ", sum=" << sum <<
		          ", min=" << min <<
		          ", max=" << max <<
		          ", mean=" << mean << "[expected=" << mean_ << "]" <<
		          ", var=" << var <<
		          ", stdev=" << stdev << " [expected=" << stdev_ << "]"
		          << std::endl;
	}

	template<typename T, gen_e GEN_E>
	template<class Iterator, typename std::enable_if_t<GEN_E == RANDOM_UNIFORM> *>
	auto Datagen<T, GEN_E>::validate(Iterator begin, Iterator end, const T a, const T b) {

		return *this;
	}

	template<typename T, gen_e GEN_E>
	template<class Iterator, typename std::enable_if_t<GEN_E == RANGE> *>
	auto Datagen<T, GEN_E>::validate(Iterator begin, Iterator end, const T start) {

		return *this;
	}


}
#endif //CPP_EXAMPLE_NNET_HXX
