//
// Created by egrzrbr on 2019-04-06.
//

#ifndef CPP_EXAMPLE_DGEN_HPP
#define CPP_EXAMPLE_DGEN_HPP


namespace DGen {


	enum gen_e {
		RANDOM_NORMAL,
		RANDOM_UNIFORM,
		DATA_GEN_SWITCH,
		RANGE,
	};


	template<typename T>
	class RandGen {

		std::default_random_engine Gen;

	  public:

		auto random_normal(T mean, T stdev);

		auto random_uniform(T min, T max);


	};


	template<typename T, gen_e GEN_E>
	class Datagen : public RandGen<T> {

		std::map<std::string, T> map;

	  public:

		//Datagen()

		template<typename std::enable_if_t<GEN_E == RANDOM_NORMAL> * = nullptr>
		constexpr T operator()(T mean, T stdev);

		template<typename std::enable_if_t<GEN_E == RANDOM_UNIFORM> * = nullptr>
		constexpr T operator()(T min, T max);

		template<typename std::enable_if_t<GEN_E == RANDOM_NORMAL> * = nullptr>
		constexpr T operator()();

		template<typename std::enable_if_t<GEN_E == RANDOM_UNIFORM> * = nullptr>
		constexpr T operator()();

		template<class Iterator, typename std::enable_if_t<GEN_E < DATA_GEN_SWITCH> * = nullptr>
		auto generate(Iterator begin, Iterator end, T a, T b);

		template<class Iterator, typename std::enable_if_t<GEN_E == RANGE> * = nullptr>
		auto generate(Iterator begin, Iterator end, T start);

		template<class Iterator, typename std::enable_if_t<GEN_E == RANDOM_UNIFORM> * = nullptr>
		auto validate(Iterator begin, Iterator end, T a, T b);

		template<class Iterator, typename std::enable_if_t<GEN_E == RANDOM_NORMAL> * = nullptr>
		auto validate(Iterator begin, Iterator end, T a, T b);

		template<class Iterator, typename std::enable_if_t<GEN_E == RANGE> * = nullptr>
		auto validate(Iterator begin, Iterator end, T start);

	};

	template<typename T_IN, typename T_OUT, gen_e GEN_IN, gen_e GEN_OUT>
	struct DatagenInOut {


		Datagen<T_IN, GEN_IN> in;
		Datagen<T_OUT, GEN_OUT> out;


	};


	enum data_e {
		TRAIN_IMG,
		TRAIN_LAB,
		TEST_IMG,
		TEST_LAB,
	};

	template<debug_e DEBUG_E>
	class LoadMnist {

		constexpr static size_t MAXN_TRAIN = (size_t) 6e4;
		constexpr static size_t MAXN_TEST = (size_t) 1e4;

		constexpr static size_t ROWS = 28;
		constexpr static size_t COLS = 28;

		constexpr static size_t LABELS = 10;


		std::array<int, MAXN_TRAIN> tTrain;
		std::array<int, MAXN_TEST> tTest;
		std::vector<int> xTrain;
		std::vector<int> xTest;

		std::vector<int> zTrain;
		std::vector<int> zTest;

		std::string path;
		std::string xTrainFname = "train-images.idx3-ubyte";
		std::string tTrainFname = "train-labels.idx1-ubyte";
		std::string xTestFname = "t10k-images.idx3-ubyte";
		std::string tTestFname = "t10k-labels.idx1-ubyte";

	  public:

		constexpr auto get_labels_size() {

			return 10;
		}

		constexpr auto get_pattern_size() {

			return ROWS * COLS;
		}

		constexpr auto get_train_batch_size() {

			return MAXN_TRAIN;
		}

		constexpr auto get_test_batch_size() {

			return MAXN_TEST;
		}

		auto cbegin_xtrain() {

			return xTrain.cbegin();
		}

		auto cend_xtrain() {

			return xTrain.cend();
		}

		auto cbegin_ztrain() {

			return zTrain.cbegin();
		}

		auto cend_ztrain() {

			return zTrain.cend();
		}


		auto cbegin_xtest() {

			return xTest.cbegin();
		}

		auto cend_xtest() {

			return xTest.cend();
		}

		auto cbegin_ztest() {

			return zTest.cbegin();
		}

		auto cend_ztest() {

			return zTest.cend();
		}


		template<data_e MNIST_E>
		auto cbegin() {

			if constexpr (MNIST_E == TRAIN_IMG) {
				return xTrain.cbegin();
			} else if constexpr(MNIST_E == TEST_IMG) {
				return xTest.cbegin();
			} else if constexpr (MNIST_E == TRAIN_LAB) {
				return tTrain.cbegin();
			} else if constexpr(MNIST_E == TEST_LAB) {
				return tTest.cbegin();
			}
		}

		template<data_e MNIST_E>
		auto cend() {

			if constexpr (MNIST_E == TRAIN_IMG) {
				return xTrain.cend();
			} else if constexpr(MNIST_E == TEST_IMG) {
				return xTest.cend();
			} else if constexpr (MNIST_E == TRAIN_LAB) {
				return tTrain.cend();
			} else if constexpr(MNIST_E == TEST_LAB) {
				return tTest.cend();
			}
		}

		template<data_e MNIST_E>
		auto begin() {

			if constexpr (MNIST_E == TRAIN_IMG) {
				return xTrain.begin();
			} else if constexpr(MNIST_E == TEST_IMG) {
				return xTest.begin();
			} else if constexpr (MNIST_E == TRAIN_LAB) {
				return tTrain.begin();
			} else if constexpr(MNIST_E == TEST_LAB) {
				return tTest.begin();
			}
		}

		template<data_e MNIST_E>
		auto end() {

			if constexpr (MNIST_E == TRAIN_IMG) {
				return xTrain.end();
			} else if constexpr(MNIST_E == TEST_IMG) {
				return xTest.end();
			} else if constexpr (MNIST_E == TRAIN_LAB) {
				return tTrain.end();
			} else if constexpr(MNIST_E == TEST_LAB) {
				return tTest.end();
			}
		}


		template<char SIDE, data_e MNIST_E>
		auto iterator() {

			if constexpr(SIDE == 'b') {
				return begin<MNIST_E>();
			} else if constexpr (SIDE == 'e') {
				return end<MNIST_E>();
			}
		}


		template<size_t SIZE, typename Container_t>
		auto display_numbers(const Container_t &container, size_t id) {


			for (int k = 0; k < ROWS; k++) {
				if (k == 0) {
					std::cout << "IMAGE : " << id << " :\n";
				}
				for (int j = 0; j < COLS; j++) {
					std::cout << (uint) container[D3(SIZE, ROWS, COLS, id, k, j)] << '\t';
				}
				std::cout << '\n';
			}
			for (int i = 0; i < ROWS; i++) {
				std::cout << '=';
			}
			std::cout << '\n';

			return *this;
		}

		template<size_t SIZE, typename Container_t>
		auto display_image(const Container_t &container, size_t id) {


			for (int k = 0; k < ROWS; k++) {
				if (k == 0) {
					std::cout << "IMAGE : " << id << " :\n";
				}
				for (int j = 0; j < COLS; j++) {
					std::cout << (container[D3(SIZE, ROWS, COLS, id, k, j)] > 0 ? '#' : '-');
				}
				std::cout << '\n';
			}
			for (int i = 0; i < ROWS; i++) {
				std::cout << '=';
			}
			std::cout << '\n';

			return *this;
		}

		template<size_t SIZE, typename Container_t>
		auto display_label(const Container_t &container, size_t id) {


			return *this;
		}

		template<data_e MNIST_E>
		auto display(size_t id) {

			if constexpr(MNIST_E == TRAIN_IMG) {
				display_image<MAXN_TRAIN>(xTrain, id);
			} else if constexpr(MNIST_E == TEST_IMG) {
				display_image<MAXN_TEST>(xTest, id);
			} else if constexpr(MNIST_E == TRAIN_LAB) {
				display_label<MAXN_TRAIN>(tTrain, id);
			} else if constexpr(MNIST_E == TEST_LAB) {
				display_label<MAXN_TEST>(tTest, id);
			}

			return *this;
		}

		template<typename T, typename Container_t>
		auto read_input(Container_t &container, size_t offset, const std::string &filename) {


			if (not is_file_exist(filename)) {
				std::cout << "File : " << filename << "do not exist..." << std::endl;
				exit(1);
			}

			std::basic_ifstream<char> icin(PREPEND_PATH(filename), std::ios::in | std::ios::binary);

			icin.seekg(offset, std::ios::beg);

			std::transform(
					std::istreambuf_iterator<char>(icin),
					std::istreambuf_iterator<char>(),
					container.begin(),
					[&](char x) -> T { return static_cast<T>(static_cast<uint8_t>(x)); }
			);

			icin.close();


			return *this;

		}


		auto input_xtrain() {

			read_input<int>(xTrain, 16, xTrainFname);

			return *this;

		}


		auto input_ttrain() {

			read_input<int>(tTrain, 8, tTrainFname);

			return *this;
		}

		auto input_xtest() {

			read_input<int>(xTest, 16, xTestFname);

			return *this;
		}


		auto input_ttest() {

			read_input<int>(tTest, 8, tTestFname);

			return *this;

		}

		auto load() {

			TIME_START

			input_xtrain();
			input_xtest();
			input_ttrain();
			input_ttest();

			std::cout << "time of mnist load : ";

			TIME_CHECK

			return *this;
		}

		auto parse_labels() {

			int l;

			auto ztrain_iter = zTrain.begin();
			auto ttrain_iter = tTrain.begin();

			if constexpr (DEBUG_E >= DEBUG0) {

				TIME_START

				std::for_each(tTrain.cbegin(), tTrain.cend(),
				              [&](const int l) {
					              *(ztrain_iter + l) = 1;
					              std::advance(ztrain_iter, LABELS);
					              return l;
				              }
				);

				std::cout << "... <DEBUG 0> : LOAD MNIST : parse labels (std) : ";
				TIME_CHECK

			}

			if constexpr(DEBUG_E >= DEBUG1 and DEBUG_E <= DEBUG2) {

				TIME_START

				for (size_t i = 0, j = 0; i < tTrain.size(); i++, j += LABELS) {
					l = tTrain[i];

					if constexpr (DEBUG_E == DEBUG2) {
						std::cout << "[" << i << "," << j << "] : " << l << std::endl;
						for (size_t k = 0; k < LABELS; k++) {
							std::cout << "[" << k << "]";
						}
						std::cout << std::endl;
						for (size_t k = 0; k < LABELS; k++) {
							std::cout << " " << zTrain[j + k] << " ";
						}
						std::cout << std::endl;
					}
				}

				std::cout << "... <DEBUG 1> : LOAD MNIST : parse labels (loop) : ";
				TIME_CHECK


			} else if constexpr(DEBUG_E >= DEBUG3) {

				TIME_START

				for (size_t i = 0, j = 0; i < tTrain.size(); i++, j += LABELS) {
					l = tTrain[i];
					zTrain[j + l] = 1;

					if constexpr (DEBUG_E == DEBUG4) {
						std::cout << "[" << i << "," << j << "] : " << l << std::endl;
						for (size_t k = 0; k < LABELS; k++) {
							std::cout << "[" << k << "]";
						}
						std::cout << std::endl;
						for (size_t k = 0; k < LABELS; k++) {
							std::cout << " " << zTrain[j + k] << " ";
						}
						std::cout << std::endl;
					}
				}

				std::cout << "... <DEBUG 3> : LOAD MNIST : parse labels (loop) : ";

				TIME_CHECK


			}

			return *this;
		}

		LoadMnist() {};

		LoadMnist(const char *path, bool doLoad = true) :
				path(path),
				zTrain(MAXN_TRAIN * LABELS, 0),
				zTest(MAXN_TEST * LABELS, 0),
				xTrain(MAXN_TRAIN * ROWS * COLS, 0),
				xTest(MAXN_TEST * ROWS * COLS, 0) {

			if (doLoad) {

				load();

				parse_labels();

			}

		}

	};


}

#include "DGen.hxx"

#endif //CPP_EXAMPLE_DGEN_HPP
