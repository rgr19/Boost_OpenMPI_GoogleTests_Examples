//
// Created by egrzrbr on 2019-04-10.
//

#ifndef CPP_EXAMPLE_TESTS_HPP
#define CPP_EXAMPLE_TESTS_HPP

bool dgen_tests() {

	try {
		auto datagen = DGen::DatagenInOut<int, float, DGen::RANDOM_UNIFORM, DGen::RANDOM_NORMAL>();

		#define NTRIALS 10000

		{
			TIME_START
			for (int i = 0; i < NTRIALS; i++) {
				DGen::rand_uniform_int(0, 1);
			}
			std::cout << "Test : rand_uniform_int :";
			TIME_CHECK
		}
		{

			TIME_START
			for (int i = 0; i < NTRIALS; i++) {
				DGen::rand_uniform_real(0, 1);
			}
			std::cout << "Test : rand_uniform_real :";
			TIME_CHECK
		}
		{
			TIME_START
			for (int i = 0; i < NTRIALS; i++) {
				DGen::rand_normal(0, 1);
			}
			std::cout << "Test : rand_normal :";
			TIME_CHECK
		}

		{
			TIME_START
			for (int i = 0; i < NTRIALS; i++) {
				datagen.in(0, 1);
			}

			std::cout << "Test : datagen.in<int>() :";
			TIME_CHECK
		}
		{
			TIME_START
			for (int i = 0; i < NTRIALS; i++) {
				datagen.out(0, 1);
			}

			std::cout << "Test : datagen.out<float>() :";
			TIME_CHECK
		}
		auto arr = std::array<int, NTRIALS>();

		{
			TIME_START

			datagen.in.generate(arr.begin(), arr.end(), 0, 1);

			std::cout << "Test : datagen.int<int>generate() :";
			TIME_CHECK
		}
		{
			TIME_START

			datagen.out.generate(arr.begin(), arr.end(), 0, 1);

			std::cout << "Test : datagen.out<float>.enerate() :";
			TIME_CHECK
		}
		std::cout << "Finish." << std::endl;

		return true;
	} catch (const std::exception &e) {
		std::cout << e.what() << std::endl;
		return false;
	}
}

#endif //CPP_EXAMPLE_TESTS_HPP
