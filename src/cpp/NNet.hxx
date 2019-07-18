//
// Created by egrzrbr on 2019-04-06.
//

#ifndef CPP_EXAMPLE_NNET_HXX
#define CPP_EXAMPLE_NNET_HXX


#include "NNet.hpp"

namespace NNet {


	template<typename T>
	LayerBase::LayerBase(T l, size_t N, size_t M, const char *type) : m_shape{N, M}, m_type(type) {


		make_name(l);


		std::cout << "LayerBase : name = "
		          << m_name << std::endl;
	}


	void LayerBase::make_name(size_t l) {

		m_name = "[" + m_type + " : " + std::to_string(l) + "]:(" + std::to_string(m_shape.first) + "," +
		         std::to_string(m_shape.second) + ")";
	}

	auto LayerBase::get_name() {

		return m_name;
	}

	void LayerBase::print() {

		std::cout << m_name << std::endl;
	}

	void LayerBase::print_name() {

		print();

	}


	LayerNull::LayerNull(size_t id) : LayerBase(id, 0, 0, NAME) {

	}

	auto LayerNull::get_next() {

		return nullptr;
	}


	template<typename T, size_t mB, size_t N, size_t M, size_t ... shapeNN>
	LayerHidden<T, mB, N, M, shapeNN...>::LayerHidden(size_t id, const char *_type) : LayerBase(id, N, M, _type) {

	}

	template<typename T, size_t mB, size_t N, size_t M, size_t ... shapeNN>
	void LayerHidden<T, mB, N, M, shapeNN...>::print() {

		LayerBase::print();

	}

	template<typename T, size_t mB, size_t N, size_t M, size_t ...shapeNN>
	LayerHidden<T, mB, N, M, shapeNN...>::LayerHidden(size_t id) : LayerHidden(id, NAME) {


	}

	template<typename T, size_t mB, size_t N, size_t M, size_t ...shapeNN>
	void LayerHidden<T, mB, N, M, shapeNN...>::print_W() {

		for (int i = 0; i < M * N; i++) {
			std::cout << W[i] << std::endl;
		}

	}

	template<typename T, size_t mB, size_t N, size_t M, size_t ...shapeNN>
	void LayerHidden<T, mB, N, M, shapeNN...>::print_b() {

		for (int i = 0; i < M * mB; i++) {
			std::cout << b[i] << std::endl;
		}
	}

	template<typename T, size_t mB, size_t N, size_t M, size_t... shapeNN>
	auto LayerHidden<T, mB, N, M, shapeNN...>::get_W() {

		return &W;
	}

	template<typename T, size_t mB, size_t N, size_t M, size_t...shapeNN>
	auto LayerHidden<T, mB, N, M, shapeNN...>::W_begin() {

		return W.begin();
	}

	template<typename T, size_t mB, size_t N, size_t M, size_t...shapeNN>
	auto LayerHidden<T, mB, N, M, shapeNN...>::W_end() {

		return W.end();
	}

	template<typename T, size_t mB, size_t N, size_t M, size_t...shapeNN>
	auto LayerHidden<T, mB, N, M, shapeNN...>::dW_begin() {

		return dW.begin();
	}

	template<typename T, size_t mB, size_t N, size_t M, size_t...shapeNN>
	auto LayerHidden<T, mB, N, M, shapeNN...>::dW_end() {

		return dW.end();
	}

	template<typename T, size_t mB, size_t N, size_t M, size_t...shapeNN>
	auto LayerHidden<T, mB, N, M, shapeNN...>::Theta_begin() {

		return Theta.begin();
	}

	template<typename T, size_t mB, size_t N, size_t M, size_t...shapeNN>
	auto LayerHidden<T, mB, N, M, shapeNN...>::Theta_end() {

		return Theta.end();
	}

	template<typename T, size_t mB, size_t N, size_t M, size_t...shapeNN>
	auto LayerHidden<T, mB, N, M, shapeNN...>::dTheta_begin() {

		return dTheta.begin();
	}

	template<typename T, size_t mB, size_t N, size_t M, size_t...shapeNN>
	auto LayerHidden<T, mB, N, M, shapeNN...>::dTheta_end() {

		return dTheta.end();
	}

	template<typename T, size_t mB, size_t N, size_t M, size_t...shapeNN>
	auto LayerHidden<T, mB, N, M, shapeNN...>::Delta_begin() {

		return Delta.begin();
	}

	template<typename T, size_t mB, size_t N, size_t M, size_t...shapeNN>
	auto LayerHidden<T, mB, N, M, shapeNN...>::Delta_end() {

		return Delta.end();
	}

	template<typename T, size_t mB, size_t N, size_t M, size_t...shapeNN>
	auto LayerHidden<T, mB, N, M, shapeNN...>::V_begin() {

		return V.begin();
	}

	template<typename T, size_t mB, size_t N, size_t M, size_t...shapeNN>
	auto LayerHidden<T, mB, N, M, shapeNN...>::V_end() {

		return V.end();
	}

	template<typename T, size_t mB, size_t N, size_t M, size_t...shapeNN>
	auto LayerHidden<T, mB, N, M, shapeNN...>::dV_begin() {

		return dV.begin();
	}

	template<typename T, size_t mB, size_t N, size_t M, size_t...shapeNN>
	auto LayerHidden<T, mB, N, M, shapeNN...>::dV_end() {

		return dV.end();
	}

	template<typename T, size_t mB, size_t N, size_t M, size_t...shapeNN>
	auto LayerHidden<T, mB, N, M, shapeNN...>::b_begin() {

		return b.begin();
	}

	template<typename T, size_t mB, size_t N, size_t M, size_t...shapeNN>
	auto LayerHidden<T, mB, N, M, shapeNN...>::b_end() {

		return b.end();
	}


	template<typename T, size_t B, size_t mB, size_t N, size_t M>
	LayerOutput<T, B, mB, N, M>::LayerOutput(size_t id) : LayerHidden<T, mB, N, M>(id, type) {

		std::cout << "... <INFO> Layer Output : classifiers [Z] : size : " << Z.size() << std::endl;

	}

	template<typename T, size_t B, size_t mB, size_t N, size_t M>
	auto LayerOutput<T, B, mB, N, M>::get_Z() {

		return &Z;
	}

	template<typename T, size_t B, size_t mB, size_t N, size_t M>
	auto LayerOutput<T, B, mB, N, M>::Z_begin() {

		return Z.begin();
	}

	template<typename T, size_t B, size_t mB, size_t N, size_t M>
	auto LayerOutput<T, B, mB, N, M>::Z_end() {

		return Z.end();
	}

	template<typename T, size_t B, size_t mB, size_t N>
	LayerInput<T, B, mB, N>::LayerInput(size_t id) : LayerBase(id, 0, N, NAME) {

		if (V.empty()) {
			std::cout << "... <ERROR> : Layer Input not allocated [V]..." << std::endl;
		} else {
			std::cout << "... <INFO> : Layer Input allocated [V]..." << std::endl;
		}

	}

	template<typename T, size_t B, size_t mB, size_t N>
	void LayerInput<T, B, mB, N>::print() {

		LayerBase::print();
	}

	template<typename T, size_t B, size_t mB, size_t N>
	auto LayerInput<T, B, mB, N>::get_V() {

		return &V;
	}

	template<typename T, size_t B, size_t mB, size_t N>
	auto LayerInput<T, B, mB, N>::V_begin() {

		return V.begin();
	}

	template<typename T, size_t B, size_t mB, size_t N>
	auto LayerInput<T, B, mB, N>::V_end() {

		return V.end();
	}

	template<class Curr_t, class Next_t>
	Node<Curr_t, Next_t>::Node(size_t l, Curr_Ptr_t curr, Next_Ptr_t next) : l(l), m_curr(curr), m_next(next) {

		assert(m_curr != nullptr);
		assert(m_next != nullptr);

	}

	template<class Curr_t, class Next_t>
	Node<Curr_t, Next_t>::Node(size_t l, Next_Ptr_t next) : l(l), m_next(next) {

		m_curr = new Curr_t(l);

		assert(m_curr != nullptr);
		assert(m_next != nullptr);

	}

	template<class Curr_t, class Next_t>
	void Node<Curr_t, Next_t>::print() {

		m_curr->print();
		m_next->print();
	}

	template<class Curr_t, class Next_t>
	auto Node<Curr_t, Next_t>::get_next() {

		return m_next->get_next();
	}

	template<class Curr_t, class Next_t>
	void Node<Curr_t, Next_t>::print_name() {

		m_curr->print();

	}


	template<class Curr_t, class Next_Linked_Ptr_t>
	Linked<Curr_t, Next_Linked_Ptr_t>
	::Linked(Curr_Ptr_t curr, Next_Linked_Ptr_t next, size_t l) : l(l) {

		m_head = new node_t(l, curr, next);

		assert(m_head != nullptr);
	}

	template<class Curr_t, class Next_Linked_Ptr_t>
	Linked<Curr_t, Next_Linked_Ptr_t>
	::Linked(Next_Linked_Ptr_t next, size_t l) : l(l) {

		m_head = new node_t(l, next);
		assert(m_head != nullptr);

	}

	template<class Curr_t, class Next_Linked_Ptr_t>
	void Linked<Curr_t, Next_Linked_Ptr_t>
	::print() {

		if (l == 0) std::cout << "## LINKED PRINT :..." << std::endl;

		if (m_head != nullptr) {
			m_head->print();
		}
	}

	template<class Curr_t, class Next_Linked_Ptr_t>
	void Linked<Curr_t, Next_Linked_Ptr_t>
	::print_name() {

		m_head->m_curr->print();
	}

	template<class Curr_t, class Next_Linked_Ptr_t>
	auto Linked<Curr_t, Next_Linked_Ptr_t>
	::get_name() {

		return m_head->m_curr->get_name();
	}

	template<class Curr_t, class Next_Linked_Ptr_t>
	auto Linked<Curr_t, Next_Linked_Ptr_t>
	::get_curr() {

		return m_head->m_curr;
	}

	template<class Curr_t, class Next_Linked_Ptr_t>
	auto Linked<Curr_t, Next_Linked_Ptr_t>
	::get_next() {


		return m_head->m_next;

	}

	template<class Curr_t, class Next_Linked_Ptr_t>
	void Linked<Curr_t, Next_Linked_Ptr_t>
	::print_V() {

		std::cout << get_name() << " : feed [V] : " << std::endl;
		for (const auto &a : m_head->m_curr->V) {
			std::cout << a << std::endl;
		}

	}

	template<class Curr_t, class Next_Linked_Ptr_t>
	auto Linked<Curr_t, Next_Linked_Ptr_t>
	::get_W() {

		return m_head->m_curr->get_W();
	}

	template<class Curr_t, class Next_Linked_Ptr_t>
	auto Linked<Curr_t, Next_Linked_Ptr_t>
	::get_V() {

		return m_head->m_curr->get_V();
	}

	template<class Curr_t, class Next_Linked_Ptr_t>
	auto Linked<Curr_t, Next_Linked_Ptr_t>
	::V_begin() {

		return m_head->m_curr->V_begin();
	}

	template<class Curr_t, class Next_Linked_Ptr_t>
	auto Linked<Curr_t, Next_Linked_Ptr_t>
	::V_end() {

		return m_head->m_curr->V_end();
	}

	template<class Curr_t, class Next_Linked_Ptr_t>
	auto Linked<Curr_t, Next_Linked_Ptr_t>
	::Z_begin() {

		return m_head->m_curr->Z_begin();
	}

	template<class Curr_t, class Next_Linked_Ptr_t>
	auto Linked<Curr_t, Next_Linked_Ptr_t>
	::Z_end() {

		return m_head->m_curr->Z_end();
	}

	template<class Curr_t, class Next_Linked_Ptr_t>
	auto Linked<Curr_t, Next_Linked_Ptr_t>
	::get_Z() {

		return m_head->m_curr->get_Z();
	}

	template<class Curr_t, class Next_Linked_Ptr_t>
	auto Linked<Curr_t, Next_Linked_Ptr_t>
	::W_begin() {

		return m_head->m_curr->W_begin();
	}

	template<class Curr_t, class Next_Linked_Ptr_t>
	auto Linked<Curr_t, Next_Linked_Ptr_t>
	::W_end() {

		return m_head->m_curr->W_end();
	}

	template<class Curr_t, class Next_Linked_Ptr_t>
	auto Linked<Curr_t, Next_Linked_Ptr_t>
	::dW_begin() {

		return m_head->m_curr->dW_begin();
	}

	template<class Curr_t, class Next_Linked_Ptr_t>
	auto Linked<Curr_t, Next_Linked_Ptr_t>
	::dW_end() {

		return m_head->m_curr->dW_end();
	}

	template<class Curr_t, class Next_Linked_Ptr_t>
	auto Linked<Curr_t, Next_Linked_Ptr_t>
	::dV_begin() {

		return m_head->m_curr->dV_begin();
	}

	template<class Curr_t, class Next_Linked_Ptr_t>
	auto Linked<Curr_t, Next_Linked_Ptr_t>
	::dV_end() {

		return m_head->m_curr->dV_end();
	}

	template<class Curr_t, class Next_Linked_Ptr_t>
	auto Linked<Curr_t, Next_Linked_Ptr_t>
	::b_begin() {

		return m_head->m_curr->b_begin();
	}

	template<class Curr_t, class Next_Linked_Ptr_t>
	auto Linked<Curr_t, Next_Linked_Ptr_t>
	::b_end() {

		return m_head->m_curr->b_end();
	}

	template<class Curr_t, class Next_Linked_Ptr_t>
	auto Linked<Curr_t, Next_Linked_Ptr_t>
	::Delta_begin() {

		return m_head->m_curr->Delta_begin();
	}

	template<class Curr_t, class Next_Linked_Ptr_t>
	auto Linked<Curr_t, Next_Linked_Ptr_t>
	::Delta_end() {

		return m_head->m_curr->Delta_end();
	}

	template<class Curr_t, class Next_Linked_Ptr_t>
	auto Linked<Curr_t, Next_Linked_Ptr_t>
	::Theta_begin() {

		return m_head->m_curr->Theta_begin();
	}

	template<class Curr_t, class Next_Linked_Ptr_t>
	auto Linked<Curr_t, Next_Linked_Ptr_t>
	::Theta_end() {

		return std::move(m_head->m_curr->Theta_end());
	}


	template<class Curr_t, class Next_Linked_Ptr_t>
	auto Linked<Curr_t, Next_Linked_Ptr_t>
	::dTheta_begin() {

		return std::move(m_head->m_curr->dTheta_begin());
	}

	template<class Curr_t, class Next_Linked_Ptr_t>
	auto Linked<Curr_t, Next_Linked_Ptr_t>
	::dTheta_end() {

		return std::move(m_head->m_curr->dTheta_end());
	}

	void LayersBase::print() {

	}

	LayersBase::LayersBase() {

	}

	template<typename T_IN, typename T, size_t B, size_t mB, size_t I, size_t ... shapeNN>
	template<debug_e DEBUG_E, class Engine_Ptr, class Prev_Layer_Ptr_t, class Next_Layer_Ptr_t>
	auto LayersMaker<T_IN, T, B, mB, I, shapeNN...>
	::forward(const Engine_Ptr enginePtr, const Prev_Layer_Ptr_t prevLinkedPtr, const Next_Layer_Ptr_t nextLinkedPtr, size_t l) {

		/* Propagate forward:
		 * Linked<O, Linked<H2, Linked<H1, Linked<I>>>
		 */

		const auto &prevPtr = prevLinkedPtr->get_curr();
		const auto &nextPtr = nextLinkedPtr->get_curr();


		if (DEBUG_E == DEBUG0) {}

		if (DEBUG_E >= DEBUG1) //debug
		{

			std::cout << "FORWARD[" << l << "] " << std::endl; //
			std::cout << ".... curr = " << prevPtr->get_name() << std::endl; //out, h3
			std::cout << ".... next = " << nextPtr->get_name() << std::endl;
		}
	}

	template<typename T_IN, typename T, size_t B, size_t mB, size_t I, size_t ... shapeNN>
	template<debug_e DEBUG_E, class Engine_Ptr, class Prev_Layer_Ptr_t, class Next_Layer_Ptr_t>
	auto LayersMaker<T_IN, T, B, mB, I, shapeNN...>
	::backward(const Engine_Ptr enginePtr, const Prev_Layer_Ptr_t prevLinkedPtr, const Next_Layer_Ptr_t nextLinkedPtr, size_t l) {

		/* Propagate backward:
		 * Linked<I, Linked<H1, Linked<H2, Linked<O>>>
		 */

		const auto &prevPtr = prevLinkedPtr;
		const auto &nextPtr = nextLinkedPtr->get_curr();

		if (DEBUG_E == DEBUG0) {}

		if (DEBUG_E >= DEBUG1) //debug
		{
			std::cout << "BACKWARD[" << l << "]" << std::endl; //

			std::cout << ".... curr = " << nextPtr->get_name() << std::endl;
			std::cout << ".... prev = " << prevPtr->get_name() << std::endl;

		}
	}

	template<typename T_IN, typename T, size_t B, size_t mB, size_t I, size_t ... shapeNN>
	template<debug_e DEBUG_E, size_t N, size_t M, class Engine_Ptr, class Curr_Linked_Ptr_t>
	auto LayersMaker<T_IN, T, B, mB, I, shapeNN...>
	::compute(const Engine_Ptr enginePtr, const Curr_Linked_Ptr_t last, size_t l) {


		const auto &outp = last->get_next();

		if (DEBUG_E >= DEBUG1) {
			std::cout << "OUTLAYER[" << l << "], " << N << std::endl;
			std::cout << ".... last = " << last->get_name() << std::endl;
			std::cout << ".... out  = " << outp->get_name() << std::endl;
			std::cout << ".... back = " << last->get_name() << std::endl;
		}

		if (DEBUG_E >= DEBUG0) {
			TIME_START
			//Engine_Ptr::mat_mul_vec<T, T_IN, mB, N, M>(
			//		outp->get_curr()->W,
			//		last->get_curr()->V,
			//		outp->get_curr()->b);
			TIME_CHECK
		}

		if (DEBUG_E >= DEBUG2) {

			std::cout << "Weights:" << std::endl;
			outp->get_curr()->print_W();
			std::cout << "Local field:" << std::endl;
			outp->get_curr()->print_b();

		}

		return last;

	}

	template<typename T_IN, typename T, size_t B, size_t mB, size_t I, size_t ... shapeNN>
	template<debug_e DEBUG_E, size_t N, size_t M, size_t K, size_t ... nextShapeNN, class Engine_Ptr, class Curr_Linked_Ptr_t>
	auto LayersMaker<T_IN, T, B, mB, I, shapeNN...>
	::compute(const Engine_Ptr enginePtr, const Curr_Linked_Ptr_t currPtr, size_t l) {


		if (DEBUG_E >= DEBUG1)
			obj_assert_ptr(currPtr);

		const auto &nextPtr = currPtr->get_next();

		if (DEBUG_E >= DEBUG1)
			obj_assert_ptr(nextPtr);

		forward<DEBUG_E>(enginePtr, currPtr, nextPtr, l);

		compute<DEBUG_E, M, K, nextShapeNN...>(enginePtr, nextPtr, l + 1);

		backward<DEBUG_E>(enginePtr, currPtr, nextPtr, l);

		return nextPtr;

	}

	template<typename T_IN, typename T, size_t B, size_t mB, size_t I, size_t ... shapeNN>
	template<debug_e DEBUG_E, class Engine_Ptr, class Linked_Ptr_t>
	auto LayersMaker<T_IN, T, B, mB, I, shapeNN...>
	::compute(const Engine_Ptr enginePtr, Linked_Ptr_t layers) {


		if (DEBUG_E >= DEBUG1) {
			std::cout << "COMPUTE FORWARD AND BACK >>>" << std::endl;
			type_assert_ptr(Linked_Ptr_t);
		}

		compute<DEBUG_E, I, shapeNN...>(enginePtr, layers, 0);


		return *this;
	}

	template<typename T_IN, typename T, size_t B, size_t mB, size_t I, size_t ... shapeNN>
	template<debug_e DEBUG_E, class Datain_Ptr, class Linked_Ptr_t>
	auto LayersMaker<T_IN, T, B, mB, I, shapeNN...>
	::init_patterns(Datain_Ptr datainPtr, Linked_Ptr_t layers) {


		if constexpr(DEBUG_E >= DEBUG0) {

			std::cout << "... <DEBUG 0> init patterns" << std::endl;

			std::transform(
					datainPtr.cbegin_xtrain(),
					datainPtr.cend_xtrain(),
					layers->V_begin(),
					[&](const auto &x) -> T_IN { return static_cast<T_IN>(x) / T_IN{255}; }
			);

		}

		if constexpr(DEBUG_E >= DEBUG1) {
			//datagenPtr.in.validate(feed->begin(), feed->end(), -1, 1);

			auto sum = std::accumulate(layers->V_begin(), layers->V_end(), T_IN{0});

			std::cout << "... <DEBUG 1> : INIT PATTERNS : SUM = " << sum << std::endl;
		}
	}

	template<typename T_IN, typename T, size_t B, size_t mB, size_t I, size_t ... shapeNN>
	template<debug_e DEBUG_E, class Datain_Ptr, class Linked_Ptr_t>
	auto LayersMaker<T_IN, T, B, mB, I, shapeNN...>
	::init_classifiers(Datain_Ptr datainPtr, Linked_Ptr_t layers) {

		if constexpr(DEBUG_E >= DEBUG0) {
			std::cout << "... <DEBUG 0> init patterns" << std::endl;
			std::transform(
					datainPtr.cbegin_ztrain(),
					datainPtr.cend_ztrain(),
					layers->Z_begin(),
					[&](const auto &x) { return static_cast<T>(x); }
			);
		}

		if constexpr(DEBUG_E >= DEBUG1) {
			//datagenPtr.in.validate(feed->begin(), feed->end(), -1, 1);
			auto sum = std::accumulate(layers->Z_begin(), layers->Z_end(), T{0});
			std::cout << "... <DEBUG 1> : INIT CLASSIFIERS : SUM = " << sum << std::endl;
		}
	}

	template<typename T_IN, typename T, size_t B, size_t mB, size_t I, size_t ... shapeNN>
	template<debug_e DEBUG_E, size_t N, class Datagen_Ptr, class Linked_Ptr_t>
	auto LayersMaker<T_IN, T, B, mB, I, shapeNN...>
	::init_parameters(Datagen_Ptr &datagenPtr, Linked_Ptr_t layers) {

		T mean = 0;
		T var = T{1} / T{N};
		var = var > 0 ? var : 1;
		T stdev = std::sqrt(var);

		std::cout << "init random weights DEBUG_E = " << DEBUG_E << std::endl;

		if constexpr (DEBUG_E >= DEBUG0) {
			datagenPtr.generate(layers->W_begin(), layers->W_end(), mean, stdev);
		}

		if constexpr (DEBUG_E >= DEBUG1) {
			datagenPtr.validate(layers->W_begin(), layers->W_end(), mean, stdev);
		}

		return *this;
	}

	template<typename T_IN, typename T, size_t B, size_t mB, size_t I, size_t ... shapeNN>
	template<debug_e DEBUG_E, size_t N, size_t M, class Datain_Ptr, class Datagen_Ptr, class Curr_Linked_Ptr_t>
	void LayersMaker<T_IN, T, B, mB, I, shapeNN...>
	::init(Datain_Ptr &datainPtr, Datagen_Ptr &datagenPtr, const Curr_Linked_Ptr_t last, size_t l) {

		if (DEBUG_E > DEBUG0) {
			std::cout << "## INIT LAST :... " << std::endl;

			type_assert_ptr(Curr_Linked_Ptr_t);
		}
		const auto outp = last->get_next();


		init_classifiers<DEBUG_E>(datainPtr, outp);


		init_parameters<DEBUG_E, N>(datagenPtr, outp);

		outp->print_name();

	}

	template<typename T_IN, typename T, size_t B, size_t mB, size_t I, size_t ... shapeNN>
	template<debug_e DEBUG_E, size_t N, size_t M, size_t K, size_t ... nextShapeNN, class Datain_Ptr, class Datagen_Ptr, class Curr_Linked_Ptr_t>
	void LayersMaker<T_IN, T, B, mB, I, shapeNN...>
	::init(Datain_Ptr &datainPtr, Datagen_Ptr &datagenPtr, const Curr_Linked_Ptr_t curr, size_t l) {

		if (DEBUG_E > DEBUG0) {
			std::cout << "## INIT HIDDEN :... " << std::endl;
			type_assert_ptr(Curr_Linked_Ptr_t);
		}

		const auto next = curr->get_next();

		next->print_name();

		if (DEBUG_E > DEBUG0)
			obj_assert_ptr(next);


		curr->print_name();
		next->print_name();

		init_parameters<DEBUG_E, M>(datagenPtr, next);

		init < DEBUG_E, M, K, nextShapeNN...>(datainPtr, datagenPtr, next, l + 1);


	}

	template<typename T_IN, typename T, size_t B, size_t mB, size_t I, size_t ... shapeNN>
	template<debug_e DEBUG_E, class Datain_Ptr, class Datagen_Ptr, class Linked_Ptr_t>
	auto LayersMaker<T_IN, T, B, mB, I, shapeNN...>
	::init(Datain_Ptr datainPtr, Datagen_Ptr datagenPtr, Linked_Ptr_t layers) {

		if (DEBUG_E > DEBUG0) {
			std::cout << "#### LAYERS : INIT : START >>>" << std::endl;
			type_assert_ptr(Linked_Ptr_t);
			//type_assert_ptr(Datagen_Ptr);
			layers->print_name();
		}


		std::cout << "... init input data" << std::endl;

		init_patterns<DEBUG_E>(datainPtr, layers);

		layers->print_name();

		init<DEBUG_E, I, shapeNN...>(datainPtr, datagenPtr, layers, 1);

		return *this;
	}


	template<typename T_IN, typename T, size_t B, size_t mB, size_t I, size_t ... shapeNN>
	template<debug_e DEBUG_E, size_t N, size_t M>
	auto LayersMaker<T_IN, T, B, mB, I, shapeNN...>
	::alloc(size_t l) {

		auto next = new LayerNull(l + 1);

		obj_assert_ptr(next);

		assert(next != nullptr);

		auto link = new Linked<LayerOutput<T, B, mB, N, M>, decltype(next)>(next, l);

		assert(link != nullptr);

		return link;

	}

	template<typename T_IN, typename T, size_t B, size_t mB, size_t I, size_t ... shapeNN>
	template<debug_e DEBUG_E, size_t N, size_t M, size_t K, size_t ... nextShapeNN>
	auto LayersMaker<T_IN, T, B, mB, I, shapeNN...>
	::alloc(size_t l) {

		auto next = alloc<DEBUG_E, M, K, nextShapeNN...>(l + 1);

		assert(next != nullptr);

		obj_assert_ptr(next);

		auto link = new Linked<LayerHidden<T, mB, N, M>, decltype(next)>(next, l);

		assert(link != nullptr);

		return link;

	}

	template<typename T_IN, typename T, size_t B, size_t mB, size_t I, size_t ... shapeNN>
	template<debug_e DEBUG_E>
	auto LayersMaker<T_IN, T, B, mB, I, shapeNN...>
	::alloc() {

		std::cout << "#### LAYERS : ALLOC : START >>>" << std::endl;

		auto next = alloc<DEBUG_E, I, shapeNN...>(1);

		obj_assert_ptr(next);

		return new Linked<LayerInput<T_IN, B, mB, I>, decltype(next)>(next, 0);

	}

	template<typename T_IN, typename T, size_t B, size_t mB, size_t I, size_t ... shapeNN>
	template<class Linked_Ptr_t>
	auto LayersMaker<T_IN, T, B, mB, I, shapeNN...>
	::print(const Linked_Ptr_t layers) {

		type_assert_ptr(Linked_Ptr_t);

		layers->print();

		return *this;
	}

	template<typename T_IN, typename T, size_t B, size_t mB, size_t I, size_t ... shapeNN>
	LayersMaker<T_IN, T, B, mB, I, shapeNN...>
	::LayersMaker() {

		std::cout << "#### LAYERS : CONSTRUCT >>>" << std::endl;


		nMiniBatch = mB;
		nFullBatch = B;
		//init_random_bitmap<DEBUG1, T_IN, I, mB>(input->get_curr()->V);

		std::cout << m_shapeNN << std::endl;


	}

	template<class Layers_t, class Linked_Ptr_t, class Engine_Ptr, class Datain_Ptr, class Datagen_Ptr>
	Network<Layers_t, Linked_Ptr_t, Engine_Ptr, Datain_Ptr, Datagen_Ptr>
	::Network(Layers_t &layers, Linked_Ptr_t &linkedPtr, Engine_Ptr &enginePtr, Datain_Ptr &datainPtr, Datagen_Ptr &datagenPtr) :
			layers(layers), linkedPtr(linkedPtr), enginePtr(enginePtr), datainPtr(datainPtr), datagenPtr(datagenPtr) {


	}


	template<class Layers_t, class Linked_Ptr_t, class Engine_Ptr, class Datain_Ptr, class Datagen_Ptr>
	template<debug_e DEBUG_E, class Prev_Linked_Ptr_t, class Next_Linked_Ptr_t>
	auto Network<Layers_t, Linked_Ptr_t, Engine_Ptr, Datain_Ptr, Datagen_Ptr>
	::forward(Prev_Linked_Ptr_t prevLinkedPtr, Next_Linked_Ptr_t nextLinkedPtr, size_t l) {
		/* for l in range(1, L):
		 *     N, M = curr.shape
		 *           curr                              curr           prev             curr
		 *     b[l][:M, :mB] = einsum('ij,jm->im', W[l][:M,:N], V[l-1][:N, :mB]) - Theta[l][:M]
		 *     V[l][:M, :mB] = g(b[l][:M, :mB])
		 *    dV[l][:M, :mB] = dg(V[l][:M, :mB])
		 *
		 */

		using Prev_Linked_t = std::remove_pointer_t<Prev_Linked_Ptr_t>;
		using Next_Linked_t = std::remove_pointer_t<Next_Linked_Ptr_t>;
		constexpr layers_e PREV_EID = Prev_Linked_t::EID;
		constexpr layers_e NEXT_EID = Next_Linked_t::EID;
		constexpr size_t N = Next_Linked_t::N;
		constexpr size_t M = Next_Linked_t::M;

		auto X_begin = prevLinkedPtr->V_begin();
		auto X_end = prevLinkedPtr->V_end();


		auto V_begin = nextLinkedPtr->V_begin();
		auto V_end = nextLinkedPtr->V_end();

		auto dV_begin = nextLinkedPtr->dV_begin();
		auto dV_end = nextLinkedPtr->dV_end();
		auto b_begin = nextLinkedPtr->b_begin();
		auto b_end = nextLinkedPtr->b_end();
		auto Theta_begin = nextLinkedPtr->Theta_begin();
		auto Theta_end = nextLinkedPtr->Theta_end();
		auto W_begin = nextLinkedPtr->W_begin();
		auto W_end = nextLinkedPtr->W_end();

		if constexpr (DEBUG_E > DEBUG0) {
			std::cout << "[FORWARD] : ... N=" << N << " M=" << M << std::endl;
			std::cout << "PREV -> ";
			prevLinkedPtr->print_name();
			std::cout << "NEXT -> ";
			nextLinkedPtr->print_name();

			for (size_t i = 0; i < M; i++) {
				float sum = 0;
				for (size_t j = 0; j < N; j++) {
					const float w = *W_begin++;
					const float x = *X_begin++;
					sum += w * x;
				}
				*b_begin++ = sum;
				std::advance(X_begin, -N);
			}

			std::advance(b_begin, -M);

			if constexpr (DEBUG_E > DEBUG0) {
				std::cout << "   LOCAL FIELDS = [" << std::endl << "    ";
				for (size_t j = 0; j < mB; j++) {
					std::cout << "        mB : [" << j << std::endl << "        ";
					for (size_t i = 0; i < M; i++) {

						std::cout << *b_begin++ << "\t\t";
						if (i > 0 and i % 6 == 0) std::cout << std::endl << "        ";
					}
					std::cout << std::endl << "        ]" << std::endl << "        ";
				}

				std::cout << std::endl << "    ]" << std::endl << "    ";
			}
		}

		if constexpr (NEXT_EID == LAYER_OUTPUT) {
			if constexpr (DEBUG_E > DEBUG0) {
				std::cout << "[COMPUTE DELTAS] : ... N=" << N << " M=" << M << std::endl;
				std::cout << "PREV -> ";
				prevLinkedPtr->print_name();
				std::cout << "NEXT -> ";
				nextLinkedPtr->print_name();
			}

		}

	}

	template<class Layers_t, class Linked_Ptr_t, class Engine_Ptr, class Datain_Ptr, class Datagen_Ptr>
	template<debug_e DEBUG_E, class Prev_Linked_Ptr_t, class Next_Linked_Ptr_t>
	auto Network<Layers_t, Linked_Ptr_t, Engine_Ptr, Datain_Ptr, Datagen_Ptr>
	::backward(Prev_Linked_Ptr_t prevLinkedPtr, Next_Linked_Ptr_t nextLinkedPtr, size_t l) {


		using Prev_Linked_t = std::remove_pointer_t<Prev_Linked_Ptr_t>;
		using Next_Linked_t = std::remove_pointer_t<Next_Linked_Ptr_t>;
		constexpr layers_e PREV_EID = Prev_Linked_t::EID;
		constexpr layers_e NEXT_EID = Next_Linked_t::EID;


		if constexpr (PREV_EID == LAYER_OUTPUT) {
			if constexpr (DEBUG_E > DEBUG0) {
				std::cout << "[BACKWARD FROM OUTPUT] : ..." << std::endl;
				prevLinkedPtr->print_name();
				nextLinkedPtr->print_name();
			}


		} else if constexpr (PREV_EID == LAYER_HIDDEN) {
			if constexpr (DEBUG_E > DEBUG0) {
				std::cout << "[BACKWARD FROM HIDDEN] : ..." << std::endl;
				prevLinkedPtr->print_name();
				nextLinkedPtr->print_name();
			}

		}

	}


	template<class Layers_t, class Linked_Ptr_t, class Engine_Ptr, class Datain_Ptr, class Datagen_Ptr>
	template<debug_e DEBUG_E, class Curr_Linked_Ptr_t>
	auto Network<Layers_t, Linked_Ptr_t, Engine_Ptr, Datain_Ptr, Datagen_Ptr>
	::compute(Curr_Linked_Ptr_t curr, size_t l) {

		using Curr_Linked_t = std::remove_pointer_t<Curr_Linked_Ptr_t>;
		constexpr bool HAS_NEXT = Curr_Linked_t::HAS_NEXT;
		constexpr layers_e EID = Curr_Linked_t::EID;
		auto next = curr->get_next();

		if constexpr (HAS_NEXT) {

			if constexpr(EID == LAYER_INPUT) {
				if constexpr (DEBUG_E > DEBUG0) {
					std::cout << "[RESAMPLE INPUT]" << std::endl;
					curr->print_name();

				}
				/* 1. Resample mB minibatch from B images xTrain
				 * 2. Match mB minibatch from B labels zTrain
				 *
				 */
			}

			forward<DEBUG_E>(curr, next, l);
			compute<DEBUG_E>(next, l + 1);
			backward<DEBUG_E>(next, curr, l);

		}

	}

	template<class Layers_t, class Linked_Ptr_t, class Engine_Ptr, class Datain_Ptr, class Datagen_Ptr>
	template<debug_e DEBUG_E>
	auto Network<Layers_t, Linked_Ptr_t, Engine_Ptr, Datain_Ptr, Datagen_Ptr>
	::compute() {

		if constexpr (DEBUG_E > DEBUG0) {
			std::cout << "#### LAYERS : COMPUTE : START >>>" << std::endl;
			type_assert_ptr(Linked_Ptr_t);
		}

		compute<DEBUG_E>(linkedPtr, 0);

	}

	template<class Layers_t, class Linked_Ptr, class Engine_Ptr, class Datain_Ptr, class Datagen_Ptr>
	template<debug_e DEBUG_E, class Curr_Linked_Ptr_t>
	auto Network<Layers_t, Linked_Ptr, Engine_Ptr, Datain_Ptr, Datagen_Ptr>
	::init_parameters(Curr_Linked_Ptr_t currLinkedPtr) {

		using Curr_Linked_t =  std::remove_pointer_t<Curr_Linked_Ptr_t>;
		constexpr size_t N = Curr_Linked_t::N;

		T mean = 0;
		T var = T{1} / T{N};
		var = var > 0 ? var : 1;
		T stdev = std::sqrt(var);

		if constexpr (DEBUG_E > DEBUG0) std::cout << "mean = " << mean << ", stdev = " << stdev << std::endl;

		if constexpr (DEBUG_E > DEBUG0)std::cout << "init random weights DEBUG_E = " << DEBUG_E << std::endl;

		if constexpr (DEBUG_E >= DEBUG0) {
			datagenPtr.generate(currLinkedPtr->W_begin(), currLinkedPtr->W_end(), mean, stdev);
		}

		if constexpr (DEBUG_E >= DEBUG1) {
			datagenPtr.validate(currLinkedPtr->W_begin(), currLinkedPtr->W_end(), mean, stdev);
		}

	}

	template<class Layers_t, class Linked_Ptr_t, class Engine_Ptr, class Datain_Ptr, class Datagen_Ptr>
	template<debug_e DEBUG_E, class Curr_Linked_Ptr_t>
	auto Network<Layers_t, Linked_Ptr_t, Engine_Ptr, Datain_Ptr, Datagen_Ptr>
	::init_patterns(Curr_Linked_Ptr_t currLinkedPtr) {

		using Curr_Linked_t =  std::remove_pointer_t<Curr_Linked_Ptr_t>;

		if constexpr(DEBUG_E >= DEBUG0) {

			if constexpr (DEBUG_E > DEBUG0) std::cout << "... <DEBUG 0> init patterns" << std::endl;

			std::transform(
					datainPtr.cbegin_xtrain(),
					datainPtr.cend_xtrain(),
					currLinkedPtr->V_begin(),
					[&](const auto &x) -> T_IN { return static_cast<T_IN>(x) / T_IN{255}; }
			);

		}

		if constexpr(DEBUG_E >= DEBUG1) {
			//datagenPtr.in.validate(feed->begin(), feed->end(), -1, 1);

			auto sum = std::accumulate(currLinkedPtr->V_begin(), currLinkedPtr->V_end(), T_IN{0});

			std::cout << "... <DEBUG 1> : INIT PATTERNS : SUM = " << sum << std::endl;
		}

	}

	template<class Layers_t, class Linked_Ptr_t, class Engine_Ptr, class Datain_Ptr, class Datagen_Ptr>
	template<debug_e DEBUG_E, class Curr_Linked_Ptr_t>
	auto Network<Layers_t, Linked_Ptr_t, Engine_Ptr, Datain_Ptr, Datagen_Ptr>
	::init_classifiers(Curr_Linked_Ptr_t currLinkedPtr) {

		using Curr_Linked_t =  std::remove_pointer_t<Curr_Linked_Ptr_t>;
		using T_IN = typename Layers_t::m_T_IN;
		using T = typename Layers_t::m_T;

		if constexpr(DEBUG_E >= DEBUG0) {
			if constexpr (DEBUG_E > DEBUG0)std::cout << "... <DEBUG 0> init patterns" << std::endl;
			std::transform(
					datainPtr.cbegin_ztrain(),
					datainPtr.cend_ztrain(),
					currLinkedPtr->Z_begin(),
					[&](const auto &x) { return static_cast<T>(x); }
			);
		}

		if constexpr(DEBUG_E >= DEBUG1) {
			auto sum = std::accumulate(currLinkedPtr->Z_begin(), currLinkedPtr->Z_end(), T{0});
			std::cout << "... <DEBUG 1> : INIT CLASSIFIERS : SUM = " << sum << std::endl;
		}

	}

	template<class Layers_t, class Linked_Ptr_t, class Engine_Ptr, class Datain_Ptr, class Datagen_Ptr>
	template<debug_e DEBUG_E, class Curr_Linked_Ptr_t>
	auto Network<Layers_t, Linked_Ptr_t, Engine_Ptr, Datain_Ptr, Datagen_Ptr>
	::init(Curr_Linked_Ptr_t curr, size_t l) {

		using Curr_Linked_t = std::remove_pointer_t<Curr_Linked_Ptr_t>;
		constexpr bool HAS_NEXT = Curr_Linked_t::HAS_NEXT;
		constexpr layers_e EID = Curr_Linked_t::EID;

		if constexpr (HAS_NEXT) {

			auto next = curr->get_next();

			if constexpr (DEBUG_E > DEBUG0) std::cout << "FORWARD: curr : " << curr->get_name() << std::endl;

			if constexpr(EID == LAYER_INPUT) {
				if constexpr (DEBUG_E > DEBUG0)std::cout << "...INPUT" << std::endl;
				init_patterns<DEBUG_E>(curr);

			} else if constexpr (EID == LAYER_HIDDEN) {
				if constexpr (DEBUG_E > DEBUG0)std::cout << "...HIDDEN" << std::endl;
				init_parameters<DEBUG_E>(curr);
			}

			init<DEBUG_E>(next, l + 1);

			if constexpr (DEBUG_E > DEBUG0)std::cout << "BACKWARD: curr : " << curr->get_name() << std::endl;

			if constexpr(EID == LAYER_INPUT) {
				if constexpr (DEBUG_E > DEBUG0)std::cout << "...INPUT" << std::endl;
			} else if constexpr (EID == LAYER_HIDDEN) {
				if constexpr (DEBUG_E > DEBUG0)std::cout << "...HIDDEN" << std::endl;
			}

		} else {

			if constexpr (DEBUG_E > DEBUG0) std::cout << "MIDDLE: is last : " << curr->get_name() << std::endl;
			if constexpr (EID == LAYER_OUTPUT) {
				if constexpr (DEBUG_E > DEBUG0) std::cout << "...OUTPUT" << std::endl;
				init_classifiers<DEBUG_E>(curr);
				init_parameters<DEBUG_E>(curr);
			}
		}

	}

	template<class Layers_t, class Linked_Ptr_t, class Engine_Ptr, class Datain_Ptr, class Datagen_Ptr>
	template<debug_e DEBUG_E>
	auto Network<Layers_t, Linked_Ptr_t, Engine_Ptr, Datain_Ptr, Datagen_Ptr>
	::init() {

		if constexpr (DEBUG_E > DEBUG0) {
			std::cout << "#### LAYERS : INIT : START >>>" << std::endl;
			type_assert_ptr(Linked_Ptr_t);
			//type_assert_ptr(Datagen_Ptr);
			std::cout << "... init input data" << std::endl;

			linkedPtr->print_name();
		}

		init<DEBUG_E>(linkedPtr, 0);

		return this;

	}

	template<class Layers_t, class Linked_Ptr_t, class Engine_Ptr, class Datain_Ptr, class Datagen_Ptr>
	void Network<Layers_t, Linked_Ptr_t, Engine_Ptr, Datain_Ptr, Datagen_Ptr>
	::print() {

		layers.print(linkedPtr);

	}

	template<class Layers_t, class Linked_Ptr_t, class Engine_Ptr, class Datain_Ptr, class Datagen_Ptr>
	template<debug_e DEBUG_E, class Prev_Linked_Ptr_t, class Next_Linked_Ptr_t>
	auto
	Network<Layers_t, Linked_Ptr_t, Engine_Ptr, Datain_Ptr, Datagen_Ptr>
	::deltas(Prev_Linked_Ptr_t prevLinkedPtr, Next_Linked_Ptr_t nextLinkedPtr, size_t l) {

		return nullptr;
	}


};


#endif //CPP_EXAMPLE_NNET_HXX
