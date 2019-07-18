//
// Created by a on 3/22/19.
//

#include <vector>
#include <iostream>
#include <map>
#include <cstring>
#include <list>
#include <memory>
#include <chrono>
#include <thread>
#include <cassert>

#define DEFN 10000


template<typename T>
std::unique_ptr<T> func_make_unique_new_ptr(std::unique_ptr<T> uniqPtr, std::shared_ptr<T> sharPtr) {
	//using it as a parameter and return value NAME whenever you'randEng transferring exclusive ownership.
	return new T();
}

template<typename T>
std::shared_ptr<T> func_make_shared_new_ptr() {
	//use it for nonexclusive ownership with reference counting of pointer
	return new T();
}

#define quote(x) #x

template<typename T>
void print_id(T A) {

	std::cout << "typeid().name() : " << typeid(A).name() << ", " << quote(A) << "\n";
}

class MyClassPublic {
  protected:
	int protPub = 12345;

  public:
	int pubPub = 123;
};

class MyClassProtected {
  protected:
	int protProt = -12345;

  public:
	int pubProt = -123;
};

class MyClassSimple {
	int a;

  public:
	MyClassSimple(int a) : a(DEFN) {

		std::cout << "MyClassSimple : a(" << a << ")" << std::endl;
	}

	size_t size() {

		return sizeof *this;
	}

};

class MyClassContainer {
	int a;

	int A[DEFN];

  public:
	MyClassContainer(int a, int A_[DEFN]) : a(DEFN) {


		std::memset(A, DEFN, DEFN * sizeof(int));
		std::cout << "MyClassContainer : a(" << a << "), sizeof : " << sizeof(A) << ", " << sizeof(int) << std::endl;
	}

	size_t size() {

		return sizeof *this;

	}
};

class MyClassPointer {
	int a;

	int n;

	int *A;

  public:
	MyClassPointer(int a, int n, int *A_) : a(a), n(n), A(A_) {

		std::cout << "MyClassPointer : a(" << a << "), sizeof : "
		          << sizeof(A) << ", i : " << sizeof(int) << "/"
		          << sizeof(A[0]) << ", n : " << n << std::endl;
	}

	size_t size() {

		return size_heap() + size_stack();
	}

	size_t size_stack() {
		//static
		return sizeof *this;

	}

	size_t size_heap() {
		//dynamic
		return sizeof(A[0]) * n;
	}

};

template<typename T, unsigned int S>
class T_MyClassContainer {
	int a;

	T A[S];

  public:
	T_MyClassContainer(int a, T A[S]) : a(S) {

		std::memset(A, S, S * sizeof(T));
		std::cout << "T_MyClassContainer : a(" << a << "), sizeof : " << sizeof(A) << ", " << sizeof(T) << std::endl;
	}

	size_t size() {

		return sizeof *this;

	}
};

template<typename T, unsigned int S>
class T_MyClassPointer {
	int a;

	T *A;

  public:
	T_MyClassPointer(int a, T *A) : a(a), A(A) {

		std::memset(A, S, S * sizeof(T));
		std::cout << "T_MyClassPointer : a(" << a << "), sizeof : " << sizeof(A) << ", " << sizeof(T) << std::endl;

	}

	size_t size() {

		return size_stack() + size_heap();
	}

	size_t size_stack() {

		return sizeof *this;

	}

	size_t size_heap() {

		return sizeof(A[0]) * S;
	}

};

template<typename T>
int set_layers_simple(std::vector<size_t> layersSizes) {

	std::map<size_t, std::vector<T>> layersMap;

	for (const auto &l : layersSizes) {
		layersMap[l] = std::vector<T>(l);

	}

	for (const auto &l : layersMap) {
		std::cout << l.first << "," << l.second.size() << std::endl;
	}

	return 0;

}

template<typename T, typename P = std::unique_ptr<T>>
auto set_layers_polymorphed(std::vector<size_t> layersSizes) {

	std::map<size_t, std::vector<P>> layersMap;

	for (const auto &l : layersSizes) {
		layersMap[l] = std::vector<P>(l);
	}


	return layersMap;

}

int test_vectors() {
/*
	auto *A = (int *) std::malloc(sizeof(int) * DEFN);

	std::vector<int> Vint(1);
	std::vector<MyClassSimple> Vmcs;
	Vmcs = {MyClassSimple(1)};

	std::vector<MyClassContainer> Vmcc;
	Vmcc = {MyClassContainer(2, A)};

	std::vector<MyClassPointer> Vmcp {MyClassPointer(3, DEFN, A)};

	std::vector<T_MyClassContainer<int, DEFN>> Vtmcc;
	Vtmcc = {T_MyClassContainer<int, DEFN>(4, A)};

	std::vector<T_MyClassPointer<int, DEFN>> Vtmcp;
	Vtmcp = {T_MyClassPointer<int, DEFN>(5, A)};

	printf("sizeof : Vmcs(%zu), Vmcc(%zu), Vmcp(%zu), Vtmcc(%zu), Vtmcp(%zu)\n",
	       sizeof(Vmcs), sizeof(Vmcc), sizeof(Vmcp), sizeof(Vtmcc), sizeof(Vtmcp));
	printf("sizeof v[0] : Vmcs(%zu), Vmcc(%zu), Vmcp(%zu), Vtmcc(%zu), Vtmcp(%zu)\n",
	       sizeof(Vmcs[0]), sizeof(Vmcc[0]), sizeof(Vmcp[0]), sizeof(Vtmcc[0]), sizeof(Vtmcp[0]));
	printf("v.size() : Vmcs(%zu), Vmcc(%zu), Vmcp(%zu), Vtmcc(%zu), Vtmcp(%zu)\n",
	       Vmcs.size(), Vmcc.size(), Vmcp.size(), Vtmcc.size(), Vtmcp.size());
	printf("v[0].size() : Vmcs(%zu), Vmcc(%zu), Vmcp(%zu), Vtmcc(%zu), Vtmcp(%zu)\n",
	       Vmcs[0].size(), Vmcc[0].size(), Vmcp[0].size(), Vtmcc[0].size(), Vtmcp[0].size());

	std::vector<sint> layersSizes = {100, 100, 30, 10};

	set_layers_simple<sint>(layersSizes);

	auto mapLayers = set_layers_polymorphed<MyClassSimple>(layersSizes);

	for (auto &layer : mapLayers) {
		std::cout << layer.first << "," << layer.second.size() << std::endl;
	}
 */
}

// Base class has a pure virtual function for cloning
class LayerBase {


  public:

	std::string type;

	std::string name;

	std::pair<size_t, size_t> shape;

	//virtual ~LayerBase() {};

	//virtual LayerBase *clone() const = 0;

	virtual void print() {

		std::cout << name << std::endl;
	}

	LayerBase(size_t l, size_t N, size_t M) : shape {N, M} {

		if (N == 0) {
			type = "Input Layer";
		} else if (M == 0) {
			type = "Output Layer";
		} else {
			type = "Hidden Layer";
		}

		make_name(l);


		std::cout << "LayerBase : name = "
		          << name << std::endl;
	}

	void make_name(size_t l) {

		name = "[" + type + " : " + std::to_string(l) + "]:(" + std::to_string(shape.first) + "," +
		       std::to_string(shape.second) + ")";
	}

};

template<class Base, class Derived>
class CRTP : public Base {

};

// This CRTP class implements clone() for Derived
template<class Derived>
class LayerBaseCRTP : public LayerBase {

  public:


	virtual Derived *caster() const {

		std::cout << "Cloning!" << std::endl;
		//return static_cast<Derived const &>(*this);
	}


	//LayerBase_CRTP(size_t l, size_t N, size_t M) : LayerBase(l, N, M) {}

	void interface() {

		static_cast<Derived *>(this)->implementation();
	}

	void implementation() {

		std::cout << "Implementation Base" << std::endl;
	}

};

template<typename T, size_t mB, size_t N, size_t M>
//class LayerHidden : public LayerBaseCRTP<LayerHidden<T, N, M, mB>> {
class LayerHiddenLast;

#define Derive_LayerBase_CRTP(Type, T, ...) Type: public LayerBase_CRTP<Type>
#define sint size_t

class LayersBase {

  public:

	sint nFullBatch = 0;

	sint nMiniBatch = 0;

	sint nLayers = 0;

	sint nHidden = 0;

	//can not be LayerBase
	//has to use LayersHidden
	//or try lazy

	std::map<size_t, std::pair<const size_t, const size_t> *> layersMap;

	//std::map<size_t, LayerBase*> layersTraining;
	std::vector<LayerBase *> layersTraining;

	//std::map<size_t, LayerBase*> layersValidation;
	std::vector<LayerBase *> layersValidation;
};


template<typename T, size_t N, size_t ...>
class LayerOutput : public LayerBase {
  public:

	std::array<T, N * 1> V;

  public:


	LayerOutput(size_t id) : LayerBase(id, N, 0) {


		std::cout << "make output : " << id << ", : N = " << N << std::endl;

	}

};


template<typename T, size_t mB, size_t N, size_t M, size_t ... shapeNN>
//class LayerHidden : public LayerBaseCRTP<LayerHidden<T, N, M, mB>> {
class LayerHidden : public LayerBase {
  public:
	std::array<T, M * N> W;

	std::array<T, M * N> dW;

	std::array<T, M * mB> V;

	std::array<T, M * mB> dV;

	std::array<T, M * mB> b;

	std::array<T, M * mB> Delta;

	std::array<T, M * 1> Theta;

	std::array<T, M * 1> dTheta;

	LayerHidden<T, mB, M, shapeNN..., 0> * LH;

	LayerOutput<T, M> * LO;

	const char type;

  public:
	//LayerHidden(size_t id) {}

	void print() {

		LayerBase::print();

		switch (type) {
			case 'h': {
				LH->print();
				break;
			}
			case 'o': {
				LO->print();
				break;
			}
		}

	}


	LayerHidden(size_t id, void * L, const char type_) :
			type(type_),
			LayerBase(id, N, M),
			LH(type_ == 'h' ? static_cast< decltype(LH) > (L) : nullptr),
			LO(type_ == 'o' ? static_cast< decltype(LO) > (L) : nullptr) {


		switch (type) {
			case 'h': {
				assert(LH != nullptr);
				std::cout << "make hidden<next : hidden> : " << id << ", : N = " << N << ", M = " << M << std::endl;
				break;
			}
			case 'o': {
				assert(LO != nullptr);
				std::cout << "make hidden<next : output> : " << id << ", : N = " << N << ", M = " << M << std::endl;
				break;
			}
		}

	}

};


template<typename T, size_t mB, size_t N, size_t ...shapeNN>
class LayerInput : public LayerBase {


  public:
	std::array<T, N * mB> V;

	LayerHidden<T, mB, N, shapeNN...> * LH;

  public:

	void print() {

		LayerBase::print();
		LH->print();
	}

	LayerInput(size_t id, decltype(LH) LH) : LH(LH), LayerBase(id, 0, N) {

		assert(LH != nullptr);
		std::cout << "make input : " << id << ", : N = " << N << std::endl;
	}

};


template<typename T, size_t B, size_t mB, size_t I, size_t ... shapeNN>
class Layers : public LayersBase {


	std::shared_ptr<LayerInput<T, mB, I, shapeNN...>> layerInput;

  public:


	template<size_t M>
	auto make_next_layer(size_t l) {

		std::cout << "last layer : " << l << "/" << nLayers << " : " << " M = " << M << std::endl;

		return new LayerOutput<T, M>(l);

	}


	template<size_t N, size_t M, size_t ... nextShapeNN>
	auto make_next_layer(size_t l) {

		size_t lenNextShapeNN = sizeof ... (nextShapeNN);

		if (lenNextShapeNN > 0) {
			std::cout << "make layer : " << l << "/" << nLayers << " : N = " << N << ", M = " << M << ", len(nextNN) = "
			          << sizeof ... (nextShapeNN) << std::endl;

			auto NL = make_next_layer<M, nextShapeNN...>(l + 1);
			assert(NL != nullptr);

			std::cout << "here should be hidden next..." << std::endl;
			return new LayerHidden<T, mB, N, M, nextShapeNN...>(l, NL, 'h');
		} else {
			std::cout << "pre last layer : " << l << "/" << nLayers << " : N = " << N << ", M = " << M
			          << ", len(nextNN) = "
			          << sizeof ... (nextShapeNN) << std::endl;
			auto NL = make_next_layer<M>(l + 1);
			assert(NL != nullptr);
			std::cout << "here should be output next..." << std::endl;
			return new LayerHidden<T, mB, N, M, nextShapeNN...>(l, NL, 'o');

		}

	}

	Layers() {

		std::cout << " LayersMaker... " << std::endl;
		this->nHidden = sizeof ... (shapeNN);
		this->nLayers = this->nHidden + 1;

		this->nMiniBatch = mB;
		this->nFullBatch = B;
		auto NL = make_next_layer<I, shapeNN...>(1);
		layerInput = std::make_shared<LayerInput<T, mB, I, shapeNN...>>(0, std::move(NL));

		layerInput->print();

	}

};


int main() {

	const size_t B = 100;
	const size_t mB = 10;

	LayersBase layers = Layers<int, B, mB, 1000, 100, 100, 30, 10>();

	//auto * lh = (LayerHidden<int, 100, 100, mB> * ) layersWrap.layersTraining[1];
	//lh->V;


	std::this_thread::sleep_for(std::chrono::milliseconds(1000));

	return 0;
}
