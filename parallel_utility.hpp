#include <omp.h>
#include <assert.h>
#include "linalgcpp.hpp"

using namespace linalgcpp;

Vector<double> Mult(const SparseMatrix<double>& A, const Vector<double>& b){
	Vector<double> Ab(A.Rows());
	#pragma omp parallel for
	for(int i=0;i<A.Rows();i++){
		std::vector<int> indices = A.GetIndices(i);
		std::vector<double> data = A.GetData(i);
		double sum=0.0;
		for(int j=0;j<data.size();j++){
			sum+=data[i]*b[indices[i]];
		}
		Ab[i]=sum;
	}
	return Ab;
}