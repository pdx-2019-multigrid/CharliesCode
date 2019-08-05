#include <omp.h>
#include <assert.h>
#include "linalgcpp.hpp"

using namespace linalgcpp;

/**
Vector<double> paraMult(const SparseMatrix<double>& A, const Vector<double>& b){
    const int aRows = A.Rows();
    Vector<double> Ab(aRows);
    using std::vector;
    vector<vector<int>> indArr(aRows);
    vector<vector<double>> datArr(aRows);
    vector<int> dataSizes(aRows);

    #pragma omp parallel for schedule(static)
    for(int i = 0; i < aRows; i++){
        indArr[i] = A.GetIndices(i);
        datArr[i] = A.GetData(i);
        dataSizes[i] = datArr[i].size();
    }
    #pragma omp parallel for schedule(dynamic,1)
    for(int j = 0; j < dataSizes[i]; j++){
		for(int i = 0; i < aRows; i++){
            Ab[i] += datArr[i][j] * b[indArr[i][j]];
        }
    }
    return Ab;
}

Vector<double> paraMult(const SparseMatrix<double>& A, const Vector<double>& b){
	Vector<double> Ab(A.Rows());
	std::vector<int> indArr[A.Rows()];
	std::vector<double> datArr[A.Rows()];
	#pragma omp parallel for schedule(dynamic)
	for(int i=0;i<A.Rows();i++){
		indArr[i] = A.GetIndices(i);
		datArr[i] = A.GetData(i);
	}
	
	#pragma omp parallel for
	for(int i=0;i<A.Rows();i++){
		double sum=0.0;
		for(int j=0;j<datArr[i].size();j++){
			sum+=datArr[i][j]*b[indArr[i][j]];
		}
		Ab[i]=sum;
	}
	return Ab;
}

*/

//==================================SparseMatrix=======================================================
Vector<double> paraMult(const SparseMatrix<double>& A, const Vector<double>& b){
	int M = A.Rows();
	Vector<double> Ab(M);
	#pragma omp parallel for
	for(int i=0;i<M;i++){
		std::vector<int> indices = A.GetIndices(i);
		std::vector<double> data = A.GetData(i);
		double sum=0.0;
		int N = data.size();
		for(int j=0;j<N;j++){
			sum+=data[j]*b[indices[j]];
		}
		Ab[i]=sum;
	}
	return Ab;
}

Vector<double> Mult(const SparseMatrix<double>& A, const Vector<double>& b){
	int M = A.Rows();
	Vector<double> Ab(M);
	for(int i=0;i<M;i++){
		std::vector<int> indices = A.GetIndices(i);
		std::vector<double> data = A.GetData(i);
		double sum=0.0;
		int N = data.size();
		for(int j=0;j<N;j++){
			sum+=data[j]*b[indices[j]];
		}
		Ab[i]=sum;
	}
	return Ab;
}

//==================================DenseMatrix=======================================

Vector<double> ParaMult(const DenseMatrix& A, const Vector<double>& b){
	int M=A.Rows();
	int N=A.Cols();
	Vector<double> Ab(M);
	#pragma omp parallel for
	for(int i=0;i<M;i++){
		double sum=0.0;
		for(int j=0;j<N;j++){
			sum+=A(i,j)*b[j];
		}
		Ab[i]=sum;
	}
	return Ab;
}

Vector<double> Mult(const DenseMatrix& A, const Vector<double>& b){
	int M=A.Rows();
	int N=A.Cols();
	Vector<double> Ab(M);
	for(int i=0;i<M;i++){
		for(int j=0;j<N;j++){
			Ab[i]+=A(i,j)*b[j];
		}
	}
	return Ab;
}

/**
TODO:
A^Tx
AB
A^TB
*/
