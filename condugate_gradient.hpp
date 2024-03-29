/*! @file

    @brief A collection of condugate gradient and preconditioned condugate gradient methods.
*/


#include <assert.h>
#include "linalgcpp.hpp"
#include "parallel_utility.hpp"

using namespace linalgcpp;


Vector<double> entrywise_mult(const Vector<double>& a, const Vector<double>& b){
	//assert (a.size()==b.size());
	Vector<double> c(a.size());
	for(int k=0;k<a.size();++k){
		c[k]=a[k]*b[k];
	}
	return c;
}

Vector<double> entrywise_inv(const Vector<double>& a){
	//assert (a.size()==b.size());
	Vector<double> c(a.size());
	for(int k=0;k<a.size();++k){
		c[k]=1.0/a[k];
	}
	return c;
}

//M=D+L, the lower triangular system (forward Gauss-Seidel)
Vector<double> DLsolver(const SparseMatrix<double>& M, Vector<double> b){
	//assert(M.Rows()==M.Cols()&&M.Rows()==b.size());
	for(int i=0;i<M.Rows();++i){
		std::vector<int> indices = M.GetIndices(i);
		std::vector<double> data = M.GetData(i);
		double sum=0;
		double pivot;
		for(int j=0;j<indices.size();++j){
			if(indices[j]<i){
				sum+=b[indices[j]]*data[j];
			}
			if(indices[j]==i){
				pivot=data[j];
			}
		}
		b[i]-=sum;
		b[i]/=pivot;
	}
	return b;
}

//solve the upper triangular system (backward Gauss-Seidel)
Vector<double> DUsolver(const SparseMatrix<double>& MT, Vector<double> b){
	//assert(MT.Rows()==MT.Cols()&&MT.Rows()==b.size());
	for(int i=MT.Rows()-1;i>=0;--i){
		std::vector<int> indices = MT.GetIndices(i);
		std::vector<double> data = MT.GetData(i);
		double sum=0;
		double pivot;
		for(int j=0;j<indices.size();++j){
			if(indices[j]>i){
				sum+=b[indices[j]]*data[j];
			}
			if(indices[j]==i){
				pivot=data[j];
			}
		}
		b[i]-=sum;
		b[i]/=pivot;
	}
	return b;
}


/*! @brief Solve system Mx=r, where M is the symmetric Gauss-Seidel matrix of A, in O(# of non-zero entries in A)

    @param A the sparse maxtrix from which we generate M
    @param r the right-hand-side of the system
*/
Vector<double> Solve_Gauss_Seidel(const SparseMatrix<double>& A, Vector<double> r){
	int n = A.Cols();
	
	//step 1: solve the lower triangular system for y: (D+L)y=r
	r=DLsolver(A,r);
	
	//step 2: solve the upper triangular system for x: (D+U)x=Dy
	r=entrywise_mult(Vector<double>(&A.GetDiag()[0],n),r);
	return DUsolver(A,r);
	
}

Vector<double> Solve_Jacobian(const SparseMatrix<double>& A, Vector<double> r){
	//assert(M.Rows()==M.Cols()&&M.Rows()==b.size());
	int n = A.Cols();
	Vector<double> diag(&A.GetDiag()[0],n);
	for(int i=0;i<n;++i){
		r[i]/=diag[i];
	}
	return r;
}


Vector<double> PCG(const SparseMatrix<double>& A, const Vector<double>& b, Vector<double>(*Msolver)(const SparseMatrix<double>& , Vector<double>),int max_iter,double tol){
	//level of difficulty: medium				    
	//assert A is s.p.d.
	
    int n = A.Cols();
	
    Vector<double> x(n,0.0);
    Vector<double> r(b);
	Vector<double> pr = Msolver(A,r);
    Vector<double> p(pr);
    Vector<double> g(n);
    double delta0 = r.Mult(pr);
    double delta = delta0, deltaOld, tau, alpha;

    for(int k=0;k<max_iter;k++){
        g = A.Mult(p);
        tau = p.Mult(g);
        alpha = delta / tau;
        x = x + (alpha * p);
        //x.Print("x at iteration: "+std::to_string(k));
        r = r - (alpha * g);
        pr = Msolver(A,r);
		deltaOld = delta;
		delta = r.Mult(pr);
		//std::cout<<"delta at iteration "<<k<<" is "<<delta<<std::endl;
        if(delta < tol * tol * delta0){
            std::cout<<"converge at iteration "<<k<<std::endl;
            return x;
        }
        p = pr + ((delta / deltaOld)* p);
    }
	
    std::cout<<"failed to converge in "<<max_iter<<" iterations"<<std::endl;
    return x;
	
}

/*! @brief The regular condugate gradient method, time complexity O(max_iter*N^2)

    @param A an s.p.d. matrix
    @param b the right-hand-side of system
    @param max_iter maximum number of iteration before exit
	@param tol epsilon
*/
Vector<double> CG(const SparseMatrix<double>& A, const Vector<double>& b, int max_iter,double tol, bool para){
    //assert A is s.p.d.
    int n = A.Cols();
    Vector<double> x(n,0.0);
    Vector<double> r(b);
    Vector<double> p(r);
    Vector<double> g(n);
    double delta0 = b.Mult(b);
    double delta = delta0, deltaOld, tau, alpha;

    for(int k=0;k<max_iter;k++){
		if(para)g = paraMult(A,p);
		else g = Mult(A,p);
        tau = p.Mult(g);
        alpha = delta / tau;
        x = x + (alpha * p);
        //x.Print("x at iteration: "+std::to_string(k));
        r = r - (alpha * g);
        deltaOld = delta;
		delta = r.Mult(r);
		//std::cout<<"delta at iteration "<<k<<" is "<<delta<<std::endl;
    
        if(delta < tol * tol * delta0){
            //std::cout<<"converge at iteration "<<k<<std::endl;
            return x;
        }
        p = r + ((delta / deltaOld)* p);
    }
	
    std::cout<<"failed to converge in "<<max_iter<<" iterations"<<std::endl;
    return x;
}

