#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>

typedef void (solver)(double**, double**, double**, int);

void zero(double **A, int n) {
	memset(&A[0][0], 0, sizeof(double) * n * n);
}

void copy(double** dest, double** src, int n) {
	memcpy(&dest[0][0], &src[0][0], sizeof(double) * n * n);
}

void init(double** A, int n) {
	for (int i = 0; i < n; ++i) 
		A[i][i] = 1;
}

/**
 * creates a new matrix of size n x n
 */
double** new_matrix(int n) {
	double **A = (double**) malloc(sizeof(double*) * n);
	A[0]  = (double*) malloc(sizeof(double) * n * n);

	for(int i=0; i < n; i++)
		A[i] = A[0] + n*i;
	return A;
}

/**
 * creates an n x n matrix and fills it with random doubles
 */
double** rand_matrix(int n) {
	double **A = new_matrix(n);
	for(int i=0; i < n; i++)
		for(int j=0; j < n; j++)
			A[i][j] = (double) rand() / (double) (RAND_MAX / 100);
	return A;
}

/**
 * prints matrix A
 *
 * n = size of one dimension of A
 */
void show(double **A, int n) {
	for(int i=0; i < n; i++) {
		for(int j=0; j < n; j++)
			printf("%.3f ",A[i][j]);
		printf("\n");
	}
	printf("\n");
}

/**
 * multiplies matrices A and B, and stores the result in matrix C
 */
void mult(double **A, double **B, double **C, int n) {
	for(int i=0; i < n; i++)
		for(int j=0; j < n; j++) {
			double val = 0;
			for(int k=0; k < n; k++)
				val += A[i][k] * B[k][j];
			C[i][j] = val;
		}
}

/**
 * basic LU factorization
 * (no optimizations)
 */
void lu(double **A, double **L, double **U, int n) {

	zero (L, n);
	copy (U, A, n);
	init (L, n);

	for(int j=0; j < n; j++){
		// this loop can be parallelized
		for(int i=j+1; i < n; i++) {
			double m = U[i][j] / U[j][j];
			L[i][j] = m;
			// this loop can also be parallelized
			for(int k=j; k < n; k++)
				U[i][k] -= m*U[j][k];
		}
	}
}
/**
* GCC: gcc -o lu -fopenmp -std=c99 lu.c
*/

/**
 * LU factorization with parallel optimizations
 */
void luParallel(double **A, double **L, double **U, int n) {

	zero (L, n);
	copy (U, A, n);
	init (L, n);

	#pragma omp parallel
	{
		for(int j=0; j < n; j++) {
      			#pragma omp for
			for(int i=j+1; i < n; i++) {
				double m = U[i][j] / U[j][j];
				L[i][j] = m;
				for(int k=j; k < n; k++)
					U[i][k] -= m*U[j][k];
			}
			#pragma omp barrier
		}
	}
}

/**
 * LU factorization with all optimizations implemented
 * (parallel, loop unrolling, vectorization)
 */
void luAll(double **A, double **L, double **U, int n) {

	zero (L, n);
	copy (U, A, n);
	init (L, n);

 	 #pragma omp parallel
	{
		for(int j=0; j < n; j++) {
     			#pragma omp for
			for(int i=j+1; i < n; i++) {
				double m = U[i][j] / U[j][j];
				L[i][j] = m;
        			#pragma vector always
        			#pragma unroll 4
				for(int k=j; k < n; k++)
					U[i][k] -= m*U[j][k];
			}
			#pragma omp barrier
		}
	}
}

/**
 * LU factorization with loop unrolling
 */
void luUnroll(double **A, double **L, double **U, int n) {
	zero (L, n);
	copy (U, A, n);
	init (L, n);
	for(int j=0; j < n; j++){
		for(int i=j+1; i < n; i++) {
			double m = U[i][j] / U[j][j];
			L[i][j] = m;
			#pragma unroll 4
			for(int k=j; k < n; k++)
				U[i][k] -= m*U[j][k];
		}
	}
}

/**
 * LU factorization with vectorization
 */
void luVec(double **A, double **L, double **U, int n) {
	zero (L, n);
	copy (U, A, n);
	init (L, n);
	for(int j=0; j < n; j++){
		for(int i=j+1; i < n; i++) {
			double m = U[i][j] / U[j][j];
			L[i][j] = m;
			#pragma vector always
			for(int k=j; k < n; k++)
				U[i][k] -= m*U[j][k];
		}
	}
}

/**
* LU Factorization with loop unrolling and vectorization
* Compiler options
* Intel compiler: icc -xHOST -O3
* GCC: gcc -march=native -O3
*/
void luUnrollVec(double **A, double **L, double **U, int n) {
	zero (L, n);
	copy (U, A, n);
	init (L, n);
	for(int j=0; j < n; j++){
		for(int i=j+1; i < n; i++) {
			double m = U[i][j] / U[j][j];
			L[i][j] = m;
			#pragma vector always
			#pragma unroll 4
			for(int k=j; k < n; k++)
				U[i][k] -= m*U[j][k];
		}
	}
}

void check_LU(solver *f, int n) {
	double **A = rand_matrix(n);
	double **L = new_matrix(n);
	double **U = new_matrix(n);
	(*f)(A,L,U,n);
	double **LU = new_matrix(n);
	mult(L,U,LU,n);
	for(int i=0; i < n; i++)
		for(int j=0; j < n; j++)
			if(abs(A[i][j] - LU[i][j]) > 0.00000001) {
				printf("Test Failed!! \n");
				show(A,n); show(L,n); show(U,n);
			}
	free(L); free(U); free(A);
	printf("Test Passed!! \n");
}

float benchmark_LU(solver *f, int n, int trials){
	double t = 0;
	double **L = new_matrix(n);
	double **U = new_matrix(n);
	for(int i=0; i < trials; i++) {
		double **A = rand_matrix(n);
		t -= omp_get_wtime();
		(*f)(A,L,U,n);
		t += omp_get_wtime();
		free(A);
	}
	free(L); free(U);
	//printf( "A matrix of size %d x %d took an average of %f seconds to factor.\n", n, n, ((float)t)/(trials*CLOCKS_PER_SEC));
	return (float) t / trials;
}

void benchmark() {
	solver *solvers[6] = { &lu, &luVec, &luUnroll, &luUnrollVec, &luParallel, &luAll };
	for(int i=0; i < 6; i++)
		check_LU(solvers[i],25);
	printf("n, lu, luVec, luUnroll, luVecUnroll, luParallel, luAll\n");
	for(int n=100; n <= 2000; n*=2) {
		printf("%d, ", n);
		for(int s=0; s < 6; s++) {
			float t = benchmark_LU(solvers[s], n, 8);
			printf("%f, ",t);
		}
		printf("\n");
	}
}



int main() {
	benchmark();
	return 0;
}
