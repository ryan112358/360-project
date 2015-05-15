#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

typedef void (solver)(double**, double**, double**, int);
 
void zero(double **A, int n) { 
	for(int i=0; i < n; i++)
		for(int j=0; j < n; j++)
			A[i][j] = 0; 
}

double** new_matrix(int n) {
	double **A = malloc(sizeof(double*) * n);
	A[0]  = malloc(sizeof(double) * n * n);
 
	for(int i=0; i < n; i++)
		A[i] = A[0] + n*i;
	return A;
}

double** rand_matrix(int n) {
	double **A = new_matrix(n);
	for(int i=0; i < n; i++)
		for(int j=0; j < n; j++)
			A[i][j] = (double) rand() / (double) (RAND_MAX / 100);
	return A;
}

void show(double **A, int n) {
	for(int i=0; i < n; i++) {
		for(int j=0; j < n; j++)
			printf("%.3f ",A[i][j]);
		printf("\n");
	}
	printf("\n");
}

void mult(double **A, double **B, double **C, int n) {
	for(int i=0; i < n; i++)
		for(int j=0; j < n; j++) {
			double val = 0;
			for(int k=0; k < n; k++)
				val += A[i][k] * B[k][j];
			C[i][j] = val;
		}
}

void lu(double **A, double **L, double **U, int n) {
	zero(L,n); zero(U,n);
	
	for(int i=0; i < n; i++)
		for(int j=0; j < n; j++)
			U[i][j] = A[i][j];
 
	for(int i=0; i < n; i++)
		L[i][i] = 1;
	
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
void luParallel(double **A, double **L, double **U, int n) {
	zero(L,n); zero(U,n);
	
	for(int i=0; i < n; i++)
		for(int j=0; j < n; j++)
			U[i][j] = A[i][j];
 
	for(int i=0; i < n; i++)
		L[i][i] = 1;
	
	for(int j=0; j < n; j++){
		// this loop can be parallelized
		#pragma omp parallel for schedule(static, 8) 
		for(int i=j+1; i < n; i++) {
			double m = U[i][j] / U[j][j];
			L[i][j] = m;
			// this loop can also be parallelized
			for(int k=j; k < n; k++)
				U[i][k] -= m*U[j][k];
		}
	}
}

void luAll(double **A, double **L, double **U, int n) {
	zero(L,n); zero(U,n);
	
	for(int i=0; i < n; i++)
		for(int j=0; j < n; j++)
			U[i][j] = A[i][j];
 
	for(int i=0; i < n; i++)
		L[i][i] = 1;
	
	for(int j=0; j < n; j++){
		#pragma omp parallel for schedule(static, 8) 
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

void luUnroll(double **A, double **L, double **U, int n) {
	zero(L,n); zero(U,n);
	
	for(int i=0; i < n; i++)
		for(int j=0; j < n; j++)
			U[i][j] = A[i][j];
 
	for(int i=0; i < n; i++)
		L[i][i] = 1;
	
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


void luVec(double **A, double **L, double **U, int n) {
	zero(L,n); zero(U,n);
	
	for(int i=0; i < n; i++)
		for(int j=0; j < n; j++)
			U[i][j] = A[i][j];
 
	for(int i=0; i < n; i++)
		L[i][i] = 1;
	
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
	zero(L,n); zero(U,n);
	
	for(int i=0; i < n; i++)
		for(int j=0; j < n; j++)
			U[i][j] = A[i][j];
 
	for(int i=0; i < n; i++)
		L[i][i] = 1;
	
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
	clock_t t = 0;
	double **L = new_matrix(n);
	double **U = new_matrix(n);
	for(int i=0; i < trials; i++) {
		double **A = rand_matrix(n);
		t -= clock();
		(*f)(A,L,U,n);
		t += clock();
		free(A);
	}
	free(L); free(U);
	//printf( "A matrix of size %d x %d took an average of %f seconds to factor.\n", n, n, ((float)t)/(trials*CLOCKS_PER_SEC)); 
	return (float) t/(trials*CLOCKS_PER_SEC);
}

void benchmark() {
	solver *solvers[6] = { &lu, &luVec, &luUnroll, &luUnrollVec, &luParallel, &luAll };
	for(int i=0; i < 6; i++)
		check_LU(solvers[i],25);	
	printf("n, lu, luVec, luUnroll, luVecUnroll, luParallel, luAll\n");
	for(int n=64; n <= 4096; n*=2) {
		printf("%d, ", n);
		for(int s=0; s < 6; s++) {
			float t = benchmark_LU(solvers[s], n, 1);
			printf("%f, ",t);
		}
		printf("\n");
	}
}

	
  
int main() {
	check_LU(&lu,3);

	//show(A,3);
	//show(L,3);
	//show(U,3);
	
	benchmark();
	
	return 0;
}
