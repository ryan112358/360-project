#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
 
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
* LU Factorization with loop unrolling and vectorization
* Compiler options
* Intel compiler: icc -xHOST -O3 
* GCC: gcc -march=native -O3
*/
void lu_v2(double **A, double **L, double **U, int n) {
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
			#pragma vector always
			#pragma unroll 4
			for(int k=j; k < n; k++)
				U[i][k] -= m*U[j][k];
		}
	}
}

void check_LU(double **A, int n) {
	double **L = new_matrix(n);
	double **U = new_matrix(n);
	lu(A,L,U,n);
	double **LU = new_matrix(n);
	mult(L,U,LU,n);
	for(int i=0; i < n; i++)
		for(int j=0; j < n; j++)
			if(abs(A[i][j] - LU[i][j]) > 0.00000001) {
				printf("Test Failed!! \n");
				show(A,n); show(L,n); show(U,n);
			}
	printf("Test Passed!! \n");
}

void benchmark_LU(int n, int trials){
	clock_t t = 0;
	double **L = new_matrix(n);
	double **U = new_matrix(n);
	for(int i=0; i < trials; i++) {
		double **A = rand_matrix(n);
		t -= clock();
		lu(A,L,U,n);
		t += clock();
		free(A);
	}
	free(L); free(U);
	printf( "A matrix of size %d x %d took an average of %f seconds to factor.\n", n, n, ((float)t)/(trials*CLOCKS_PER_SEC)); 
}

void benchmark() {
	for(int n=64; n <= 2048; n*=2) {
		benchmark_LU(n, 1);
	}
}
  
int main() {
	double **A = rand_matrix(3);
	double **L = new_matrix(3);
	double **U = new_matrix(3);
	
	lu(A,L,U,3);
	check_LU(A,3);

	//show(A,3);
	//show(L,3);
	//show(U,3);
	
	benchmark();
	
	return 0;
}
