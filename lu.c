#include <stdio.h>
#include <stdlib.h>
#include <math.h>
 
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
  
int main() {
	double **A = rand_matrix(3);
	double **L = new_matrix(3);
	double **U = new_matrix(3);
	
	lu(A,L,U,3);
	check_LU(A,3);

	//show(A,3);
	//show(L,3);
	//show(U,3);
	
	return 0;
}
