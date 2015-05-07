#include <stdio.h>
#include <stdlib.h>
#include <math.h>
 
void zero(double **A, int n) { 
	for(int i=0; i < n; i++)
		for(int j=0; j < n; j++)
			A[i][j] = 0; 
}

void show(double **A, int n) {
	for(int i=0; i < n; i++) {
		for(int j=0; j < n; j++)
			printf("%f.3 ", A[i][j]);
		printf("\n");
	}
}

void mult(double **A, double **B, double **C, int n) {
	for(int i=0; i < n; i++)
		for(int j=0; j < n; j++) {
			C[i][j] = 0;
			for(int k=0; k < n; k++)
				C[i][j] += A[i][k] * B[k][j];
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
  
int main() {
	double A[3][3] = { {1,2,3}, {4,5,6}, {7,8,9} };
	double L[3][3], U[3][3];
	
	lu((double**)A,(double**)L,(double**)U,3);
	
	show((double**)L,3);
	show((double**)U,3);
	
	return 0;
}
