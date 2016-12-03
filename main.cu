#include <stdio.h>
#include <stdlib.h>

#define N 2

__global__ void MatAdd(int A[][N], int B[][N], int C[][N]){
           int i = threadIdx.x;
           int j = threadIdx.y;

           C[i][j] = A[i][j] + B[i][j];
       }

__global__ void MatMult(double A[][N], double B[][N], double C[][N], int width){
               int i = threadIdx.x;
               int j = threadIdx.y;
               int k;
               double Pvalue = 0;

               for(k = 0; k < width; k++){
                   Pvalue += A[i][k] * B[k][j];
               }

               C[i][j] = Pvalue;
}

__global__ void MatTrans(double A[][N], double C[][N], int width, int height){
               int i = threadIdx.x;
               int j = threadIdx.y;

               C[i][j] = A[j][i];

}

__global__ void multByValue(double val, double A[][N], double C[][N]) {
               int i = threadIdx.x;
               int j = threadIdx.y;
               C[i][j] = val * A[i][j];

}

__global__ void coFactor(double A[][N], double C[][N]) {
                  
}
int main(){

double A[3][2] = {{1,2},{3,4},{5,6}};
double B[N][N] = {{1,2},{3,4}};
double C[3][2] = {{0,0},{0,0},{0,0}};

double (*pA)[N], (*pB)[N], (*pC)[N];

cudaMalloc((void**)&pA, (3*2)*sizeof(double));
cudaMalloc((void**)&pB, (N*N)*sizeof(double));
cudaMalloc((void**)&pC, (3*2)*sizeof(double));

cudaMemcpy(pA, A, (3*2)*sizeof(double), cudaMemcpyHostToDevice);
cudaMemcpy(pB, B, (N*N)*sizeof(double), cudaMemcpyHostToDevice);
cudaMemcpy(pC, C, (3*2)*sizeof(double), cudaMemcpyHostToDevice);

int numBlocks = 1;
dim3 threadsPerBlock(3,2);
//MatMult<<<numBlocks,threadsPerBlock>>>(pA, pB, pC,2);
//MatTrans<<<numBlocks,threadsPerBlock>>>(pA, pC, 2, 2);
multByValue<<<numBlocks,threadsPerBlock>>>(5, pA, pC);

cudaMemcpy(C, pC, (3*2)*sizeof(double), cudaMemcpyDeviceToHost);

int i, j; printf("C = \n");
for(i=0;i<3;i++){
    for(j=0;j<2;j++){
        printf("%f ", C[i][j]);
    }
    printf("\n");
}

cudaFree(pA);
cudaFree(pB);
cudaFree(pC);

printf("\n");
return 0;
}
