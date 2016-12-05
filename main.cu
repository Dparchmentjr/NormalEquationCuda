#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

#define N 500



struct timeval start, end;

typedef struct Matrix {
    int rows;
    int cols;
    double ** matrix;
} Matrix;

Matrix buildMatrix(int r, int c) {
    Matrix temp;
    temp.rows = r;
    temp.cols = c;
    temp.matrix = (double**)malloc(r * sizeof(double*));

    unsigned int i;
    for(i = 0; i < r; i++)
    {
        temp.matrix[i] = (double*)calloc(c,sizeof(double));
    }

    return temp;
}

void displayMatrix(Matrix m) {
    printf("\n");
    for(int i = 0; i < m.rows; i++) {
        printf("| ");
        for(int j = 0; j < m.cols; j++) {
            printf("%8.5f ", m.matrix[i][j]);
        }
        printf("|\n");
    }
    printf("\n");
}

void setValue(Matrix* m, double val, int r, int c) {
    m->matrix[r][c] = val;
}

Matrix multByValue(double val, Matrix m) {
    Matrix result = buildMatrix(m.rows, m.cols);
    for(int i = 0; i < result.rows; i++) {
        for(int j = 0; j < result.cols; j++) {
            result.matrix[i][j] = val * m.matrix[i][j];
        }
    }
    return result;
}

Matrix multByMatrix(Matrix mat1, Matrix mat2) {
    Matrix result;
    if(mat1.cols == mat2.rows) {
        result = buildMatrix(mat1.rows, mat2.cols);
    }
    else if(mat1.cols == mat2.cols) {
        result = buildMatrix(mat2.rows, mat1.rows);
    } else {
        result = buildMatrix(mat1.rows, mat1.cols);
    }

    for(int i = 0; i < mat1.rows; i++) {
        for(int j = 0; j < mat2.cols; j++) {
            for(int k = 0; k < mat1.cols; k++) {
                result.matrix[i][j] += mat1.matrix[i][k] * mat2.matrix[k][j];
            }
        }
    }
    return result;
}

double getDeterminate(Matrix m) {
    int i, j, k;
    double ratio;
    int rows = m.rows, cols = m.cols;
    Matrix temp = buildMatrix(m.rows, m.cols);
    for(int i = 0; i < m.rows; i++) {
        for(int j = 0; j < m.cols; j++) {
            temp.matrix[i][j] = m.matrix[i][j];
        }
    }
    if(rows == cols) {
         if(rows == 2 && cols == 2) {
             return (temp.matrix[0][0] * temp.matrix[1][1]) - (temp.matrix[0][1] * temp.matrix[1][0]);
         }
        for(i = 0; i < rows; i++){
        for(j = 0; j < cols; j++){
            if(j>i){
                ratio = temp.matrix[j][i]/temp.matrix[i][i];
                for(k = 0; k < rows; k++) {
                    temp.matrix[j][k] -= ratio * temp.matrix[i][k];
                }
            }
        }
    }
    double det = 1;
    for(i = 0; i < rows; i++) {
        det *= temp.matrix[i][i];
    }
        return det;
    }
    return 0;
}

Matrix transpose(Matrix m) {
    Matrix temp;
    if(m.rows == m.cols) {
        temp = buildMatrix(m.rows, m.cols);
    } else {
        temp = buildMatrix(m.cols, m.rows);
    }
    int i, j;
    for(i=0; i<temp.cols; i++) {
        for(j=0; j<temp.rows; j++) {
            temp.matrix[j][i] = m.matrix[i][j];
        }
    }
    return temp;
}

Matrix coFactorCPU(Matrix m) {
    Matrix temp = buildMatrix(m.rows, m.cols);
    int i, j;
    for(i = 0; i < m.rows; i++) {
        for(j = 0; j < m.cols; j++) {
            if((j + i) % 2 == 1) {
                temp.matrix[i][j] = m.matrix[i][j] * -1;
            }
            else {
                temp.matrix[i][j] = m.matrix[i][j];
            }
        }
    }
    return temp;
}

void starttime() {
  gettimeofday( &start, 0 );
}



void endtime() {
   gettimeofday( &end, 0 );
   double elapsed = ( end.tv_sec - start.tv_sec ) * 1000.0 + ( end.tv_usec - start.tv_usec ) / 1000.0;
   printf("%f ms\n", elapsed);
}



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

__global__ void MatTrans(double A[][N], double C[][N]){
               int i = threadIdx.x;
               int j = threadIdx.y;

               C[j][i] = A[i][j];

}

__global__ void multMatByValue(double val, double A[][N], double C[][N]) {
               int i = threadIdx.x;
               int j = threadIdx.y;
               C[i][j] = val * A[i][j];

}

__global__ void coFactor(double A[][N], double C[][N]) {
               int i = threadIdx.x;
               int j = threadIdx.y;

               if((j+i) % 2 == 1){
                 C[i][j] = A[i][j] * -1;
               }
               else{
                 C[i][j] = A[i][j];
               }
              //C[i][j] = A[i][j] * pow(-1,((j+i)%2));
}

__global__ void determinate(double A[][N], double *d, int width, int height){
    int i = threadIdx.x;
    int j = threadIdx.y;
    //printf("Hello from determinate\n");
    if(j > i){
        double ratio = A[j][i]/A[i][i];
        for(int k = 0; k < width; k++){
            A[j][k] -= ratio * A[i][k];
        }
    }
    for(int k = 0; k < width; k++){
        if(k == i){
            *d *= A[i][k];
        }
    }
}



__global__ void determinate_gpu(double A[][N], double* d, int width, int height){
    int i = threadIdx.x;
    int j = threadIdx.y;

    if(j > i){
        double ratio = A[j][i]/A[i][i];
        for(int k = 0; k < width; k++){
            A[j][k] -= ratio * A[i][k];
        }
    }
    for(int k = 0; k < width; k++){
        if(k == i){
            *d *= A[i][k];
        }
    }
}

int main(){

double A[N][N];
double C[N][N];
Matrix a = buildMatrix(N, N);
Matrix b = buildMatrix(N, N);

double E[N][N];
for(int i = 0; i < N; i++) {
  for(int j = 0; j < N; j++) {
    E[i][j] = 1;
  }
}


for(int i = 0; i< N;i++){
  for(int j = 0; j < N;j++ ){
    C[i][j] = 0;
    A[i][j] = 1;
    a.matrix[i][j] = 1;
    b.matrix[i][j] = 1;
  }
}

double B[N][N];

double det = 1.0;
double *d_det;

double (*pA)[N], (*pB)[N], (*pC)[N],(*pE)[N];

cudaMalloc((void**)&pA, (N*N)*sizeof(double));
cudaMalloc((void**)&pB, (N*N)*sizeof(double));
cudaMalloc((void**)&pC, (N*N)*sizeof(double));
cudaMalloc((void**)&pE, (N*N)*sizeof(double));
cudaMalloc((void**)&d_det, sizeof(double));

cudaMemcpy(pA, A, (N*N)*sizeof(double), cudaMemcpyHostToDevice);
cudaMemcpy(pB, B, (N*N)*sizeof(double), cudaMemcpyHostToDevice);
cudaMemcpy(pC, C, (N*N)*sizeof(double), cudaMemcpyHostToDevice);
cudaMemcpy(pE, E, (N*N)*sizeof(double), cudaMemcpyHostToDevice);
cudaMemcpy(d_det, &det, sizeof(double), cudaMemcpyHostToDevice);



int numBlocks = 1;
dim3 threadsPerBlock(N,N);
starttime();
printf("***Transpose on %d x %d Matrix on CPU***\n",N,N);
Matrix d = transpose(a);
endtime();
starttime();
printf("***Transpose on %d x %d Matrix on GPU***\n",N,N);
MatTrans<<<numBlocks,threadsPerBlock>>>(pA, pC);
endtime();
starttime();
printf("***Multiplication on %d x %d Matrix on CPU***\n",N,N);
Matrix c = multByMatrix(a, b);
endtime();
starttime();
printf("***Multiplication on %d x %d Matrix on GPU***\n",N,N);
MatMult<<<numBlocks,threadsPerBlock>>>(pA,pA,pC,N);
endtime();
starttime();
printf("***Multiplication by single value on %d x %d Matrix with %d on GPU***\n",N,N,5);
multMatByValue<<<numBlocks, threadsPerBlock>>>(5, pA, pC);
endtime();
starttime();
printf("***Multiplication by single value on %d x %d Matrix with %d on CPU***\n",N,N,5);
Matrix e = multByValue(5,a);
endtime();
starttime();
printf("***Cofactor of %d x %d Matrix on GPU***\n",N,N);
coFactor<<<numBlocks, threadsPerBlock>>>(pA,pC);
endtime();
starttime();
printf("***Cofactor of %d x %d Matrix on CPU***\n",N,N);
Matrix f = coFactorCPU(a);
endtime();
starttime();
printf("***Determinate of %d x %d Matrix on CPU***\n",N,N);
double determinate = getDeterminate(a);
endtime();
starttime();
printf("***Determinate of %d x %d Matrix on GPU***\n",N,N);
determinate_gpu<<<numBlocks, threadsPerBlock>>>(pE, d_det, N, N);
endtime();
cudaMemcpy(&det, d_det, sizeof(double), cudaMemcpyDeviceToHost);
cudaMemcpy(C, pC, (N*N)*sizeof(double), cudaMemcpyDeviceToHost);


cudaFree(pA);
cudaFree(pB);
cudaFree(pC);
cudaFree(d_det);


printf("\n");
return 0;
}
