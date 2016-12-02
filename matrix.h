#ifndef MATRIX_H
#define MATRIX_H

typedef struct Matrix {
    int rows;
    int cols;
    double ** matrix;
} Matrix;

/** Build/Destroy/Display Methods
-------------------------------------------*/
Matrix buildMatrix(int r, int c);

Matrix displayMatrix(Matrix m);

/** Matrix Get and Set
-------------------------------------------*/
void setValue(Matrix * m, double val, int r, int c);

double getValue(Matrix m, int r, int c);

/** Matrix Multiplication
-------------------------------------------*/
Matrix multByValue(double val, Matrix m);

Matrix multByMatrix(Matrix mat1, Matrix mat2);

/** Matrix Inversion
-------------------------------------------*/
Matrix invertMatrix(Matrix m);

Matrix minorOf(Matrix m);

/** Matrix Determinate
-------------------------------------------*/
double getDeterminate(Matrix m);

/** Matrix Transpose
-------------------------------------------*/
Matrix transpose(Matrix m);

/** Matrix Cofactor
-------------------------------------------*/
Matrix coFactor(Matrix m);

#endif
