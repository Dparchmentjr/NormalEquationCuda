#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include "matrix.h"

/** Build and Display Methods
---------------------------------------------------------------------*/
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

Matrix displayMatrix(Matrix m) {
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

/** Set and Get Methods
Note: These methods will only be used by the main.c file, in order
for easier manipulation of created matrices. The matrix property will
be used in the actual matrix math functions for better speed.
---------------------------------------------------------------------*/
void setValue(Matrix* m, double val, int r, int c) {
    m->matrix[r][c] = val;
}

double getValue(Matrix m, int r, int c) {
    return m.matrix[r][c];
}

/** Matrix Multiplication
---------------------------------------------------------------------*/
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
    //In order for matrix multiplication to occur
    //the columns of the first matrix must match
    //the rows of the second matrix
    Matrix result;
    if(mat1.cols == mat2.rows) {
        //Then the newly built array must be of
        //the size (mat1.rows, mat2.cols)
        result = buildMatrix(mat1.rows, mat2.cols);
    }
    else if(mat1.cols == mat2.cols) {
        result = buildMatrix(mat2.rows, mat1.rows);
    } else {
        result = buildMatrix(mat1.rows, mat1.cols);
    }
    double product = 0;
    // double **matrix1 = mat1.matrix;
    // double **matrix2 = mat2.matrix;
    //Perform the dot product
    for(int i = 0; i < mat1.rows; i++) {
        for(int j = 0; j < mat2.cols; j++) {
            for(int k = 0; k < mat1.cols; k++) {
                result.matrix[i][j] += mat1.matrix[i][k] * mat2.matrix[k][j];
            }
        }
    }
    return result;
}

/** Matrix Determinate
---------------------------------------------------------------------*/
double getDeterminate(Matrix m) {
    int i, j, k;
    double ratio;
    int rows = m.rows, cols = m.cols;
    //We don't want to change the given matrix, just
    //find it's determinate so use a temporary matrix
    //instead for the calculations
    Matrix temp = buildMatrix(m.rows, m.cols);
    for(int i = 0; i < m.rows; i++) {
        for(int j = 0; j < m.cols; j++) {
            temp.matrix[i][j] = m.matrix[i][j];
        }
    }
    //If the matrix is not square then the derminate does not exist
    if(rows == cols) {
        //If the matrix is 2x2 matrix then the
        //determinate is ([0,0]x[1,1]) - ([0,1]x[1,0])
         if(rows == 2 && cols == 2) {
             return (temp.matrix[0][0] * temp.matrix[1][1]) - (temp.matrix[0][1] * temp.matrix[1][0]);
         }
        //Otherwise if it is n*n...we do things the long way
        //we will be using a method where we convert the
        //matrix into an upper triangle, and then the det
        //will simply be the multiplication of all diagonal
        //indexes ---Idea from Khan Academy
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
    double det = 1; //storage for determinant
    for(i = 0; i < rows; i++) {
        det *= temp.matrix[i][i];
    }
        return det;
    }
    return 0;
}

/** Matrix Transpose
    Pretty self explanatory
---------------------------------------------------------------------*/
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

/** Matrix Cofactor
---------------------------------------------------------------------*/
Matrix coFactor(Matrix m) {
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

/** Matrix Inverse
Note: This method will assume the determinate is not equal to 0
---------------------------------------------------------------------*/
Matrix minorOf(Matrix m) {
    Matrix temp = buildMatrix(m.rows, m.cols);
    Matrix detMatrix = buildMatrix(m.rows - 1, m.cols - 1);
    Matrix minor = buildMatrix(m.rows, m.cols);
    int rw = 0, cl = 0, numOfIncludes = 0;
    for(int i = 0; i < m.rows; i++) {
        for(int j = 0; j < m.cols; j++) {
            temp.matrix[i][j] = m.matrix[i][j];
        }
    }
    for(int c = 0; c < temp.rows; c++) {
        for(int d = 0; d < temp.cols; d++) {
            rw = 0;
            for(int i = 0; i < temp.rows; i++) {
                cl = 0;
                for(int j = 0; j < temp.cols; j++) {
                    if(i == c || j == d) {
                        if(numOfIncludes >= detMatrix.rows) {
                            cl++;
                        }
                    } else {
                        cl++;
                        numOfIncludes++;
                        setValue(&detMatrix, temp.matrix[i][j], rw, cl - 1);
                    }
                }
                if(numOfIncludes >= detMatrix.rows) {
                    rw++;
                    numOfIncludes = 0;
                }
            }
            setValue(&minor, getDeterminate(detMatrix), c, d);
        }
        rw = 0;
        cl = 0;
    }
    return minor;
}

Matrix invertMatrix(Matrix m) {
    Matrix adjugate = transpose(coFactor(minorOf(m)));
    return multByValue((1/getDeterminate(m)), adjugate);
}
