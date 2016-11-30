#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "matrix.h"
int main() {
    Matrix mat1 = buildMatrix(6, 2);
    Matrix p = buildMatrix(1, 6);
    
    setValue(&p, 53000, 0, 0);setValue(&p, 53000, 0, 1);setValue(&p, 53000, 0, 2);setValue(&p, 53000, 0, 3);setValue(&p, 53000, 0, 4);setValue(&p, 53000, 0, 5);

    setValue(&mat1, 1, 0, 0);setValue(&mat1, 1700, 0, 1);
    setValue(&mat1, 1, 1, 0);setValue(&mat1, 2100, 1, 1);
    setValue(&mat1, 1, 2, 0);setValue(&mat1, 1900, 2, 1);
    setValue(&mat1, 1, 3, 0);setValue(&mat1, 1300, 3, 1);
    setValue(&mat1, 1, 4, 0);setValue(&mat1, 1600, 4, 1);
    setValue(&mat1, 1, 5, 0);setValue(&mat1, 2200, 5, 1);
    //displayMatrix(mat1);
    //Theta = (inv(X'*X)*X')*p
    Matrix XT = transpose(mat1);
    Matrix XI = invertMatrix(multByMatrix(XT, mat1));
    Matrix XM = multByMatrix(XI, XT);
    // displayMatrix(XM);
    displayMatrix(transpose(p));
    Matrix Theta = multByMatrix(XM, transpose(p));
    // displayMatrix(Theta);
}

//(2, 6) x (6, 1)