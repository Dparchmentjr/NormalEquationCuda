#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "matrix.h"
int main() {
    Matrix m, p;
    FILE *fp = fopen("ex1data1.txt", "r");
    char line[100000];
    float mArr[100000], pArr[100000];
    int rowCount = 0, lineCount = 0, counter = 0;
    float mVal = 0, pVal = 0;
    while(fgets(line, sizeof(line), fp) != NULL) {
        if(lineCount == 0) {
            //Make sure the first line is the row count
            sscanf(line, "%d", &rowCount);
        }else {
            sscanf(line, "%f,%f", &mVal, &pVal);
            mArr[counter] = mVal;pArr[counter] = pVal;
            counter++;
        }
        lineCount++;
    }
    counter--;
    m = buildMatrix(rowCount, 2);
    p = buildMatrix(1, rowCount);
    for(int i = 0; i < m.rows; i++) {
        for(int j = 0; j < m.cols; j++) {
            if(j == 0) {
                m.matrix[i][j] = 1;
            }else {
                m.matrix[i][j] = mArr[i];
            }
        }
    }
    for(int i = 0; i < p.rows; i++) {
        for(int j = 0; j < p.cols; j++) {
            p.matrix[i][j] = pArr[j];
        }
    }

    Matrix XT = transpose(m);
    Matrix XI = invertMatrix(multByMatrix(XT, m));
    Matrix XM = multByMatrix(XI, XT);
    Matrix Theta = multByMatrix(XM, transpose(p));
    displayMatrix(Theta);
}
