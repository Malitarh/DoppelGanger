#include <iostream>
#include <amp.h>
#include <stdlib.h>
#include <ctime>

using namespace concurrency;

void printMatrix(int* matrix, int rows, int cols) {
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            std::cout << matrix[cols * row + col] << " \t";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

void MultiplyWithOutAMP(int* a, int* b, int* result, int rowsA, int colrow, int colsB) {
    
    for (int row = 0; row < rowsA; row++) {
        for (int col = 0; col < colsB; col++) {
            // Multiply the row of A by the column of B to get the row, column of product.
            for (int inner = 0; inner < colrow; inner++) {
                result[colrow * row + col] += a[row * colrow + inner] * b[inner * colrow + col];
            }
        }
    }
}

void MultiplyWithAMP(int* a, int* b, int* result, int rowsA, int colrow, int colsB) {

    array_view<int, 2> aMatrix(rowsA, colrow, a);

    array_view<int, 2> bMatrix(colrow, colsB, b);

    array_view<int, 2> product(rowsA, colsB, result);

    parallel_for_each(product.extent,
        [=](index<2> idx) restrict(amp) {
            int row = idx[0];
            int col = idx[1];
            for (int inner = 0; inner <colrow; inner++) {
                product[idx] += aMatrix(row, inner) * bMatrix(inner, col);
            }
        });
    product.synchronize();
}

const int MATRIX_SIZE = 1000;

int main() {
    int* aMatrix = (int*) malloc(sizeof(int) * MATRIX_SIZE * MATRIX_SIZE);
    int* bMatrix = (int*) malloc(sizeof(int) * MATRIX_SIZE * MATRIX_SIZE);
    int* result = (int*) malloc(sizeof(int) * MATRIX_SIZE * MATRIX_SIZE);
    
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            aMatrix[i * MATRIX_SIZE + j] = trunc ((double) rand() / MAXSHORT * 10);
            bMatrix[i * MATRIX_SIZE + j] = trunc ((double) rand() / MAXSHORT * 10);
            result[i * MATRIX_SIZE + j] = 0;
        }
    }
    //printMatrix(aMatrix, MATRIX_SIZE, MATRIX_SIZE);
    //printMatrix(bMatrix, MATRIX_SIZE, MATRIX_SIZE);

    unsigned int start_time = clock();
    MultiplyWithOutAMP(aMatrix, bMatrix, result, MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE);
    unsigned int end_time = clock() - start_time;

    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            result[i * MATRIX_SIZE + j] = 0;
        }
    }

    start_time = clock();
    MultiplyWithAMP(aMatrix, bMatrix, result, MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE);
    
    std::cout << "With AMP: " << clock() - start_time << " ms\n";
    std::cout << "Without AMP: " << end_time << " ms\n";
}