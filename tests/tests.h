#ifndef S21_MATRIX_OOP_TESTS_H_
#define S21_MATRIX_OOP_TESTS_H_

#include <gtest/gtest.h>

#include "../s21_matrix_oop.h"

// void printMatrix(const S21Matrix &m);
void fillMatrixStep(const S21Matrix &m, double step);
void fillMatrixValue(const S21Matrix &m, double value);
void fillMatrixArray(S21Matrix &m, double *values, int len);

#endif