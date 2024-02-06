#include "tests.h"

/**
 * @brief To print a matrix in a terminal
 * @attention To use the function also uncoment it in tests.h
 */
/*
void printMatrix(const S21Matrix &m) {
  for (int i = 0; i < m.GetRows(); ++i) {
    for (int j = 0; j < m.GetCols(); ++j) {
      std::cout << m(i, j) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}
 */

/**
 * @brief To fill a matrix with values from arithmetic progression
 * begining from 0
 */
void fillMatrixStep(const S21Matrix &m, double step) {
  double num = 0.0;
  for (int i = 0; i < m.GetRows(); ++i) {
    for (int j = 0; j < m.GetCols(); ++j) {
      m(i, j) = num;
      num += step;
    }
  }
}

/**
 * @brief To fill a matrix with the same value
 */
void fillMatrixValue(const S21Matrix &m, double value) {
  for (int i = 0; i < m.GetRows(); ++i) {
    for (int j = 0; j < m.GetCols(); ++j) {
      m(i, j) = value;
    }
  }
}

/**
 * @brief To fill a matrix with values from given array
 * @attention Function may SegFault in case (len < length of values)
 */
void fillMatrixArray(S21Matrix &m, double *values, int len) {
  int v = 0;
  for (int i = 0; i < m.GetRows(); ++i) {
    for (int j = 0; j < m.GetCols(); ++j) {
      if (v < len) {
        m(i, j) = values[v];
      } else {
        m(i, j) = 0;
      }
      ++v;
    }
  }
}