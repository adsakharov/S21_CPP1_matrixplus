#ifndef S21_MATRIX_OOP_H_
#define S21_MATRIX_OOP_H_

#include <cmath>  // std::abs
#include <iostream>
#include <utility>  // std:swap

#define EPSILON 1e-7
#define DEFAULTRAWS 3
#define DEFAULTCOLS 3

class S21Matrix {
 public:
  // Constructors and assignment operators
  S21Matrix();
  S21Matrix(int rows, int cols);
  S21Matrix(const S21Matrix& other);
  S21Matrix(S21Matrix&& other) noexcept;
  S21Matrix& operator=(S21Matrix&& other) noexcept;
  S21Matrix& operator=(const S21Matrix& other);
  S21Matrix& operator+=(const S21Matrix& other);
  S21Matrix& operator-=(const S21Matrix& other);
  S21Matrix& operator*=(double number);
  S21Matrix& operator*=(const S21Matrix& other);

  // Destructor
  ~S21Matrix();

  // Other operator overloadings
  double& operator()(int row, int col) const;
  S21Matrix operator+(const S21Matrix& other) const;
  S21Matrix operator-(const S21Matrix& other) const;
  S21Matrix operator*(double number) const;
  friend S21Matrix operator*(double number, const S21Matrix& other);
  S21Matrix operator*(const S21Matrix& other);
  bool operator==(const S21Matrix& other) const;

  // Other functions
  bool EqMatrix(const S21Matrix& other) const;
  void SumMatrix(const S21Matrix& other);
  void SubMatrix(const S21Matrix& other);
  void MulNumber(const double num);
  void MulMatrix(const S21Matrix& other);
  S21Matrix Transpose() const;
  S21Matrix CalcComplements();
  double Determinant();
  S21Matrix InverseMatrix();

  // Accessors and mutators
  int GetRows() const noexcept;
  int GetCols() const noexcept;
  void SetRows(int new_rows);
  void SetCols(int new_cols);
  void SetRowsCols(int new_rows, int new_cols);

 private:
  int rows_, cols_;
  double* matrix_;

  void SwapRows(int row1, int row2);
  S21Matrix GetComplementaryMinor(int row, int col);
  S21Matrix GetIdentityMatrix(int size);
};

#endif