#include "s21_matrix_oop.h"

/**
 * @brief Default constructor. It creates S21Matrix with a
 * size [DEFAULTRAWS x DEFAULTCOLS], where every element = 0
 * @details Preprocessor macros DEFAULTRAWS and DEFAULCOLS are defined
 * in the header s21_matrix_oop.h
 */
S21Matrix::S21Matrix() : rows_{DEFAULTRAWS}, cols_{DEFAULTCOLS} {
  if (rows_ <= 0 || cols_ <= 0) {
    throw std::out_of_range(
        "Incorrect input, arguments has to be greater than 0");
  }
  matrix_ = new double[rows_ * cols_]{};
}

/**
 * @brief Default constructor. It creates S21Matrix with a
 * size [rows x cols], where every element = 0
 * @param rows Number of raws
 * @param cols Number of columns
 */
S21Matrix::S21Matrix(int rows, int cols) : rows_{rows}, cols_{cols} {
  if (rows_ <= 0 || cols_ <= 0) {
    throw std::out_of_range(
        "Incorrect input, arguments has to be greater than 0");
  }
  matrix_ = new double[rows_ * cols_]{};
}

/**
 * @brief Copy constructor. It copies S21Matrix object
 * @param other S21Mztrix object to be copied
 */
S21Matrix::S21Matrix(const S21Matrix& other)
    : S21Matrix(other.rows_, other.cols_) {
  for (int i = 0; i < rows_; ++i) {
    for (int j = 0; j < cols_; ++j) {
      (*this)(i, j) = other(i, j);
    }
  }
}

/**
 * @brief Move constructor
 * It creates a new S21Matrix object and destroys an old
 * @param other S21Matrix object to be moved
 */
S21Matrix::S21Matrix(S21Matrix&& other) noexcept
    : rows_{other.rows_}, cols_{other.cols_}, matrix_{other.matrix_} {
  other.matrix_ = nullptr;
  other.rows_ = 0;
  other.cols_ = 0;
}

/**
 * @brief Destructor
 */
S21Matrix::~S21Matrix() {
  if (matrix_) {
    delete[] matrix_;
    matrix_ = nullptr;
  }
}

/**
 * @brief Copy assignment operator overloading
 * @details It's used for l-values
 * @param other S21Matrix object to be copied
 * @return Copied S21Matrix
 */
S21Matrix& S21Matrix::operator=(const S21Matrix& other) {
  if (this != &other) {
    S21Matrix tmp(other);
    rows_ = tmp.rows_;
    cols_ = tmp.cols_;
    delete[] matrix_;
    matrix_ = tmp.matrix_;
    tmp.matrix_ = nullptr;
  }
  return *this;
}

/**
 * @brief Move assignment operator overloading
 * @details It's used for r-values
 * @param other S21Matrix object to be moved
 * @return Moved S21Matrix
 */
S21Matrix& S21Matrix::operator=(S21Matrix&& other) noexcept {
  if (this != &other) {
    std::swap(rows_, other.rows_);
    std::swap(cols_, other.cols_);
    std::swap(matrix_, other.matrix_);
  }
  return *this;
}

/**
 * @brief Addition assignment operator overloading
 * @param other The matrix to be added to the current matrix
 * @return Sum of 2 matrices
 */
S21Matrix& S21Matrix::operator+=(const S21Matrix& other) {
  SumMatrix(other);
  return *this;
}

/**
 * @brief Subtraction assignment operator overloading
 * @param other The matrix to be subtracted from the current matrix
 * @return Difference between the curent matrix and the other matrix
 */
S21Matrix& S21Matrix::operator-=(const S21Matrix& other) {
  SubMatrix(other);
  return *this;
}

/**
 * @brief Multiplication assignment operator overloading
 * @param number The number to multiply the current matrix
 * @return The matrix multiplied to the number
 */
S21Matrix& S21Matrix::operator*=(double number) {
  MulNumber(number);
  return *this;
}

/**
 * @brief Multiplication assignment operator overloading
 * @param other The (right) matrix to multiply the current (left) matrix
 * @return Multiplication of the current and the other matrices
 */
S21Matrix& S21Matrix::operator*=(const S21Matrix& other) {
  MulMatrix(other);
  return *this;
}

/**
 * @brief Operator + overloading
 * @param other The matrix to be added to the current matrix
 * @return Sum of 2 matrices
 */
S21Matrix S21Matrix::operator+(const S21Matrix& other) const {
  if (rows_ != other.rows_ || cols_ != other.cols_) {
    throw std::logic_error("Incorrect input, different matrix dimensions");
  }
  S21Matrix temp_matrix{*this};
  temp_matrix.SumMatrix(other);
  return temp_matrix;
}

/**
 * @brief Operator - overloading
 * @param other The matrix to be subtracted from the current matrix
 * @return Difference between the curent matrix and the other matrix
 */
S21Matrix S21Matrix::operator-(const S21Matrix& other) const {
  if (rows_ != other.rows_ || cols_ != other.cols_) {
    throw std::logic_error("Incorrect input, different matrix dimensions");
  }
  S21Matrix temp_matrix{*this};
  temp_matrix.SubMatrix(other);
  return temp_matrix;
}

/**
 * @brief Operator * overloading
 * @param number The number to multiply the current matrix
 * @return The matrix multiplied to the number
 */
S21Matrix S21Matrix::operator*(double number) const {
  S21Matrix temp_matrix{*this};
  temp_matrix.MulNumber(number);
  return temp_matrix;
}

/**
 * @brief Operator * overloading
 * @details Friend function is used in case a number stays before a matrix
 * @param number The number to multiply the matrix
 * @return The matrix multiplied to the number
 */
S21Matrix operator*(double number, const S21Matrix& other) {
  S21Matrix temp_matrix = other * number;
  return temp_matrix;
}

/**
 * @brief Operator * overloading
 * @param other The (right) matrix to multiply the current (left) matrix
 * @return Multiplication of the current and the other matrices
 */
S21Matrix S21Matrix::operator*(const S21Matrix& other) {
  S21Matrix temp_matrix{*this};
  temp_matrix.MulMatrix(other);
  return temp_matrix;
}

/**
 * @brief Operator == overloading
 * @param other The matrix to be compared to the current matrix
 * @return true if the matrices are equal
 * @return false if the matrices are not equal
 */
bool S21Matrix::operator==(const S21Matrix& other) const {
  return EqMatrix(other);
}

/**
 * @brief Index operator overloading
 * @param row Index of the row of the matrix
 * @param col Index of the column of the matrix
 * @return Value of the element of the matrix with index (row, col)
 */
double& S21Matrix::operator()(int row, int col) const {
  if (row >= rows_ || col >= cols_ || row < 0 || col < 0) {
    throw std::out_of_range("Incorrect input, index is out of range");
  }
  return matrix_[row * cols_ + col];
}

/**
 * @brief Comparing of 2 matrices if they are equal
 * @param other The matrix to be compared to the current matrix
 * @return true if the matrices are equal
 * @return false if the matrices are not equal
 */
bool S21Matrix::EqMatrix(const S21Matrix& other) const {
  if (rows_ != other.rows_ || cols_ != other.cols_) {
    return false;
  }
  for (int i = 0; i < rows_; ++i) {
    for (int j = 0; j < cols_; ++j) {
      if (std::abs((*this)(i, j) - other(i, j)) > EPSILON) {
        return false;
      }
    }
  }
  return true;
}

/**
 * @brief Sum of 2 matrices
 * @param other The matrix to be added to the current matrix
 * @return Sum of 2 matrices
 */
void S21Matrix::SumMatrix(const S21Matrix& other) {
  if (rows_ != other.rows_ || cols_ != other.cols_) {
    throw std::logic_error("Incorrect input, different matrix dimensions");
  }
  for (int i = 0; i < rows_; ++i) {
    for (int j = 0; j < cols_; ++j) {
      (*this)(i, j) += other(i, j);
    }
  }
}

/**
 * @brief Subtraction of matrices
 * @param other The matrix to be subtracted from the current matrix
 * @return Difference between the curent matrix and the other matrix
 */
void S21Matrix::SubMatrix(const S21Matrix& other) {
  if (rows_ != other.rows_ || cols_ != other.cols_) {
    throw std::logic_error("Incorrect input, different matrix dimensions");
  }
  for (int i = 0; i < rows_; ++i) {
    for (int j = 0; j < cols_; ++j) {
      (*this)(i, j) -= other(i, j);
    }
  }
}

/**
 * @brief Multiplication of the matrix and the number
 * @param num The number to multiply the current matrix
 * @return The matrix multiplied to the number
 */
void S21Matrix::MulNumber(const double num) {
  for (int i = 0; i < rows_; ++i) {
    for (int j = 0; j < cols_; ++j) {
      (*this)(i, j) *= num;
    }
  }
}

/**
 * @brief Multiplication of the 2 matrices
 * @param other The (right) matrix to multiply the current (left) matrix
 * @return Multiplication of the current and the other matrices
 */
void S21Matrix::MulMatrix(const S21Matrix& other) {
  if (cols_ != other.rows_) {
    throw std::logic_error(
        "Incorrect input, the number of columns of the first matrix is not "
        "equal to the number of rows of the second matrix ");
  }
  S21Matrix temp_matrix{rows_, other.cols_};
  for (int i = 0; i < temp_matrix.rows_; ++i) {
    for (int j = 0; j < temp_matrix.cols_; ++j) {
      for (int k = 0; k < cols_; ++k) {
        temp_matrix(i, j) += (*this)(i, k) * other(k, j);
      }
    }
  }
  *this = std::move(temp_matrix);
}

/**
 * @brief Transposition of a matrix
 * @return Transposed matrix
 */
S21Matrix S21Matrix::Transpose() const {
  S21Matrix temp_matrix{cols_, rows_};
  for (int i = 0; i < rows_; ++i) {
    for (int j = 0; j < cols_; ++j) {
      temp_matrix(j, i) = (*this)(i, j);
    }
  }
  return temp_matrix;
}

/**
 * @brief Calculation of a determinant of a matrix
 * @details Calculation uses Gauss-Jordan elimination method with pivoting
 * @return Determinant of the matrix
 */
double S21Matrix::Determinant() {
  if (rows_ != cols_) {
    throw std::logic_error("Incorrect input, the matrix is not square");
  }
  S21Matrix temp_matrix{*this};
  double determinant = 1.0;
  for (int j = 0; j < cols_; ++j) {
    double value_pivot = 0.0;
    int i_pivot = j;
    // choose the pivot (the largest element in a lower triangle of the matrix)
    for (int i = j; i < rows_; ++i) {
      if (value_pivot < std::abs(temp_matrix(i, j))) {
        value_pivot = std::abs(temp_matrix(i, j));
        i_pivot = i;
      }
    }
    // check that pivot is larger than 0
    if (std::abs(temp_matrix(i_pivot, j)) - EPSILON <= 0.0) {
      determinant = 0.0;
      return determinant;
    }
    // swap of the rows to put the pivot to the main diagonal
    if (i_pivot != j) {
      temp_matrix.SwapRows(j, i_pivot);
      determinant *= -1.0;
    }
    // multiplication of the determinant variable and the pivot
    determinant *= temp_matrix(j, j);
    // zeroing of all the elements in the column below the main diagonal
    for (int i2 = j + 1; i2 < rows_; ++i2) {
      double k = temp_matrix(i2, j) / temp_matrix(j, j);
      for (int j2 = j; j2 < cols_; ++j2) {
        temp_matrix(i2, j2) -= k * temp_matrix(j, j2);
      }
    }
  }
  return determinant;
}

/**
 * @brief Creates a matrix of algebraic complemets to the current matrix
 * @details For each element of the matrix it's created a complementary minor
 * and it's calculated a determinant
 * @return Matrix of the algebraic complements
 */
S21Matrix S21Matrix::CalcComplements() {
  if (rows_ != cols_) {
    throw std::logic_error("Incorrect input, the matrix is not square");
  }
  S21Matrix complements_matrix{rows_, cols_};
  for (int i = 0; i < rows_; ++i) {
    for (int j = 0; j < cols_; ++j) {
      S21Matrix minor_matrix = (*this).GetComplementaryMinor(i, j);
      double complement = minor_matrix.Determinant();
      if ((i + j) % 2 == 1) {
        complement *= -1.0;
      }
      complements_matrix(i, j) = complement;
    }
  }
  return complements_matrix;
}

/**
 * @brief Calculation of an inverse matrix
 * @details Calculation uses Gauss-Jordan elimination method with pivoting.
 * Elementary transformations of the copy of the given matrix to the identity
 * matrix is doing simultaneously with the same elementary transformation of an
 * identity matrix of the size [rows_, cols_]. When the copy of the given
 * matrix is transformed into identity matrix the initial identity matrix is
 * transformed into the inverse matrix of the given matrix.
 * @return Inverse matrix
 */
S21Matrix S21Matrix::InverseMatrix() {
  if (rows_ != cols_) {
    throw std::logic_error("Incorrect input, the matrix is not square");
  } else if (std::abs((*this).Determinant()) - EPSILON <= 0.0) {
    throw std::logic_error(
        "Incorrect input, no invertible matrix: determinant is 0");
  }
  S21Matrix temp_matrix(*this);
  S21Matrix inverse_matrix = GetIdentityMatrix(rows_);
  // zeroing of the lower triangle
  for (int j = 0; j < cols_; ++j) {
    double value_pivot = 0.0;
    int i_pivot = j;
    // choose the pivot (the largest element in a lower triangle of the matrix)
    for (int i = j; i < rows_; ++i) {
      if (value_pivot < std::abs(temp_matrix(i, j))) {
        value_pivot = std::abs(temp_matrix(i, j));
        i_pivot = i;
      }
    }
    // swap of the rows to put the pivot to the main diagonal
    temp_matrix.SwapRows(j, i_pivot);
    inverse_matrix.SwapRows(j, i_pivot);
    // zeroing of all the elements in the column below the main diagonal
    for (int i = j + 1; i < rows_; ++i) {
      double k = temp_matrix(i, j) / temp_matrix(j, j);
      for (int j2 = 0; j2 < cols_; ++j2) {
        temp_matrix(i, j2) -= k * temp_matrix(j, j2);
        inverse_matrix(i, j2) -= k * inverse_matrix(j, j2);
      }
    }
    /* division of the current row by the elemant at the main diagonal -
    to put the number 1 at the main diagonal */
    double k = temp_matrix(j, j);
    for (int j2 = cols_ - 1; j2 >= 0; --j2) {
      inverse_matrix(j, j2) /= k;
      temp_matrix(j, j2) /= k;
    }
  }
  // zeroing of the upper triangle
  for (int j = 0; j < cols_; ++j) {
    for (int i = 0; i < j; ++i) {
      double k = temp_matrix(i, j);
      for (int j2 = 0; j2 < cols_; ++j2) {
        temp_matrix(i, j2) -= k * temp_matrix(j, j2);
        inverse_matrix(i, j2) -= k * inverse_matrix(j, j2);
      }
    }
  }
  return inverse_matrix;
};

/**
 * @brief Accessor to the private field rows_
 * @return value of the rows_
 */
int S21Matrix::GetRows() const noexcept { return rows_; }

/**
 * @brief Accessor to the private field cols_
 * @return value of the cols_
 */
int S21Matrix::GetCols() const noexcept { return cols_; }

/**
 * @brief Mutator for the private field rows_
 * @details Change number of rows for a given matrix.
 * If the matrix increases in size, it is filled with zeros.
 * If it decreases in size, the excess is simply discarded
 * @param new_rows Number of rows in the new matrix
 */
void S21Matrix::SetRows(int new_rows) {
  if (new_rows <= 0) {
    throw std::out_of_range("Incorrect input, the number of rows is below 1");
  }
  if (new_rows != rows_) {
    S21Matrix new_matrix{new_rows, cols_};
    int min_rows;
    if (rows_ > new_rows) {
      min_rows = new_rows;
    } else {
      min_rows = rows_;
    }
    for (int i = 0; i < min_rows; ++i) {
      for (int j = 0; j < cols_; ++j) {
        new_matrix(i, j) = (*this)(i, j);
      }
    }
    *this = std::move(new_matrix);
  }
}

/**
 * @brief Mutator for the private field cols_
 * @details Change number of columns for a given matrix.
 * If the matrix increases in size, it is filled with zeros.
 * If it decreases in size, the excess is simply discarded
 * @param new_cols Number of columns in the new matrix
 */
void S21Matrix::SetCols(int new_cols) {
  if (new_cols <= 0) {
    throw std::out_of_range(
        "Incorrect input, the number of columns is below 1");
  }
  if (new_cols != cols_) {
    S21Matrix new_matrix{rows_, new_cols};
    int min_cols;
    if (cols_ > new_cols) {
      min_cols = new_cols;
    } else {
      min_cols = cols_;
    }
    for (int i = 0; i < rows_; ++i) {
      for (int j = 0; j < min_cols; ++j) {
        new_matrix(i, j) = (*this)(i, j);
      }
    }
    *this = std::move(new_matrix);
  }
}

/**
 * @brief Mutator for the private fields rows_ and cols_
 * @details Change number of rows and columns for a given matrix.
 * If the matrix increases in size, it is filled with zeros.
 * If it decreases in size, the excess is simply discarded
 * @param new_rows Number of rows in the new matrix
 * @param new_cols Number of columns in the new matrix
 */
void S21Matrix::SetRowsCols(int new_rows, int new_cols) {
  if (new_rows <= 0 || new_cols <= 0) {
    throw std::out_of_range(
        "Incorrect input, the number of rows or columns is below 1");
  }
  if (new_rows != rows_ || new_cols != cols_) {
    S21Matrix new_matrix{new_rows, new_cols};
    int min_rows;
    if (rows_ > new_rows) {
      min_rows = new_rows;
    } else {
      min_rows = rows_;
    }
    int min_cols;
    if (cols_ > new_cols) {
      min_cols = new_cols;
    } else {
      min_cols = cols_;
    }
    for (int i = 0; i < min_rows; ++i) {
      for (int j = 0; j < min_cols; ++j) {
        new_matrix(i, j) = (*this)(i, j);
      }
    }
    *this = std::move(new_matrix);
  }
}

/**
 * @brief Swap rows of a matrix
 * @attention Private method. In case this method redifined to public the throw
 * block has to be added at the very beginning:
 * if (row1 < 0 || row1 >= (*this).GetRows() || \
 *  row2 < 0 || row2 >= (*this).GetRows()) {
 *    throw std::out_of_range("Incorrect input, \
 *    index of a row is out of a range");
 * }
 * @param row1 Index of the row to swap
 * @param row2 Index of the other row to swap
 */
void S21Matrix::SwapRows(int row1, int row2) {
  if (row1 != row2) {
    for (int j = 0; j < cols_; ++j) {
      std::swap((*this)(row1, j), (*this)(row2, j));
    }
  }
}

/**
 * @brief Get complementary minor to the element of a matrix
 * @attention Private method. In case this method redifined to public the throw
 * block has to be added at the very beginning:
 * if (row < 0 || col < 0) {
 *  throw std::logic_error("Incorrect input, negative index of row or column");
 * } else if (rows_ != cols_ ) {
 * throw std::logic_error("Incorrect input, the matrix is not square");
 * } else
 * @param row Index of the row of the element
 * @param col Index of the column of the element
 * @return Complementary minor to the element of the matrix
 */
S21Matrix S21Matrix::GetComplementaryMinor(int row, int col) {
  if (rows_ == 1) {
    throw std::logic_error(
        "Incorrect input, a 1-dimensional matrix has no minor");
  }
  S21Matrix minor_matrix{rows_ - 1, cols_ - 1};
  int i2;
  int j2;
  for (int i = 0; i < minor_matrix.rows_; ++i) {
    if (i < row) {
      i2 = i;
    } else {
      i2 = i + 1;
    }
    for (int j = 0; j < minor_matrix.cols_; j++) {
      if (j < col) {
        j2 = j;
      } else {
        j2 = j + 1;
      }
      minor_matrix(i, j) = (*this)(i2, j2);
    }
  }
  return minor_matrix;
}

/**
 * @brief Get identity matrix of a given size
 * @attention Private method. In case this method redifined to public the throw
 * block has to be added at the very beginning:
 * if (size < 1) {
 *  throw std::logic_error("Incorrect input, size of matrix smaller than 1");
 * }
 * @return Identity matrix of the given size
 */
S21Matrix S21Matrix::GetIdentityMatrix(int size) {
  S21Matrix identity_matrix{size, size};
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      if (i == j) {
        identity_matrix(i, j) = 1.0;
      } else {
        identity_matrix(i, j) = 0.0;
      }
    }
  }
  return identity_matrix;
}