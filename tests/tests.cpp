#include "tests.h"

TEST(test_constructor, default_constructor1) {
  EXPECT_GE(DEFAULTRAWS, 0);
  EXPECT_GE(DEFAULTCOLS, 0);
  S21Matrix matrix;
  ASSERT_EQ(matrix.GetRows(), DEFAULTRAWS);
  ASSERT_EQ(matrix.GetCols(), DEFAULTCOLS);
  EXPECT_NEAR(matrix(0, 0), 0.0, EPSILON);
}

TEST(test_constructor, constructor_params1) {
  S21Matrix matrix{1, 1};
  EXPECT_EQ(matrix.GetRows(), 1);
  EXPECT_EQ(matrix.GetCols(), 1);
  EXPECT_NEAR(matrix(0, 0), 0.0, EPSILON);
}

TEST(test_constructor, constructor_params2) {
  EXPECT_NO_THROW(S21Matrix matrix(1, 1));
  EXPECT_ANY_THROW(S21Matrix matrix1(-1, -1));
  EXPECT_ANY_THROW(S21Matrix matrix2(-1, 1));
  EXPECT_ANY_THROW(S21Matrix matrix3(1, -1));
  EXPECT_ANY_THROW(S21Matrix matrix4(0, 1));
  EXPECT_ANY_THROW(S21Matrix matrix5(0, 0));
  EXPECT_ANY_THROW(S21Matrix matrix6(1, 0));
}

TEST(test_constructor, copy_constructor1) {
  S21Matrix matrix{4, 5};
  fillMatrixStep(matrix, 2.0);
  S21Matrix matrix2(matrix);
  ASSERT_EQ(matrix2.GetRows(), 4);
  ASSERT_EQ(matrix2.GetCols(), 5);
  EXPECT_NEAR(matrix2(0, 0), 0.0, EPSILON);
  EXPECT_NEAR(matrix2(0, 1), 2.0, EPSILON);
  EXPECT_NEAR(matrix2(3, 4), 38.0, EPSILON);
}

TEST(test_constructor, move_constructor1) {
  S21Matrix matrix{4, 5};
  fillMatrixStep(matrix, 2.0);
  S21Matrix matrix2(std::move(matrix));
  ASSERT_EQ(matrix2.GetRows(), 4);
  ASSERT_EQ(matrix2.GetCols(), 5);
  EXPECT_NEAR(matrix2(0, 0), 0.0, EPSILON);
  EXPECT_NEAR(matrix2(0, 1), 2.0, EPSILON);
  EXPECT_NEAR(matrix2(3, 4), 38.0, EPSILON);
}

TEST(test_assignment_operator, copy_assignment1) {
  S21Matrix matrix{4, 5};
  fillMatrixStep(matrix, 2.0);
  S21Matrix matrix2;
  matrix2 = matrix;
  ASSERT_EQ(matrix2.GetRows(), 4);
  ASSERT_EQ(matrix2.GetCols(), 5);
  EXPECT_NEAR(matrix2(0, 0), 0.0, EPSILON);
  EXPECT_NEAR(matrix2(0, 1), 2.0, EPSILON);
  EXPECT_NEAR(matrix2(3, 4), 38.0, EPSILON);
}

TEST(test_assignment_operator, move_assignment1) {
  S21Matrix matrix{4, 5};
  fillMatrixStep(matrix, 2.0);
  S21Matrix matrix2;
  matrix2 = std::move(matrix);
  ASSERT_EQ(matrix2.GetRows(), 4);
  ASSERT_EQ(matrix2.GetCols(), 5);
  EXPECT_NEAR(matrix2(0, 0), 0.0, EPSILON);
  EXPECT_NEAR(matrix2(0, 1), 2.0, EPSILON);
  EXPECT_NEAR(matrix2(3, 4), 38.0, EPSILON);
}

TEST(test_assignment_operator, plus_assignment1) {
  S21Matrix matrix{4, 5};
  fillMatrixStep(matrix, 2.0);
  S21Matrix matrix2{4, 5};
  fillMatrixValue(matrix2, 2.0);
  matrix2 += matrix;
  ASSERT_EQ(matrix2.GetRows(), 4);
  ASSERT_EQ(matrix2.GetCols(), 5);
  EXPECT_NEAR(matrix2(0, 0), 2.0, EPSILON);
  EXPECT_NEAR(matrix2(0, 1), 4.0, EPSILON);
  EXPECT_NEAR(matrix2(3, 4), 40.0, EPSILON);
}

TEST(test_assignment_operator, minus_assignment1) {
  S21Matrix matrix{4, 5};
  fillMatrixStep(matrix, 2.0);
  S21Matrix matrix2{4, 5};
  fillMatrixValue(matrix2, 2.0);
  matrix2 -= matrix;
  ASSERT_EQ(matrix2.GetRows(), 4);
  ASSERT_EQ(matrix2.GetCols(), 5);
  EXPECT_NEAR(matrix2(0, 0), 2.0, EPSILON);
  EXPECT_NEAR(matrix2(0, 1), 0.0, EPSILON);
  EXPECT_NEAR(matrix2(3, 4), -36.0, EPSILON);
}

TEST(test_assignment_operator, multiplication_assignment1) {
  S21Matrix matrix{4, 5};
  fillMatrixStep(matrix, 2.0);
  matrix *= 2;
  ASSERT_EQ(matrix.GetRows(), 4);
  ASSERT_EQ(matrix.GetCols(), 5);
  EXPECT_NEAR(matrix(0, 0), 0, EPSILON);
  EXPECT_NEAR(matrix(0, 1), 4.0, EPSILON);
  EXPECT_NEAR(matrix(3, 4), 76, EPSILON);
}

TEST(test_assignment_operator, multiplication_assignment2) {
  S21Matrix matrix{4, 5};
  fillMatrixStep(matrix, 2.0);
  S21Matrix matrix2{5, 4};
  fillMatrixStep(matrix2, 1.0);
  matrix2 *= matrix;
  ASSERT_EQ(matrix2.GetRows(), 5);
  ASSERT_EQ(matrix2.GetCols(), 5);
  EXPECT_NEAR(matrix2(0, 0), 140.0, EPSILON);
  EXPECT_NEAR(matrix2(0, 1), 152.0, EPSILON);
  EXPECT_NEAR(matrix2(4, 4), 1660.0, EPSILON);
}

TEST(test_plus_operator, plus_operator1) {
  S21Matrix matrix{4, 5};
  S21Matrix matrix2{5, 5};
  S21Matrix matrix3{5, 4};
  EXPECT_ANY_THROW(matrix2 = matrix2 + matrix);
  EXPECT_ANY_THROW(matrix3 = matrix2 + matrix3);
  EXPECT_ANY_THROW(matrix3 = matrix + matrix3);
}

TEST(test_plus_operator, plus_operator2) {
  S21Matrix matrix{3, 2};
  S21Matrix matrix2{3, 2};
  fillMatrixStep(matrix, 2.0);
  fillMatrixValue(matrix2, 2.0);
  S21Matrix matrix3 = matrix + matrix2;
  ASSERT_EQ(matrix3.GetRows(), 3);
  ASSERT_EQ(matrix3.GetCols(), 2);
  EXPECT_NEAR(matrix3(0, 0), 2.0, EPSILON);
  EXPECT_NEAR(matrix3(0, 1), 4.0, EPSILON);
  EXPECT_NEAR(matrix3(2, 1), 12.0, EPSILON);
}

TEST(test_minus_operator, minus_operator1) {
  S21Matrix matrix{4, 5};
  S21Matrix matrix2{5, 5};
  S21Matrix matrix3{5, 4};
  EXPECT_ANY_THROW(matrix2 = matrix2 - matrix);
  EXPECT_ANY_THROW(matrix3 = matrix2 - matrix3);
  EXPECT_ANY_THROW(matrix3 = matrix - matrix3);
}

TEST(test_minus_operator, minus_operator2) {
  S21Matrix matrix{3, 2};
  S21Matrix matrix2{3, 2};
  fillMatrixStep(matrix, 2.0);
  fillMatrixValue(matrix2, 2.0);
  S21Matrix matrix3 = matrix - matrix2;
  ASSERT_EQ(matrix3.GetRows(), 3);
  ASSERT_EQ(matrix3.GetCols(), 2);
  EXPECT_NEAR(matrix3(0, 0), -2.0, EPSILON);
  EXPECT_NEAR(matrix3(0, 1), 0.0, EPSILON);
  EXPECT_NEAR(matrix3(2, 1), 8.0, EPSILON);
}

TEST(test_multiplication_operator, multiplication_operator1) {
  S21Matrix matrix{3, 2};
  fillMatrixStep(matrix, 2.0);
  S21Matrix matrix2 = matrix * -2.0;
  ASSERT_EQ(matrix2.GetRows(), 3);
  ASSERT_EQ(matrix2.GetCols(), 2);
  EXPECT_NEAR(matrix2(0, 0), 0.0, EPSILON);
  EXPECT_NEAR(matrix2(0, 1), -4.0, EPSILON);
  EXPECT_NEAR(matrix2(2, 1), -20.0, EPSILON);
}

TEST(test_multiplication_operator, multiplication_operator2) {
  S21Matrix matrix{3, 2};
  fillMatrixStep(matrix, 2.0);
  S21Matrix matrix2 = -2.0 * matrix;
  ASSERT_EQ(matrix2.GetRows(), 3);
  ASSERT_EQ(matrix2.GetCols(), 2);
  EXPECT_NEAR(matrix2(0, 0), 0.0, EPSILON);
  EXPECT_NEAR(matrix2(0, 1), -4.0, EPSILON);
  EXPECT_NEAR(matrix2(2, 1), -20.0, EPSILON);
}

TEST(test_multiplication_operator, multiplication_operator3) {
  S21Matrix matrix{3, 2};
  S21Matrix matrix2{2, 3};
  fillMatrixStep(matrix, 2.0);
  fillMatrixStep(matrix2, -1.0);
  S21Matrix matrix3 = matrix * matrix2;
  ASSERT_EQ(matrix3.GetRows(), 3);
  ASSERT_EQ(matrix3.GetCols(), 3);
  EXPECT_NEAR(matrix3(0, 0), -6.0, EPSILON);
  EXPECT_NEAR(matrix3(0, 1), -8.0, EPSILON);
  EXPECT_NEAR(matrix3(2, 2), -66.0, EPSILON);
}

TEST(test_multiplication_operator, multiplication_operator4) {
  S21Matrix matrix{3, 3};
  S21Matrix matrix2{2, 4};
  S21Matrix matrix3;
  EXPECT_ANY_THROW(matrix3 = matrix2 * matrix);
}

TEST(test_equality_operator, equality_operator1) {
  S21Matrix matrix{2, 3};
  S21Matrix matrix2{2, 3};
  S21Matrix matrix3{2, 3};
  S21Matrix matrix4{3, 3};
  fillMatrixStep(matrix, 2.0);
  fillMatrixStep(matrix2, 2.0);
  EXPECT_TRUE(matrix == matrix2);
  EXPECT_FALSE(matrix == matrix3);
  EXPECT_FALSE(matrix4 == matrix3);
}

TEST(test_index_operator, index_operator1) {
  S21Matrix matrix{1, 1};
  EXPECT_NO_THROW(matrix(0, 0));
  EXPECT_ANY_THROW(matrix(-1, 0));
  EXPECT_ANY_THROW(matrix(0, -1));
  EXPECT_ANY_THROW(matrix(-1, -1));
  EXPECT_ANY_THROW(matrix(0, 1));
  EXPECT_ANY_THROW(matrix(1, 0));
  EXPECT_ANY_THROW(matrix(1, 1));
}

TEST(test_eqmatrix_method, eqmatrix_method1) {
  S21Matrix matrix{2, 3};
  S21Matrix matrix2{2, 3};
  S21Matrix matrix3{2, 3};
  S21Matrix matrix4{3, 3};
  fillMatrixStep(matrix, 2.0);
  fillMatrixStep(matrix2, 2.0);
  EXPECT_TRUE(matrix.EqMatrix(matrix2));
  EXPECT_FALSE(matrix.EqMatrix(matrix3));
  EXPECT_FALSE(matrix4.EqMatrix(matrix3));
}

TEST(test_summatrix_method, summatrix_method1) {
  S21Matrix matrix{4, 5};
  S21Matrix matrix2{5, 5};
  S21Matrix matrix3{5, 4};
  EXPECT_ANY_THROW(matrix2.SumMatrix(matrix));
  EXPECT_ANY_THROW(matrix3.SumMatrix(matrix2));
  EXPECT_ANY_THROW(matrix3.SumMatrix(matrix));
}

TEST(test_summatrix_method, summatrix_method2) {
  S21Matrix matrix{3, 2};
  S21Matrix matrix2{3, 2};
  fillMatrixStep(matrix, 2.0);
  fillMatrixValue(matrix2, 2.0);
  matrix.SumMatrix(matrix2);
  ASSERT_EQ(matrix.GetRows(), 3);
  ASSERT_EQ(matrix.GetCols(), 2);
  EXPECT_NEAR(matrix(0, 0), 2.0, EPSILON);
  EXPECT_NEAR(matrix(0, 1), 4.0, EPSILON);
  EXPECT_NEAR(matrix(2, 1), 12.0, EPSILON);
}

TEST(test_submatrix_method, submatrix_method1) {
  S21Matrix matrix{4, 5};
  S21Matrix matrix2{5, 5};
  S21Matrix matrix3{5, 4};
  EXPECT_ANY_THROW(matrix2.SubMatrix(matrix));
  EXPECT_ANY_THROW(matrix3.SubMatrix(matrix2));
  EXPECT_ANY_THROW(matrix3.SubMatrix(matrix));
}

TEST(test_submatrix_method, submatrix_method2) {
  S21Matrix matrix{3, 2};
  S21Matrix matrix2{3, 2};
  fillMatrixStep(matrix, 2.0);
  fillMatrixValue(matrix2, 2.0);
  matrix.SubMatrix(matrix2);
  ASSERT_EQ(matrix.GetRows(), 3);
  ASSERT_EQ(matrix.GetCols(), 2);
  EXPECT_NEAR(matrix(0, 0), -2.0, EPSILON);
  EXPECT_NEAR(matrix(0, 1), 0.0, EPSILON);
  EXPECT_NEAR(matrix(2, 1), 8.0, EPSILON);
}

TEST(test_mulnumber_method, mulnumber_method1) {
  S21Matrix matrix{3, 2};
  fillMatrixStep(matrix, 2.0);
  matrix.MulNumber(3.0);
  ASSERT_EQ(matrix.GetRows(), 3);
  ASSERT_EQ(matrix.GetCols(), 2);
  EXPECT_NEAR(matrix(0, 0), 0.0, EPSILON);
  EXPECT_NEAR(matrix(0, 1), 6.0, EPSILON);
  EXPECT_NEAR(matrix(2, 1), 30.0, EPSILON);
}

TEST(test_mulmatrix_method, mulmatrix_method1) {
  S21Matrix matrix{3, 3};
  S21Matrix matrix2{2, 4};
  EXPECT_ANY_THROW(matrix2.MulMatrix(matrix));
}

TEST(test_mulmatrix_method, mulmatrix_method2) {
  S21Matrix matrix{3, 2};
  S21Matrix matrix2{2, 3};
  fillMatrixStep(matrix, 2.0);
  fillMatrixStep(matrix2, -1.0);
  matrix.MulMatrix(matrix2);
  ASSERT_EQ(matrix.GetRows(), 3);
  ASSERT_EQ(matrix.GetCols(), 3);
  EXPECT_NEAR(matrix(0, 0), -6.0, EPSILON);
  EXPECT_NEAR(matrix(0, 1), -8.0, EPSILON);
  EXPECT_NEAR(matrix(2, 2), -66.0, EPSILON);
}

TEST(test_transpose_method, transpose_method1) {
  S21Matrix matrix{3, 2};
  fillMatrixStep(matrix, 2.0);
  matrix = matrix.Transpose();
  ASSERT_EQ(matrix.GetRows(), 2);
  ASSERT_EQ(matrix.GetCols(), 3);
  EXPECT_NEAR(matrix(0, 0), 0.0, EPSILON);
  EXPECT_NEAR(matrix(0, 1), 4.0, EPSILON);
  EXPECT_NEAR(matrix(1, 2), 10.0, EPSILON);
}

TEST(test_determinant_method, determinant_method1) {
  S21Matrix matrix{3, 2};
  EXPECT_ANY_THROW(matrix.Determinant());
}

TEST(test_determinant_method, determinant_method2) {
  S21Matrix matrix{1, 1};
  EXPECT_NEAR(matrix.Determinant(), 0.0, EPSILON);
  S21Matrix matrix2{3, 3};
  EXPECT_NEAR(matrix2.Determinant(), 0.0, EPSILON);
}

TEST(test_determinant_method, determinant_method3) {
  S21Matrix matrix{2, 2};
  double arr[4] = {1.0, 3.0, -2.0, 5.0};
  fillMatrixArray(matrix, arr, 4);
  EXPECT_NEAR(matrix.Determinant(), 11.0, EPSILON);
}

TEST(test_determinant_method, determinant_method4) {
  S21Matrix matrix{3, 3};
  double arr[9] = {1.0, 0.0, -2.0, 0.5, 3.0, 1.0, 0.0, 2.0, -1.0};
  fillMatrixArray(matrix, arr, 9);
  EXPECT_NEAR(matrix.Determinant(), -7.0, EPSILON);
}

TEST(test_calccomplements_method, calccomplements_method1) {
  S21Matrix matrix{3, 2};
  EXPECT_ANY_THROW(matrix.CalcComplements());
}

TEST(test_calccomplements_method, calccomplements_method2) {
  S21Matrix matrix{2, 2};
  double arr[4] = {1.0, 2.0, 3.0, 4.0};
  fillMatrixArray(matrix, arr, 4);
  S21Matrix matrix2 = matrix.CalcComplements();
  ASSERT_EQ(matrix2.GetRows(), 2);
  ASSERT_EQ(matrix2.GetCols(), 2);
  EXPECT_NEAR(matrix2(0, 0), 4.0, EPSILON);
  EXPECT_NEAR(matrix2(0, 1), -3.0, EPSILON);
  EXPECT_NEAR(matrix2(1, 0), -2.0, EPSILON);
  EXPECT_NEAR(matrix2(1, 1), 1.0, EPSILON);
}

TEST(test_calccomplements_method, calccomplements_method3) {
  S21Matrix matrix{3, 3};
  double arr[9] = {2.0, 5.0, 7.0, 6.0, 3.0, 4.0, 5.0, -2.0, -3.0};
  fillMatrixArray(matrix, arr, 9);
  S21Matrix matrix2 = matrix.CalcComplements();
  ASSERT_EQ(matrix2.GetRows(), 3);
  ASSERT_EQ(matrix2.GetCols(), 3);
  EXPECT_NEAR(matrix2(0, 0), -1.0, EPSILON);
  EXPECT_NEAR(matrix2(0, 1), 38.0, EPSILON);
  EXPECT_NEAR(matrix2(0, 2), -27.0, EPSILON);
  EXPECT_NEAR(matrix2(1, 0), 1.0, EPSILON);
  EXPECT_NEAR(matrix2(1, 1), -41.0, EPSILON);
  EXPECT_NEAR(matrix2(1, 2), 29.0, EPSILON);
  EXPECT_NEAR(matrix2(2, 0), -1.0, EPSILON);
  EXPECT_NEAR(matrix2(2, 1), 34.0, EPSILON);
  EXPECT_NEAR(matrix2(2, 2), -24.0, EPSILON);
}

TEST(test_calccomplements_method, calccomplements_method4) {
  S21Matrix matrix{1, 1};
  matrix(0, 0) = 5;
  EXPECT_ANY_THROW(matrix.CalcComplements());
}

TEST(test_inversematrix_method, inversematrix_method1) {
  S21Matrix matrix{2, 3};
  double arr[6] = {2.0, 5.0, 7.0, 6.0, 3.0, 4.0};
  fillMatrixArray(matrix, arr, 6);
  EXPECT_ANY_THROW(matrix.InverseMatrix());
  S21Matrix matrix2{3, 3};
  EXPECT_ANY_THROW(matrix2.InverseMatrix());
  S21Matrix matrix3{1, 1};
  EXPECT_ANY_THROW(matrix3.InverseMatrix());
}

TEST(test_inversematrix_method, inversematrix_method2) {
  S21Matrix matrix{1, 1};
  matrix(0, 0) = -5;
  S21Matrix matrix2 = matrix.InverseMatrix();
  EXPECT_NEAR(matrix2(0, 0), -0.2, EPSILON);
}

TEST(test_inversematrix_method, inversematrix_method3) {
  S21Matrix matrix{2, 2};
  double arr[4] = {1.0, -1.0, 2.0, 4.0};
  fillMatrixArray(matrix, arr, 4);
  S21Matrix matrix2 = matrix.InverseMatrix();
  ASSERT_EQ(matrix2.GetRows(), 2);
  ASSERT_EQ(matrix2.GetCols(), 2);
  EXPECT_NEAR(matrix2(0, 0), (2.0 / 3.0), EPSILON);
  EXPECT_NEAR(matrix2(0, 1), (1.0 / 6.0), EPSILON);
  EXPECT_NEAR(matrix2(1, 0), (-1.0 / 3.0), EPSILON);
  EXPECT_NEAR(matrix2(1, 1), (1.0 / 6.0), EPSILON);
}

TEST(test_inversematrix_method, inversematrix_method4) {
  S21Matrix matrix{3, 3};
  double arr[9] = {2.0, 3.0, -1.0, 0.0, 1.0, 1.0, 1.0, -1.0, 2.0};
  fillMatrixArray(matrix, arr, 9);
  S21Matrix matrix2 = matrix.InverseMatrix();
  ASSERT_EQ(matrix2.GetRows(), 3);
  ASSERT_EQ(matrix2.GetCols(), 3);
  EXPECT_NEAR(matrix2(0, 0), 0.3, EPSILON);
  EXPECT_NEAR(matrix2(0, 1), -0.5, EPSILON);
  EXPECT_NEAR(matrix2(0, 2), 0.4, EPSILON);
  EXPECT_NEAR(matrix2(1, 0), 0.1, EPSILON);
  EXPECT_NEAR(matrix2(1, 1), 0.5, EPSILON);
  EXPECT_NEAR(matrix2(1, 2), -0.2, EPSILON);
  EXPECT_NEAR(matrix2(2, 0), -0.1, EPSILON);
  EXPECT_NEAR(matrix2(2, 1), 0.5, EPSILON);
  EXPECT_NEAR(matrix2(2, 2), 0.2, EPSILON);
}

TEST(test_accessor_methods, test_accessor_methods1) {
  S21Matrix matrix;
  EXPECT_EQ(matrix.GetRows(), DEFAULTRAWS);
  EXPECT_EQ(matrix.GetCols(), DEFAULTCOLS);
}

TEST(test_accessor_methods, test_accessor_methods2) {
  S21Matrix matrix{3, 4};
  EXPECT_EQ(matrix.GetRows(), 3);
  EXPECT_EQ(matrix.GetCols(), 4);
}

TEST(test_mutator_methods, test_mutator_methods1) {
  S21Matrix matrix{1, 1};
  EXPECT_ANY_THROW(matrix.SetRows(0));
  EXPECT_ANY_THROW(matrix.SetRows(-5));
  EXPECT_ANY_THROW(matrix.SetCols(0));
  EXPECT_ANY_THROW(matrix.SetCols(-3));
  EXPECT_ANY_THROW(matrix.SetRowsCols(0, 2));
  EXPECT_ANY_THROW(matrix.SetRowsCols(2, 0));
  EXPECT_ANY_THROW(matrix.SetRowsCols(-1, 2));
  EXPECT_ANY_THROW(matrix.SetRowsCols(2, -2));
}

TEST(test_mutator_methods, test_mutator_methods2) {
  S21Matrix matrix{1, 1};
  EXPECT_EQ(matrix.GetRows(), 1);
  EXPECT_EQ(matrix.GetCols(), 1);
  matrix.SetRows(2);
  EXPECT_EQ(matrix.GetRows(), 2);
  EXPECT_EQ(matrix.GetCols(), 1);
  matrix.SetCols(5);
  EXPECT_EQ(matrix.GetRows(), 2);
  EXPECT_EQ(matrix.GetCols(), 5);
}

TEST(test_mutator_methods, test_mutator_methods3) {
  S21Matrix matrix{3, 3};
  EXPECT_EQ(matrix.GetRows(), 3);
  EXPECT_EQ(matrix.GetCols(), 3);
  fillMatrixStep(matrix, 2.0);
  EXPECT_NEAR(matrix(2, 2), 16.0, EPSILON);
  matrix.SetRows(2);
  EXPECT_EQ(matrix.GetRows(), 2);
  EXPECT_EQ(matrix.GetCols(), 3);
  matrix.SetRows(4);
  matrix.SetCols(5);
  EXPECT_EQ(matrix.GetRows(), 4);
  EXPECT_EQ(matrix.GetCols(), 5);
  EXPECT_NEAR(matrix(0, 0), 0.0, EPSILON);
  EXPECT_NEAR(matrix(0, 1), 2.0, EPSILON);
  EXPECT_NEAR(matrix(0, 4), 0.0, EPSILON);
  EXPECT_NEAR(matrix(1, 2), 10.0, EPSILON);
  EXPECT_NEAR(matrix(2, 2), 0.0, EPSILON);
  EXPECT_NEAR(matrix(3, 4), 0.0, EPSILON);
  matrix.SetCols(1);
  matrix.SetCols(5);
  EXPECT_NEAR(matrix(0, 1), 0.0, EPSILON);
}

TEST(test_mutator_methods, test_mutator_methods4) {
  S21Matrix matrix{3, 3};
  EXPECT_EQ(matrix.GetRows(), 3);
  EXPECT_EQ(matrix.GetCols(), 3);
  fillMatrixStep(matrix, 2.0);
  EXPECT_NEAR(matrix(2, 2), 16.0, EPSILON);
  matrix.SetRowsCols(2, 5);
  EXPECT_EQ(matrix.GetRows(), 2);
  EXPECT_EQ(matrix.GetCols(), 5);
  matrix.SetRowsCols(4, 5);
  EXPECT_EQ(matrix.GetRows(), 4);
  EXPECT_EQ(matrix.GetCols(), 5);
  EXPECT_NEAR(matrix(0, 0), 0.0, EPSILON);
  EXPECT_NEAR(matrix(0, 1), 2.0, EPSILON);
  EXPECT_NEAR(matrix(0, 4), 0.0, EPSILON);
  EXPECT_NEAR(matrix(1, 2), 10.0, EPSILON);
  EXPECT_NEAR(matrix(2, 2), 0.0, EPSILON);
  EXPECT_NEAR(matrix(3, 4), 0.0, EPSILON);
  matrix.SetRowsCols(4, 1);
  matrix.SetRowsCols(4, 5);
  EXPECT_NEAR(matrix(0, 1), 0.0, EPSILON);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}