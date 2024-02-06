CC				= g++
CFLAGS		= -Wall -Werror -Wextra -std=c++17

all: clean s21_matrix_oop.a gcov_report

clean:
	@rm -rf report *.o *.a *.gcda *.gcno *.info *.out *.txt

test:
	@	@$(CC) $(CFLAGS) tests/tests.cpp tests/tests_functions.cpp s21_matrix_oop.cpp -lgtest -pthread -o tests.out
	@./tests.out

s21_matrix_oop.a:
	@$(CC) $(CFLAGS) -c s21_matrix_oop.cpp
	@ar rc s21_matrix_oop.a s21_matrix_oop.o
	@ranlib s21_matrix_oop.a
	@rm s21_matrix_oop.o

gcov_report:
	@$(CC) $(CFLAGS) tests/tests.cpp tests/tests_functions.cpp s21_matrix_oop.cpp -lgtest --coverage -pthread -o report.out
	@./report.out
	@lcov -t "report" -o report.info --no-external -c -d .
	@genhtml -o ./report report.info
	@rm -rf *.gcda *.gcno *.info *.out

style_check:
	@clang-format -style=Google -n *.cpp tests/*.cpp *.h tests/*.h

style_fix:
	@clang-format -style=Google -i *.cpp tests/*.cpp *.h tests/*.h

cppcheck:
	@cppcheck --enable=all --suppress=missingIncludeSystem --language=c++ *.cpp tests/*.cpp *.h tests/*.h

check: style_check cppcheck

leaks: clean test
	@valgrind --leak-check=full --show-reachable=yes --show-leak-kinds=all --track-origins=yes --log-file=valgrind.txt ./tests.out

.PHONY: all clean test s21_matrix_oop.a gcov_report style_check style_fix cppcheck leaks check