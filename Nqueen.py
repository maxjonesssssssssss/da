def solve_nqueens(n):
    solutions = []
    board = [-1] * n

    def backtrack(row, cols, diag1, diag2):
        if row == n:
            solutions.append(board[:])
            return
        for col in range(n):
            if col in cols or (row - col) in diag1 or (row + col) in diag2:
                continue
            board[row] = col
            backtrack(row + 1, cols | {col}, diag1 | {row - col}, diag2 | {row + col})
            board[row] = -1

    backtrack(0, set(), set(), set())
    return solutions

def print_board(board):
    n = len(board)
    for row in range(n):
        line = ""
        for col in range(n):
            line += "Q " if board[row] == col else ". "
        print(" ", line)
    print()

if __name__ == "__main__":
    print("=" * 40)
    print("   N-Queens: Backtracking + Branch & Bound")
    print("=" * 40)

    while True:
        try:
            n = int(input("\n Enter the value of N (board size): "))
            if n < 1:
                print(" Please enter a positive integer.")
                continue
            break
        except ValueError:
            print(" Invalid input. Enter a number.")

    print(f"\n Solving {n}-Queens problem...\n")
    solutions = solve_nqueens(n)

    if not solutions:
        print(f" No solutions exist for {n}-Queens.\n")
    else:
        print(f" Total solutions found: {len(solutions)}\n")
        print("-" * 40)
        for i, sol in enumerate(solutions, 1):
            print(f" Solution {i}:")
            print_board(sol)
