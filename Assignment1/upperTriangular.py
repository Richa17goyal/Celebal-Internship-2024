def upper_triangle(n):
    for i in range(n):
        for j in range(n):
            if j >= i:
                print('*', end=' ')
            else:
                print(' ', end=' ')
        print()

print("\nUpper Triangular Pattern:")
upper_triangle(5)
