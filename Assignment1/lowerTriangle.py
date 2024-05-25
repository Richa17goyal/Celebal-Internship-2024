def lower_triangle(n):
    for i in range(n):
        for j in range(i + 1):
            print('*', end=' ')
        print()


n = 5
print("Lower Triangular Pattern:")
lower_triangle(n)

