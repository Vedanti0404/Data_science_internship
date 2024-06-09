def lower_triangular(n):
    """Creates a lower triangular pattern."""
    for i in range(1, n + 1):
        print('* ' * i)
        
def upper_triangular(n):
    """Creates an upper triangular pattern."""
    for i in range(n, 0, -1):
        print('* ' * i)

def pyramid(n):
    """Creates a pyramid pattern."""
    for i in range(1, n + 1):
        print(' ' * (n - i) + '* ' * i)

# Example usage:
n = 5  # You can change this value to create larger or smaller patterns

print("Lower Triangular Pattern:")
lower_triangular(n)
print("\nUpper Triangular Pattern:")
upper_triangular(n)
print("\nPyramid Pattern:")
pyramid(n)
