import math
'''
Exact polynomial arithmetic over the rationals.

Coefficients are stored from lowest to highest degree.

>>> p = Poly([1, 2, 3])
>>> print(p)
3x**2 + 2x + 1

Polynomials are just functions with special structure,
and they can be called like regular functions.

>>> p(3)
34

Because of this special structure, we can do a lot of operations that can not be done on ordinary functions.

We can compute derivatives:

>>> print(p.derivative())
6x + 2

We can check for equality:

>>> q = Poly([0, 3, 2, 1])
>>> print(q)
x**3 + 2x**2 + 3x
>>> p == q
False

We can do addition, subtraction, multiplication, and scalar exponentiation:

>>> print (p + q)
x**3 + 5x**2 + 5x + 1
>>> print (p - q)
-x**3 + x**2 - x + 1
>>> print (p * q)
3x**5 + 8x**4 + 14x**3 + 8x**2 + 3x
>>> print(p ** 2)
9x**4 + 12x**3 + 10x**2 + 4x + 1

Technically, polynomials form a *Ring*
(a type of structure studied in MATH171 Abstract Algebra),
and so they do not support division.
(If they did, they would be called a *Field*.)
'''


class Poly:
    '''
    A dense polynomial with rational coefficients.

    The field .coeffs is a list of length at least 1, from lowest to
    highest degree, with no trailing zeros.  Coeffs[-1] is the leading
    coefficient.

    >>> p = Poly([1, 2, 3])
    >>> p.coeffs
    [1, 2, 3]
    >>> Poly([1, 2, 0, 0]).coeffs
    [1, 2]
    >>> Poly([]).coeffs
    [0]

    The `==` operator compares polynomials by value, and polynomials
    that represent the same function are equal regardless of how their
    coefficient lists were passed in:

    >>> Poly([1, 2, 3]) == Poly([1, 2, 3, 0, 0])
    True
    >>> Poly([1, 2]) == 1
    False

    `Poly` is a mutable value type, so it is deliberately unhashable
    and cannot be used as a dict key or a set element:

    >>> hash(Poly([1, 2, 3]))
    Traceback (most recent call last):
        ...
    TypeError: unhashable type: 'Poly'

    The `repr` is the constructor call, so `eval(repr(p)) == p` for
    every Poly.  The `str` is the usual algebraic notation, with terms
    in descending degree, unit coefficients suppressed, and zero
    coefficients skipped.  The zero polynomial prints as `0`:

    >>> eval(repr(Poly([1, 2, 3]))) == Poly([1, 2, 3])
    True
    >>> print(Poly([1, 2, 3]))
    3x**2 + 2x + 1
    >>> print(Poly([0]))
    0
    >>> print(Poly([5]))
    5
    >>> print(Poly([-5]))
    -5
    >>> print(Poly([1, 1]))
    x + 1
    >>> print(Poly([0, 1]))
    x
    >>> print(Poly([0, 0, -1]))
    -x**2
    >>> print(Poly([1, -1, 0, 1]))
    x**3 - x + 1
    >>> print(Poly([1, 0, 2]))
    2x**2 + 1
    '''
    def __init__(self, coeffs):
        coeffs = list(coeffs)
        while len(coeffs) > 0 and coeffs[-1] == 0:
            coeffs.pop()
        if len(coeffs) == 0:
            coeffs = [0]
        self.coeffs = coeffs

    def __eq__(self, other):
        if not isinstance(other, Poly):
            return False
        return self.coeffs == other.coeffs

    __hash__ = None

    def __repr__(self):
        return f'Poly({self.coeffs!r})'

    def __str__(self):
        terms = []
        for i in range(len(self.coeffs) - 1, -1, -1):
            c = self.coeffs[i]
            if c != 0:
                size = abs(c)
                if i == 0:
                    body = str(size)
                elif i == 1:
                    body = 'x' if size == 1 else f'{size}x'
                else:
                    if size == 1:
                        body = f'x**{i}'
                    else:
                        body = f'{size}x**{i}'
                terms.append((c < 0, body))
        if len(terms) == 0:
            return '0'
        xs = ''
        for pair in terms:
            negative = pair[0]
            body = pair[1]
            if xs == '':
                if negative:
                    xs = '-' + body
                else:
                    xs = body
            else:
                if negative:
                    xs = xs + ' - ' + body
                else:
                    xs = xs + ' + ' + body
        return xs

    def __call__(self, x):
        return self.eval_horner(x)

    def __add__(self, other):
        a, b = self.coeffs, other.coeffs
        n = max(len(a), len(b))
        a = a + [0] * (n - len(a))
        b = b + [0] * (n - len(b))
        return Poly([a[i] + b[i] for i in range(n)])

    def __neg__(self):
        return Poly([-c for c in self.coeffs])

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        return Poly(_karatsuba(self.coeffs, other.coeffs))

    def __pow__(self, n):
        result = Poly([1])
        for _ in range(n):
            result = result * self
        return result
    def derivative(self):
        new_coeffs = []
        for i in range(1, len(self.coeffs)):
            new_coeffs.append(i * self.coeffs[i])
        return Poly(new_coeffs)

    def eval_naive(self, x):
        '''
        Evaluate by summing a_i * x**i one term at a time.  The
        per-term `x ** i` recomputes a power each iteration, so the
        runtime is Theta(n) power lifts, each Theta(i) for the
        exponentiation, giving Theta(n**2) overall.

        >>> Poly([1, 2, 3]).eval_naive(2)
        17
        '''
        total = 0
        for i, c in enumerate(self.coeffs):
            total += c * x ** i
        return total

    def eval_horner(self, x):
        '''
        Evaluate with Horner's rule:

            (((a_n x + a_{n-1}) x + a_{n-2}) x + ...) x + a_0

        This is Theta(n) and uses only one multiply and one add per
        coefficient.  It is also numerically more stable than the
        naive evaluation when consecutive coefficients have opposite
        signs, because no large intermediate x**i is computed.

        >>> Poly([1, 2, 3]).eval_horner(2)
        17
        '''
        total = 0
        for c in reversed(self.coeffs):
            total = total * x + c
        return total

    def mul_naive(self, other):
        '''
        Naive multiplication uses Theta(n**2) multiplications.

        >>> Poly([1, 2]) .mul_naive(Poly([3, 4]))
        Poly([3, 10, 8])
        '''
        a, b = self.coeffs, other.coeffs
        if not a or not b:
            return Poly([0])
        out = [0] * (len(a) + len(b) - 1)
        for i, ai in enumerate(a):
            for j, bj in enumerate(b):
                out[i + j] += ai * bj
        return Poly(out)


def _karatsuba(a, b):
    '''
    Product of two coefficient lists by Karatsuba's algorithm.

    The inputs are coefficient lists, lowest degree first.
    The return value is the coefficient list of the product.
    The recursion splits both operands at the midpoint and
    uses only three half-size products.
    That gives the recurrence

        T(n) = 3 T(n/2) + Theta(n)
             = Theta(n^log2 3)          # by master theorem
             = O(n^1.586)               # by defn of O()

    See <https://www.geeksforgeeks.org/dsa/multiply-two-polynomials-2/> for a basic tutorial/implementation you can adapt for your code.

    >>> _karatsuba([1, 2], [3, 4])
    [3, 10, 8]
    >>> _karatsuba([1], [1])
    [1]
    >>> _karatsuba([0], [1, 2, 3])
    [0, 0, 0]
    '''
    n = max(len(a), len(b))
    if (n & (n - 1)) != 0:
        n = 1 << (math.floor(math.log2(n)) + 1)
    a_padded = a + [0] * (n - len(a))
    b_padded = b + [0] * (n - len(b))
    if n == 1:
        return [a_padded[0] * b_padded[0]]
    m = n // 2
    a0 = a_padded[:m]
    a1 = a_padded[m:]
    b0 = b_padded[:m]
    b1 = b_padded[m:]

    z0 = _karatsuba(a0, b0)
    z2 = _karatsuba(a1, b1)
    a0_plus_a1 = [a0[i] + a1[i] for i in range(m)]
    b0_plus_b1 = [b0[i] + b1[i] for i in range(m)]
    z1 = _karatsuba(a0_plus_a1, b0_plus_b1)
    middle_term = [z1[i] - z0[i] - z2[i] for i in range(len(z1))]

    result = [0] * (2 * n)
    for i in range(len(z0)):
        result[i] += z0[i]
    for i in range(len(middle_term)):
        result[m + i] += middle_term[i]
    for i in range(len(z2)):
        result[2 * m + i] += z2[i]

    return result[:len(a) + len(b) - 1]
