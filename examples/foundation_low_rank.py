"""Matrix geometry, singular values, truncated low rank and LoRA shape checks.

This is a small, pure-Python numerical exercise; NOT a general-purpose SVD
implementation. Run: python examples/foundation_low_rank.py
"""
from __future__ import annotations

import math


def dot(a, b):
    if len(a) != len(b):
        raise ValueError("dot product dimensions differ")
    return sum(x*y for x, y in zip(a, b))


def matrix_times_vector(matrix, vector):
    if not matrix or any(len(row) != len(vector) for row in matrix):
        raise ValueError("matrix/vector dimensions differ")
    return [dot(row, vector) for row in matrix]


def multiply(a, b):
    if not a or not b or not a[0] or not b[0]:
        raise ValueError("empty matrix")
    if any(len(row) != len(a[0]) for row in a):
        raise ValueError("ragged left matrix")
    if any(len(row) != len(b[0]) for row in b):
        raise ValueError("ragged right matrix")
    if len(a[0]) != len(b):
        raise ValueError("incompatible matrix dimensions")
    return [
        [sum(a[i][k] * b[k][j] for k in range(len(b)))
         for j in range(len(b[0]))]
        for i in range(len(a))
    ]


def transpose(matrix):
    if not matrix or any(len(row) != len(matrix[0]) for row in matrix):
        raise ValueError("empty or ragged matrix")
    return [list(row) for row in zip(*matrix)]


def projection(x, u):
    norm_squared = dot(u, u)
    if norm_squared <= 0:
        raise ValueError("projection onto zero vector")
    factor = dot(x, u) / norm_squared
    return [factor * coordinate for coordinate in u]


def singular_values_2x2(a):
    """2D singular values from eigenvalues of A^T A; descending."""
    if len(a) != 2 or any(len(row) != 2 for row in a):
        raise ValueError("requires a 2-by-2 matrix")
    gram = multiply(transpose(a), a)
    trace = gram[0][0] + gram[1][1]
    determinant = gram[0][0]*gram[1][1] - gram[0][1]*gram[1][0]
    discriminant = max(0.0, trace*trace - 4*determinant)
    lambda_max = max(0.0, (trace + math.sqrt(discriminant))/2)
    lambda_min = max(0.0, (trace - math.sqrt(discriminant))/2)
    return math.sqrt(lambda_max), math.sqrt(lambda_min)


def frobenius_distance(a, b):
    if not a or len(a) != len(b) or any(
        len(row_a) != len(row_b) for row_a, row_b in zip(a, b)
    ):
        raise ValueError("incompatible shapes")
    return math.sqrt(sum(
        (x-y)**2
        for row_a, row_b in zip(a,b)
        for x, y in zip(row_a, row_b)
    ))


def parameter_counts(d_in, d_out, rank):
    if not all(type(x) is int and x > 0 for x in (d_in, d_out, rank)):
        raise ValueError("positive integer dimensions required")
    return d_in*d_out, rank*(d_in+d_out)


def verify_projection():
    x = [2.0, 1.0]
    u = [1.0, 1.0]
    p = projection(x, u)
    assert all(math.isclose(a,b,abs_tol=1e-12)
               for a,b in zip(p,[1.5,1.5]))
    residual = [x[i]-p[i] for i in range(2)]
    assert math.isclose(dot(residual, u),0.0,abs_tol=1e-12)
    try:
        projection(x,[0.0,0.0])
        raise AssertionError("zero projection axis should fail")
    except ValueError:
        pass


def verify_singular_values():
    original = [[3.0,0.0],[0.0,1.0]]
    s1,s2=singular_values_2x2(original)
    assert math.isclose(s1,3.0,abs_tol=1e-12)
    assert math.isclose(s2,1.0,abs_tol=1e-12)
    rank_one = [[3.0,0.0],[0.0,0.0]]
    assert math.isclose(frobenius_distance(original,rank_one),1.0)
    worse = [[2.0,0.0],[0.0,0.0]]
    assert math.isclose(frobenius_distance(original,worse),
                        math.sqrt(2),abs_tol=1e-12)
    dependent = [[1.0,2.0],[2.0,4.0]]
    singulars = singular_values_2x2(dependent)
    assert math.isclose(singulars[0],5.0,abs_tol=1e-12)
    assert math.isclose(singulars[1],0.0,abs_tol=1e-12)


def verify_lora():
    original, additional = parameter_counts(4096,4096,8)
    assert original == 16_777_216
    assert additional == 65_536
    assert additional/original == 1/256
    original2, additional2 = parameter_counts(4096,2048,16)
    assert original2 == 8_388_608 and additional2 == 98_304
    assert additional2/original2 == 3/256

    # A: r x d_in, B: d_out x r, so B A is d_out x d_in.
    A = [[1.0, -2.0]]
    B = [[3.0], [4.0]]
    delta = multiply(B,A)
    assert delta == [[3.0,-6.0],[4.0,-8.0]]
    # Rank one determinant = 0.
    assert math.isclose(
        delta[0][0]*delta[1][1] - delta[0][1]*delta[1][0],
        0.0,abs_tol=1e-12
    )
    x=[5.0,7.0]
    direct=matrix_times_vector(delta,x)
    factored=matrix_times_vector(B,matrix_times_vector(A,x))
    assert direct == factored

    try:
        parameter_counts(2048,4096,0)
        raise AssertionError("rank zero must fail under this contract")
    except ValueError:
        pass


if __name__=="__main__":
    verify_projection()
    verify_singular_values()
    verify_lora()
    print("PASS: orthogonal projection and residual")
    print("PASS: 2x2 singular values, optimal diagonal rank-one toy example")
    print("PASS: LoRA parameter counts, low-rank shape and factorization")
