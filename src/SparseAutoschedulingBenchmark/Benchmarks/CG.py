"""https://arxiv.org/abs/2007.00640 Page 21"""


def benchmark_cg(
    xp, A_bench, b_bench, x_bench, rel_tol=1e-8, abs_tol=1e-20, max_iters=10_000
):
    A = xp.lazy(xp.from_benchmark(A_bench))
    b = xp.lazy(xp.from_benchmark(b_bench))
    x = xp.lazy(xp.from_benchmark(x_bench))

    tolerance = max(
        xp.compute(xp.lazy(rel_tol) * xp.sqrt(xp.vecdot(b, b)))[()], abs_tol
    )
    # tol_sq used to avoid having to sqrt dot products when checking tolerance
    tol_sq = tolerance * tolerance
    max_iters = min(max_iters, xp.shape(A)[0])

    r = b - A @ x
    p = r
    it = 0
    rr = xp.compute(xp.vecdot(r, r))[()]

    while rr >= tol_sq and it < max_iters:
        Ap = A @ p
        alpha = rr / xp.vecdot(r, Ap)
        x += alpha * p
        r -= alpha * Ap

        new_rr = xp.compute(xp.vecdot(r, r))[()]

        if new_rr <= tol_sq:
            break

        beta = new_rr / rr
        p = r + beta * p
        rr = new_rr

        x = xp.lazy(xp.compute(x))
        r = xp.lazy(xp.compute(r))
        p = xp.lazy(xp.compute(p))
        it += 1
    x_solution = xp.compute(x)
    return xp.to_benchmark(x_solution)
