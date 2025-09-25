from dataclasses import dataclass
from lark import Lark
import numpy as np

class EinsumExpr:
    pass


nary_ops = {
    "+": "add",
    "add": "add",
    "-": "subtract",
    "sub": "subtract",
    "subtract": "subtract",
    "*": "multiply",
    "mul": "multiply",
    "multiply": "multiply",
    "/": "divide",
    "div": "divide",
    "divide": "divide",
    "//": "floor_divide",
    "fld": "floor_divide",
    "floor_divide": "floor_divide",
    "%": "remainder",
    "mod": "remainder",
    "remainder": "remainder",
    "**": "power",
    "pow": "power",
    "power": "power",
    "==": "equal",
    "eq": "equal",
    "equal": "equal",
    "!=": "not_equal",
    "ne": "not_equal",
    "not_equal": "not_equal",
    "<": "less",
    "lt": "less",
    "less": "less",
    "<=": "less_equal",
    "le": "less_equal",
    "less_equal": "less_equal",
    ">": "greater",
    "gt": "greater",
    "greater": "greater",
    ">=": "greater_equal",
    "ge": "greater_equal",
    "greater_equal": "greater_equal",
    "&": "bitwise_and",
    "bitwise_and": "bitwise_and",
    "|": "bitwise_or",
    "bitwise_or": "bitwise_or",
    "^": "bitwise_xor",
    "bitwise_xor": "bitwise_xor",
    "<<": "bitwise_left_shift",
    "lshift": "bitwise_left_shift",
    "bitwise_left_shift": "bitwise_left_shift",
    ">>": "bitwise_right_shift",
    "rshift": "bitwise_right_shift",
    "bitwise_right_shift": "bitwise_right_shift",
    "and": "logical_and",
    "or": "logical_or",
    "not": "logical_not",
    "min": "minimum",
    "max": "maximum",
    "logaddexp": "logaddexp",
}


unary_ops = {
    "+": "positive",
    "pos": "positive",
    "positive": "positive",
    "-": "negative",
    "neg": "negative",
    "negative": "negative",
    "~": "bitwise_invert",
    "invert": "bitwise_invert",
    "bitwise_invert": "bitwise_invert",
    "not": "logical_not",
    "logical_not": "logical_not",
    "~": "bitwise_invert",
    "invert": "bitwise_invert",
    "bitwise_invert": "bitwise_invert",
    "abs": "absolute",
    "absolute": "absolute",
    "sqrt": "sqrt",
    "exp": "exp",
    "log": "log",
    "log1p": "log1p",
    "log10": "log10",
    "log2": "log2",
    "neg": "negative",
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "sinh": "sinh",
    "cosh": "cosh",
    "tanh": "tanh",
    "asin": "arcsin",
    "acos": "arccos",
    "atan": "arctan",
    "asinh": "arcsinh",
    "acosh": "arccosh",
    "atanh": "arctanh",
}


reduction_ops = {
    "+": "sum",
    "add": "sum",
    "sum": "sum",
    "*": "prod",
    "mul": "prod",
    "prod": "prod",
    "and": "all",
    "or": "any",
    "min": "min",
    "max": "max",
    "argmin": "argmin",
    "argmax": "argmax",
    "mean": "mean",
    "std": "std",
    "var": "var",
    "count_nonzero": "count_nonzero",
    #"&": "bitwise_and",
    #"|": "bitwise_or",
    #"^": "bitwise_xor",
}

@dataclass
class Access(EinsumExpr):
    tns: str
    idxs: list[str]
    def get_loops(self) -> set[str]:
        return set(self.idxs)

    def run(self, xp, loops, kwargs):
        assert len(self.idxs) == len(set(self.idxs))
        perm = sorted(range(len(self.idxs)), key=lambda i: loops.index(self.idxs[i]))
        tns = kwargs[self.tns]
        tns = xp.transpose(tns, perm)
        return xp.expand_dims(tns, [i for i in range(len(loops)) if loops[i] not in self.idxs])

@dataclass
class Call(EinsumExpr):
    func: str
    args: list[EinsumExpr]

    def get_loops(self) -> set[str]:
        return set().union(*[arg.get_loops() for arg in self.args])
    
    def run(self, xp, idxs, kwargs):
        if len(self.args) == 1:
            func = getattr(xp, unary_ops[self.func])
        else:
            func = getattr(xp, nary_ops[self.func])
        vals = [arg.run(xp, idxs, kwargs) for arg in self.args]
        return func(*vals)
    

@dataclass
class Einsum:
    arg: EinsumExpr
    op: str
    tns: str
    idxs: str

    def run(self, xp, kwargs):
        loops = self.arg.get_loops()
        assert set(self.idxs).issubset(loops)
        loops = sorted(loops)
        arg = self.arg.run(xp, loops, kwargs)
        axis = tuple(i for i in range(len(loops)) if loops[i] not in self.idxs)
        if self.op is not None:
            op = getattr(xp, reduction_ops.get(self.op, None))
            val = op(arg, axis=axis)
        else:
            assert set(self.idxs) == set(loops)
            val = arg
        axis = [self.idxs.index(l) for l in loops if l in self.idxs]
        return np.permute_dims(val, axis)


l = Lark("""
    start: increment | assign
    increment: access BINARY "=" expr
    assign: access "=" expr
    expr: call | access
    access: TNS "[" (IDX ",")* IDX? "]"
    call: call_prefix | call_unary | call_binary
    call_prefix: (WORD "(" (IDX ",")* IDX?  ")") 
    call_unary: UNARY "(" expr ")"
    call_binary: expr BINARY expr
    UNARY: "+" | "-" | "not" | "~"
    BINARY: "+" | "-" | "*" | "or" | "and" | "|" | "&" | "^" | "<<" | ">>" | "//" | "/" | "%" | "**" | ">" | "<" | ">=" | "<=" | "==" | "!=" | "max" | "min"
    IDX: WORD
    TNS: WORD
    %import common.WORD 
    %ignore " "           // Disregard spaces in text
""")

def _parse_einsum(t) -> EinsumExpr:
    if t.data == "start":
        return _parse_einsum(t.children[0])
    elif t.data == "expr":
        return _parse_einsum(t.children[0])
    elif t.data == "access":
        tns = t.children[0].value
        idxs = [c.value for c in t.children[1:]]
        return Access(tns, idxs)
    elif t.data == "call":
        if t.children[0].data == "call_prefix":
            func = t.children[0].children[0].value
            args = [_parse_einsum(c) for c in t.children[0].children[1:]]
            return Call(func, args)
        elif t.children[0].data == "call_unary":
            func = t.children[0].children[0].value
            arg = _parse_einsum(t.children[0].children[1])
            return Call(func, [arg])
        elif t.children[0].data == "call_binary":
            left = _parse_einsum(t.children[0].children[0])
            func = t.children[0].children[1].value
            right = _parse_einsum(t.children[0].children[2])
            return Call(func, [left, right])
    elif t.data == "increment":
        access_node = t.children[0]  # This is the access node on the left side
        op = t.children[1].value     # This is the binary operator
        expr_node = t.children[2]    # This is the expr node on the right side
        
        # Extract tensor name and indices from access node
        tns = access_node.children[0].value
        idxs = [c.value for c in access_node.children[1:]]
        
        # Parse the right side expression
        input_expr = _parse_einsum(expr_node)
        
        return Einsum(input_expr, op, tns, idxs)
    elif t.data == "assign":
        access_node = t.children[0]  # This is the access node on the left side
        expr_node = t.children[1]    # This is the expr node on the right side
        
        # Extract tensor name and indices from access node
        tns = access_node.children[0].value
        idxs = [c.value for c in access_node.children[1:]]
        
        # Parse the right side expression
        input_expr = _parse_einsum(expr_node)
        
        return Einsum(input_expr, None, tns, idxs)
    else:
        raise ValueError(f"Unknown tree data: {t.data}")

def parse_einsum(expr: str) -> Einsum:
    tree = l.parse(expr)
    print(f"Parsed tree: {tree.pretty()}")
    return _parse_einsum(tree)

def run_einsum(xp, prgm, **kwargs):
    prgm = parse_einsum(prgm)
    return prgm.run(xp, kwargs)

A = np.random.rand(5,5)
B = np.random.rand(5,5)

C = run_einsum(np, "C[i,j] = A[i,j] + B[j,i]", A=A, B=B)

C_ref = A + B.T

assert np.allclose(C, C_ref)
print("✅ Test 1 passed: Basic addition with transpose")

# Test 2: Matrix multiplication using += (increment/accumulation)
print("Running Test 2: Matrix multiplication...")
A = np.random.rand(3, 4)
B = np.random.rand(4, 5)
C = run_einsum(np, "C[i,j] += A[i,k] * B[k,j]", A=A, B=B)
C_ref = A @ B
assert np.allclose(C, C_ref)
print("✅ Test 2 passed: Matrix multiplication")

# Test 3: Element-wise multiplication
print("Running Test 3: Element-wise multiplication...")
A = np.random.rand(4, 4)
B = np.random.rand(4, 4)
C = run_einsum(np, "C[i,j] = A[i,j] * B[i,j]", A=A, B=B)
C_ref = A * B
assert np.allclose(C, C_ref)
print("✅ Test 3 passed: Element-wise multiplication")

# Test 4: Sum reduction using +=
print("Running Test 4: Sum reduction...")
A = np.random.rand(3, 4)
C = run_einsum(np, "C[i] += A[i,j]", A=A)
C_ref = np.sum(A, axis=1)
assert np.allclose(C, C_ref)
print("✅ Test 4 passed: Sum reduction")

# Test 5: Maximum reduction using max=
print("Running Test 5: Maximum reduction...")
A = np.random.rand(3, 4)
C = run_einsum(np, "C[i] max= A[i,j]", A=A)
C_ref = np.max(A, axis=1)
assert np.allclose(C, C_ref)
print("✅ Test 5 passed: Maximum reduction")

# Test 6: Outer product
print("Running Test 6: Outer product...")
A = np.random.rand(3)
B = np.random.rand(4)
C = run_einsum(np, "C[i,j] = A[i] * B[j]", A=A, B=B)
C_ref = np.outer(A, B)
assert np.allclose(C, C_ref)
print("✅ Test 6 passed: Outer product")

# Test 7: Batch matrix multiplication using +=
print("Running Test 7: Batch matrix multiplication...")
A = np.random.rand(2, 3, 4)
B = np.random.rand(2, 4, 5)
C = run_einsum(np, "C[b,i,j] += A[b,i,k] * B[b,k,j]", A=A, B=B)
C_ref = np.matmul(A, B)
assert np.allclose(C, C_ref)
print("✅ Test 7 passed: Batch matrix multiplication")

# Test 8: Minimum reduction using min=
print("Running Test 8: Minimum reduction...")
A = np.random.rand(3, 4)
C = run_einsum(np, "C[i] min= A[i,j]", A=A)
C_ref = np.min(A, axis=1)
assert np.allclose(C, C_ref)
print("✅ Test 8 passed: Minimum reduction")

print("\n🎉 All tests passed! The einsum implementation is working correctly.")