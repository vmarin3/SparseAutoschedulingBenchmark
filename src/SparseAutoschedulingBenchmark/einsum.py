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
        if len(self.args) == 1
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
    start: INCREMENT | ASSIGN
    INCREMENT: ACCESS OP = EXPR
    ASSIGN: ACCESS = EXPR
    EXPR: CALL | ACCESS
    ACCESS: TNS "[" (IDX ",")* IDX? "]"
    CALL = CALL_PREFIX | CALL_UNARY | CALL_BINARY
    CALL_PREFIX: (FUNC "(" (IDX ",")* IDX?  ")") 
    CALL_UNARY: UNARY "(" EXPR ")"
    CALL_BINARY: EXPR BINARY EXPR
    FUNC: WORD | UNARY | BINARY
    UNARY: "+" | "-" | "not" | "~"
    BINARY: "+" | "-" | "*" | "or" | "and" | "|" | "&" | "^" | "<<" | ">>" | "//" | "/" | "%" | "**" | ">" | "<" | ">=" | "<=" | "==" | "!="
    IDX: WORD
    %import common.WORD 
    %ignore " "           // Disregard spaces in text
""")

def _parse_einsum(t) -> EinsumExpr:
    if t.data == "ACCESS":
        tns = t.children[0].value
        idxs = [c.value for c in t.children[1:]]
        return Access(tns, idxs)
    elif t.data == "CALL":
        if t.children[0].data == "CALL_PREFIX":
            func = t.children[0].children[0].value
            args = [_parse_einsum(c) for c in t.children[0].children[1:]]
            return Call(func, args)
        elif t.children[0].data == "CALL_UNARY":
            func = t.children[0].children[0].value
            arg = _parse_einsum(t.children[0].children[1])
            return Call(func, [arg])
        elif t.children[0].data == "CALL_BINARY":
            left = _parse_einsum(t.children[0].children[0])
            func = t.children[0].children[1].value
            right = _parse_einsum(t.children[0].children[2])
            return Call(func, [left, right])
    elif t.data == "INCREMENT":
        input_expr = _parse_einsum(t.children[0])
        op = t.children[1].value
        tns = t.children[2].value
        idxs = [c.value for c in t.children[3:]]
        return Einsum(input_expr, op, tns, idxs)
    elif t.data == "ASSIGN":
        input_expr = _parse_einsum(t.children[0])
        tns = t.children[2].value
        idxs = [c.value for c in t.children[3:]]
        return Einsum(input_expr, None, tns, idxs)
    else:
        raise ValueError(f"Unknown tree data: {t.data}")

def parse_einsum(expr: str) -> Einsum:
    return _parse_einsum(l.parse(expr))

def run_einsum(xp, prgm, **kwargs):
    prgm = parse_einsum(prgm)

A = np.random.rand(5,5)
B = np.random.rand(5,5)

C = run_einsum(np, "C[i,j] = A[i,j] + B[j,i]", A=A, B=B)

C_ref = A + B.T

assert np.allclose(C, C_ref)