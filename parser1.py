from typing import List, Tuple, Union

class SyntaxNode:
    def __init__(self, name, subs = None):
        self.name = name
        self.subs = subs if subs is not None else []

    def _format(self, indent=0):
        output = f"{'  ' * indent}{self.name}\n"
        for node in self.subs:
            output += node._format(indent + 1)
        return output
    
    def __str__(self):
        return self._format()

    def __repr__(self):
        return self.__str__()
    
class ErrorReport: 
    def __init__(self, position : int, message: str):
        self.position = position
        self.message = message

cfg = {
    -1: "S' → Program",
    0: "Program → DeclList",
    1: "DeclList → Decl DeclList",
    2: "DeclList → ε",
    3: "Decl → VarDecl",
    4: "Decl → FuncDecl",
    5: "VarDecl → type id ;",
    6: "VarDecl → type id = Expr ;",
    7: "FuncDecl → type id ( ParamList ) Block",
    8: "ParamList → Param , ParamList",
    9: "ParamList → Param",
    10: "ParamList → ε",
    11: "Param → type id",
    12: "Block → { StmtList }",
    13: "StmtList → Stmt StmtList",
    14: "StmtList → ε",
    15: "Stmt → IfStmt",
    16: "Stmt → LoopStmt",
    17: "Stmt → ReturnStmt",
    18: "Stmt → ExprStmt",
    19: "Stmt → VarDecl",
    20: "Stmt → Block",
    21: "IfStmt → if ( Expr ) Stmt ElsePart",
    22: "ElsePart → else Stmt",
    23: "ElsePart → ε",
    24: "LoopStmt → while ( Expr ) Stmt",
    25: "LoopStmt → for ( Expr ; Expr ; Expr ) Stmt",
    26: "ReturnStmt → return Expr ;",
    27: "ExprStmt → Expr ;",
    28: "Expr → EqualityExpr",
    29: "EqualityExpr → EqualityExpr == AddExpr",
    30: "EqualityExpr → AddExpr",
    31: "AddExpr → AddExpr + MulExpr",
    32: "AddExpr → MulExpr",
    33: "MulExpr → MulExpr * UnaryExpr",
    34: "MulExpr → UnaryExpr",
    35: "UnaryExpr → - UnaryExpr",
    36: "UnaryExpr → id ( ArgList )",
    37: "UnaryExpr → id",
    38: "UnaryExpr → num",
    39: "UnaryExpr → ( Expr )",
    40: "ArgList → Expr , ArgList",
    41: "ArgList → Expr",
    42: "ArgList → ε"
}

