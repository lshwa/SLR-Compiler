from typing import List, Tuple, Union

class ParseTree:
    def __init__(self, symbol, children=None):
        self.symbol = symbol
        self.children = children or []

    def __repr__(self, level=0):
        ret = "  " * level + self.symbol + "\n"
        for child in self.children:
            ret += child.__repr__(level + 1)
        return ret

class ErrorReport:
    def __init__(self, position: int, message: str):
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

# --- CFG 전처리 ---
productions = {}
nonterminals = set()
terminals = set()

for rule in cfg.values():
    lhs, rhs = rule.split("→")
    lhs = lhs.strip()
    rhs = rhs.strip()
    rhs_symbols = rhs.split() if rhs != 'ε' else []
    if lhs not in productions:
        productions[lhs] = []
    productions[lhs].append(rhs_symbols)
    nonterminals.add(lhs)

for rhs_lists in productions.values():
    for rhs in rhs_lists:
        for symbol in rhs:
            if symbol not in productions:
                terminals.add(symbol)

# --- FIRST / FOLLOW ---
FIRST = {}
FOLLOW = {}

for nt in nonterminals:
    FIRST[nt] = set()
for t in terminals:
    FIRST[t] = {t}
FIRST['ε'] = {'ε'}

def compute_FIRST():
    changed = True
    while changed:
        changed = False
        for lhs in productions:
            for rhs in productions[lhs]:
                if not rhs:
                    if 'ε' not in FIRST[lhs]:
                        FIRST[lhs].add('ε')
                        changed = True
                    continue
                i = 0
                while i < len(rhs):
                    sym = rhs[i]
                    if sym not in FIRST:
                        FIRST[sym] = {sym}  # ← 핵심 수정: KeyError 방지
                    before = len(FIRST[lhs])
                    FIRST[lhs].update(FIRST[sym] - {'ε'})
                    after = len(FIRST[lhs])
                    if after > before:
                        changed = True
                    if 'ε' not in FIRST[sym]:
                        break
                    i += 1
                else:
                    if 'ε' not in FIRST[lhs]:
                        FIRST[lhs].add('ε')
                        changed = True

for nt in nonterminals:
    FOLLOW[nt] = set()
FOLLOW['Program'].add('$')

def compute_FOLLOW():
    changed = True
    while changed:
        changed = False
        for lhs in productions:
            for rhs in productions[lhs]:
                trailer = FOLLOW[lhs].copy()
                for symbol in reversed(rhs):
                    if symbol in nonterminals:
                        before = len(FOLLOW[symbol])
                        FOLLOW[symbol].update(trailer)
                        after = len(FOLLOW[symbol])
                        if after > before:
                            changed = True
                        if 'ε' in FIRST[symbol]:
                            trailer.update(FIRST[symbol] - {'ε'})
                        else:
                            trailer = FIRST[symbol].copy()
                    else:
                        trailer = FIRST[symbol].copy()

# --- LR(0) Item 클래스 ---
class Item:
    def __init__(self, lhs: str, rhs: List[str], dot: int):
        self.lhs = lhs
        self.rhs = rhs
        self.dot = dot

    def is_complete(self):
        return self.dot >= len(self.rhs)

    def next_symbol(self):
        return self.rhs[self.dot] if not self.is_complete() else None

    def __eq__(self, other):
        return (self.lhs == other.lhs and self.rhs == other.rhs and self.dot == other.dot)

    def __hash__(self):
        return hash((self.lhs, tuple(self.rhs), self.dot))

def closure(items: List[Item]) -> List[Item]:
    closure_set = set(items)
    changed = True
    while changed:
        changed = False
        new_items = set()
        for item in closure_set:
            symbol = item.next_symbol()
            if symbol in productions:
                for prod in productions[symbol]:
                    new_item = Item(symbol, prod, 0)
                    if new_item not in closure_set:
                        new_items.add(new_item)
                        changed = True
        closure_set.update(new_items)
    return list(closure_set)

def goto(items: List[Item], symbol: str) -> List[Item]:
    moved_items = []
    for item in items:
        if item.next_symbol() == symbol:
            moved_items.append(Item(item.lhs, item.rhs, item.dot + 1))
    return closure(moved_items)

states = []
transitions = {}

def items_equal(a: List[Item], b: List[Item]) -> bool:
    return set(a) == set(b)

def add_state(items: List[Item]):
    for i, s in enumerate(states):
        if items_equal(s, items):
            return i
    states.append(items)
    return len(states) - 1

def build_dfa():
    start_items = closure([Item("S'", ['Program'], 0)])
    add_state(start_items)
    queue = [0]
    while queue:
        current = queue.pop(0)
        symbols = set()
        for item in states[current]:
            sym = item.next_symbol()
            if sym:
                symbols.add(sym)
        for sym in symbols:
            next_items = goto(states[current], sym)
            next_state = add_state(next_items)
            if (current, sym) not in transitions:
                transitions[(current, sym)] = next_state
                if next_state not in queue:
                    queue.append(next_state)

parsing_table = {}

def find_production_number(lhs: str, rhs: List[str]) -> int:
    for num, rule in cfg.items():
        rule_lhs, rule_rhs = rule.split("→")
        rule_lhs = rule_lhs.strip()
        rule_rhs = rule_rhs.strip()
        rhs_symbols = rule_rhs.split() if rule_rhs != 'ε' else []
        if rule_lhs == lhs and rhs_symbols == rhs:
            return num
    return -1

def build_parsing_table():
    for state_id, items in enumerate(states):
        parsing_table[state_id] = {}

        for item in items:
            if item.is_complete():
                if item.lhs == "S'" and item.rhs == ['Program']:
                    parsing_table[state_id]['$'] = 'acc'
                else:
                    prod_num = find_production_number(item.lhs, item.rhs)
                    for terminal in FOLLOW[item.lhs]:
                        parsing_table[state_id][terminal] = f"r{prod_num}"
            else:
                next_sym = item.next_symbol()
                next_state = transitions.get((state_id, next_sym))
                if next_state is not None:
                    if next_sym in nonterminals:
                        parsing_table[state_id][next_sym] = str(next_state)
                    else:
                        parsing_table[state_id][next_sym] = f"s{next_state}"

    # reduce 이후 GOTO가 누락되지 않도록 반드시 모든 nonterminal에 대한 GOTO를 등록
    for state_id in range(len(states)):
        for nonterm in nonterminals:
            goto_state = transitions.get((state_id, nonterm))
            if goto_state is not None:
                parsing_table[state_id][nonterm] = str(goto_state)

# 테이블 생성 실행
compute_FIRST()
compute_FOLLOW()
build_dfa()
build_parsing_table()

# --- parser() 함수는 건드리지 않음 ---
def parser(tokens: List[str]) -> Tuple[bool, Union[ParseTree, ErrorReport]]:
    print(">> Token List:", tokens)
    tokens.append('$')
    input_pointer = 0
    state_stack = [0]
    symbol_stack = []
    tree_stack = []

    while True:
        current_state = state_stack[-1]
        current_token = tokens[input_pointer]

        action = parsing_table.get(current_state, {}).get(current_token)

        if not action:
            return False, ErrorReport(input_pointer, f"unexpected token '{current_token}'")

        if action == 'acc':
            return True, tree_stack[-1]

        if action.startswith('s'):
            next_state = int(action[1:])
            state_stack.append(next_state)
            symbol_stack.append(current_token)
            tree_stack.append(ParseTree(current_token))
            input_pointer += 1

        elif action.startswith('r'):
            prod_num = int(action[1:])
            rule = cfg[prod_num]
            lhs, rhs = rule.split('→')
            lhs = lhs.strip()
            rhs = rhs.strip()
            rhs_symbols = rhs.split() if rhs != 'ε' else []

            if rhs == 'ε':
                children = []
            else:
                children = [tree_stack.pop() for _ in rhs_symbols][::-1]
                for _ in rhs_symbols:
                    state_stack.pop()
                    symbol_stack.pop()

            symbol_stack.append(lhs)
            tree_stack.append(ParseTree(lhs, children))

            goto_state = parsing_table[state_stack[-1]].get(lhs)
            if goto_state is None:
                if rhs == 'ε':  # ε-production: 그대로 현재 상태 사용
                    goto_state = str(state_stack[-1])
                else:
                    return False, ErrorReport(input_pointer, f"missing GOTO for '{lhs}'")
            state_stack.append(int(goto_state))
            
            if not action:
                expected = list(parsing_table.get(current_state, {}).keys())
                print(">> Debug Info:")
                print(f"  Current State: {current_state}")
                print(f"  Current Token: '{current_token}'")
                print(f"  State Stack: {state_stack}")
                print(f"  Symbol Stack: {symbol_stack}")
                print(f"  Expected: {expected}")
                return False, ErrorReport(input_pointer, f"unexpected token '{current_token}'")
            

print("\n======= States =======")
for i, state in enumerate(states):
    print(f"\nState {i}:")
    for item in state:
        print(f"  {item}")

print("\n======= Parsing Table (Partial) =======")
for state, actions in parsing_table.items():
    print(f"\nState {state}:")
    for symbol, act in actions.items():
        print(f"  {symbol}: {act}")