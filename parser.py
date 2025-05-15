from typing import List, Tuple, Union

'''
각 노드 : 하나의 문법 기호 symbol을 나타내고, 하위 노드(children)은 기호로부터 파생된 자식 트리
'''
class ParseTree:
    def __init__(self, symbol, children=None):
        self.symbol = symbol
        self.children = children or []

    def __repr__(self, level=0):
        ret = "  " * level + self.symbol + "\n"
        for child in self.children:
            ret += child.__repr__(level + 1)
        return ret

'''
position(int) : error 발생 토큰의 index
message(str) : error message (for debuging)
'''
class ErrorReport:
    def __init__(self, position: int, message: str):
        self.position = position
        self.message = message

'''
all CFG 정리
처음에 S-> Program 추가로 CFG 문법의 시작점 추가 
'''
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
    19: "Stmt → VarDecl",   # 새로 추가된 CFG 문법
    20: "Stmt → Block",
    21: "IfStmt → if ( Expr ) Stmt ElsePart",
    22: "ElsePart → else Stmt",
    23: "ElsePart → ε",
    24: "LoopStmt → while ( Expr ) Stmt",
    25: "LoopStmt → for ( Expr ; Expr ; Expr ) Stmt",
    26: "ReturnStmt → return Expr ;",
    27: "ExprStmt → id = Expr ;",  # 수정된 CFG 문법 
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

# CFG 문법 parsing 
productions = {}    # non terminal 기호 -> 가능한 우변 리스트
nonterminals = set() # non terminal 집합
terminals = set() # terminal 집합 


for rule in cfg.values():
    lhs, rhs = rule.split("→") # → 를 통해 lhs, rhs 리스트 추가 
    lhs = lhs.strip()
    rhs = rhs.strip()
    rhs_symbols = rhs.split() if rhs != 'ε' else []
    if lhs not in productions:
        productions[lhs] = []
    productions[lhs].append(rhs_symbols)
    # lhs는 항상 nonterminal
    nonterminals.add(lhs)   

for rhs_lists in productions.values():
    for rhs in rhs_lists:
        for symbol in rhs:
            if symbol not in productions:
                terminals.add(symbol)

'''
FIRST : 각 긱호의 FIRST 집합 
FOLLOW : 각 기호의 FOLLOW 집합 
'''
FIRST = {}
FOLLOW = {}

for nt in nonterminals:
    FIRST[nt] = set()
for t in terminals:
    FIRST[t] = {t}
FIRST['ε'] = {'ε'}

"""
    FIRST SET Rule
    1. A → ε , FIRST(A) include ε
    2. A -> B1 B2 .... BN 
        (1). FIRST(B1) - ε to FIRST (A)
        (2). FIRST(B1) has ε, add FIRST(B2) and so on until BN

"""
def handle_epsilon_addition(lhs: str) -> bool:
    if 'ε' not in FIRST[lhs]:
        FIRST[lhs].add('ε')
        return True
    return False


def compute_FIRST():
    changed = True
    while changed:
        changed = False

        for lhs, rhs_list in productions.items():
            for rhs in rhs_list:
                if not rhs or rhs == ['ε']:
                    changed |= handle_epsilon_addition(lhs)
                    continue

                for i, symbol in enumerate(rhs):
                    if symbol not in FIRST:
                        FIRST[symbol] = {symbol}

                    before = len(FIRST[lhs])
                    FIRST[lhs].update(FIRST[symbol] - {'ε'})
                    after = len(FIRST[lhs])

                    if after > before:
                        changed = True

                    if 'ε' not in FIRST[symbol]:
                        break
                else:
                    changed |= handle_epsilon_addition(lhs)
    '''
    print("\n FIRST SET 출력 (디버깅용)\n")
    for symbol in sorted(FIRST.keys()):
        print(f"FIRST({symbol}) = {{','.join(sorted(FIRST[symbol]))}}
    '''

for nt in nonterminals:
    FOLLOW[nt] = set()
FOLLOW['Program'].add('$')


"""
    FOLLOW SET Rule

    1. 시작 기호의 FOLLOW에는 반드시 $ 추가

    2. A → α B β 형태의 production에서:
        (1) FIRST(β) - ε 를 FOLLOW(B)에 추가

        (2) β 가 없거나 FIRST(β)에 ε이 있으면,
            FOLLOW(A) 를 FOLLOW(B)에 추가
"""
def update_follow_set(lhs: str, trailer: set, symbol: str) -> bool:
    before = len(FOLLOW[symbol])
    FOLLOW[symbol].update(trailer)
    after = len(FOLLOW[symbol])
    return after > before

def compute_FOLLOW():
    changed = True

    while changed:
        changed = False

        for lhs, rhs_list in productions.items():
            for rhs in rhs_list:
                follow_trail = FOLLOW[lhs].copy()

                for sym in reversed(rhs):
                    if sym in nonterminals:
                        if update_follow_set(lhs, follow_trail, sym):
                            changed = True

                        if 'ε' in FIRST[sym]:
                            follow_trail.update(FIRST[sym] - {'ε'})
                        else:
                            follow_trail = FIRST[sym].copy()
                    else:
                        follow_trail = FIRST.get(sym, set()).copy()

'''
    print("\n FOLLOW SET 출력 디버깅용")
    for key in sorted(FOLLOW):
        print(f"FOLLOW({key}) = {{ {', '.join(sorted(FOLLOW[key]))} }}")
'''

"""
    ITEM 정의 : A → α • β 에서 dot이 현재 포인터의 위치 의미
"""
class Item:
    def __init__(self, lhs: str, rhs: List[str], dot: int):
        self.lhs = lhs      # 좌변 기호 (non-terminal)
        self.rhs = rhs      # 우변 기호들
        self.dot = dot      # 점(dot) 위치, rhs에서 현재까지 처리된 길이

    def is_complete(self) -> bool: # dot이 우변 끝에 도달하면 True 반환

        return self.dot >= len(self.rhs)

    def next_symbol(self): # dot 바로 뒤의 기호 반환, 끝일떄는 None 반환
  
        if self.is_complete():
            return None
        return self.rhs[self.dot]

    def __eq__(self, other):
        if not isinstance(other, Item):
            return False
        return (
            self.lhs == other.lhs and
            self.rhs == other.rhs and
            self.dot == other.dot
        )

    def __hash__(self):
        return hash((self.lhs, tuple(self.rhs), self.dot))

'''
for test : dot 위치 기준, production 문자열 출력 
    def __repr__(self):

        body = self.rhs[:]
        body.insert(self.dot, '•')
        return f"{self.lhs} → {' '.join(body)}"
'''

'''
주어진 LR(0) 항목 집합에 대해 closure를 계산해서 새로운 항목이 생길때까지 확장함.
'''
def closure(items: List[Item]) -> List[Item]:
    closure_set = set(items)
    changed = True

    while changed:
        changed = False
        additions = set()

        for item in closure_set:
            next_sym = item.next_symbol()

            if next_sym is None:
                continue

            # next_symbol이 non-terminal일 때 그 production 들도 추가
            if next_sym in productions:
                for rhs in productions[next_sym]:
                    candidate = Item(next_sym, rhs, 0)

                    if candidate not in closure_set and candidate not in additions:
                        additions.add(candidate)
                        changed = True

        if additions:
            for new_item in additions:
                closure_set.add(new_item)

    return list(closure_set)



def goto(items: List[Item], symbol: str) -> List[Item]: # 주어진 Item에서 symbol 소비 이후 도달 가능한 state set, 
    shifted = []

    for itm in items:
        next_sym = itm.next_symbol()

        if next_sym == symbol:
            advanced = Item(itm.lhs, itm.rhs, itm.dot + 1)
            shifted.append(advanced)

    if not shifted:
        return []

    result = closure(shifted)
    return result

states = []
transitions = {}

def items_equal(a: List[Item], b: List[Item]) -> bool:
    if len(a) != len(b):    # 두 item list가 같은 상태인지 확인
        return False
    return set(a) == set(b) # 같은 state 인데 새로 생기는 거 방지

def add_state(items: List[Item]) -> int:

    for idx, state in enumerate(states):  # 이미 존재하는 상태인지 확인
        if items_equal(state, items):
            return idx

    # 없어서 새로 추가
    states.append(items)
    return len(states) - 1


def build_dfa():
    # 초기 상태: S' → • Program
    start_items = closure([Item("S'", ['Program'], 0)])
    add_state(start_items)
    queue = [0]

    while queue:
        current = queue.pop(0)
        symbols = set()
        # 현재 상태에서 갈 수 있는 모든 symbol 수집

        for item in states[current]:
            sym = item.next_symbol()

            if sym:
                symbols.add(sym)

        # 각 symbol에 대해 이동 가능한 상태 계산
        for sym in symbols:
            next_items = goto(states[current], sym)
            next_state = add_state(next_items)

            # transitions 테이블에 등록함.
            if (current, sym) not in transitions:
                transitions[(current, sym)] = next_state

                # new state일 때, 큐에 추가 
                if next_state not in queue:
                    queue.append(next_state)

parsing_table = {}

'''
CFG 문법에서 production 해당번호 찾는 함수 
'''
def find_production_number(lhs: str, rhs: List[str]) -> int:

    for num, rule in cfg.items():
        if "→" not in rule:
            continue 

        # production rule을 좌변과 우변으로 나눔 
        rule_lhs, rule_rhs = rule.split("→")
        rule_lhs = rule_lhs.strip()
        rule_rhs = rule_rhs.strip()

        if rule_rhs == 'ε':
            rule_rhs_symbols = [] 
        else:
            rule_rhs_symbols = rule_rhs.split()

        # lhs와 rhs가 일치하는 production 찾아서 번호 반환
        if rule_lhs == lhs and rule_rhs_symbols == rhs:
            return num

    # 에러 (일치하는 production이 없다.)
    return -1

''' build_parsing_table()
DFA 상태들을 기반으로 parsing table 구성
상태별로 가능한 이동 또는 reduce/accept 동작 정의 
'''
def build_parsing_table():
    for state_id, items in enumerate(states):
        parsing_table[state_id] = {}

        for item in items:
            # item이 rhs 끝까지 도달 한 경우 
            if item.is_complete():
                is_accept = (item.lhs == "S'" and item.rhs == ['Program'])

                if is_accept: # accept 조건 
                    parsing_table[state_id]['$'] = 'acc'
                else: # reduce 조건 
                    prod_num = find_production_number(item.lhs, item.rhs)

                    for terminal in FOLLOW[item.lhs]:
                        parsing_table[state_id][terminal] = f"r{prod_num}"

            else:
                # Shift or GOTO (dot 뒤에 symbol이 존재할 때)
                next_sym = item.next_symbol()
                next_state = transitions.get((state_id, next_sym))

                if next_state is not None:
                    if next_sym in nonterminals: # GOTO 
                        parsing_table[state_id][next_sym] = str(next_state)
                    else:   # SHIFT
                        parsing_table[state_id][next_sym] = f"s{next_state}"

    # reduce 이후 GOTO 누락 에러 처리 
    for state_id in range(len(states)):
        for nonterm in nonterminals:
            goto_state = transitions.get((state_id, nonterm))
            if goto_state is not None:
                parsing_table[state_id][nonterm] = str(goto_state)
    
    ''' 
    print("\nParsing Table") # 디버깅용 
    for state in sorted(parsing_table.keys()):
        print(f"\nState {state}:")
        for symbol, action in parsing_table[state].items():
            print(f"  {symbol:8} → {action}")
    '''

'''
CFG -> FIRST / FOLLOW 계산 -> DFA state 생성 -> ACTION / GOTO 테이블 생성 
'''
compute_FIRST()
compute_FOLLOW()
build_dfa()
build_parsing_table()


"""
    토큰 리스트 LR parsing table 기반 파싱 
    성공 -> ParseeTree 반환, 실패 -> ErrorReport 반환
"""
def parser(tokens: List[str]) -> Tuple[bool, Union[ParseTree, ErrorReport]]:

    # 입력 종료 기호 
    tokens.append('$')  
    input_pointer = 0

    state_stack = [0]       # 상태 스택
    symbol_stack = []       # 기호 스택
    tree_stack = []         # 파싱 트리 구성용 스택

    while True:
        current_state = state_stack[-1]
        current_token = tokens[input_pointer]

        # 현재 상태에서 현재 token으로 action 조회 
        action = parsing_table.get(current_state, {}).get(current_token)

        if not action: # error : 유효하지 않은 토큰 입력 
            return False, ErrorReport(input_pointer, f"예상되지 않는 token입니다. '{current_token}'")

        # acc success case 
        if action == 'acc':
            return True, tree_stack[-1]

        if action.startswith('s'):  # Shift : next state로 이동
            next_state = int(action[1:])
            state_stack.append(next_state)
            symbol_stack.append(current_token)
            tree_stack.append(ParseTree(current_token))
            input_pointer += 1

        elif action.startswith('r'):    # Reduce : production에 따라 tree 병합
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

            # GOTO 계산
            goto_state = parsing_table[state_stack[-1]].get(lhs)
            if goto_state is None:
                
                # ε production인 경우 현재 상태 그대로 유지
                if rhs == 'ε':
                    goto_state = str(state_stack[-1])

                else:
                    return False, ErrorReport(input_pointer, f"GOTO state가 없습니다. '{lhs}'")

            state_stack.append(int(goto_state))

        # error : 아예 action이 실행이 안되는 경우 예외처리 
        if not action:
            expected = list(parsing_table.get(current_state, {}).keys())
            return False, ErrorReport(input_pointer, f"예상되지 않는 토큰입니다. '{current_token}' (expected: {expected})")
        
'''
print("States")
for i, state in enumerate(states):
    print(f"\nState {i}:")
    for item in state:
        print(f"{item}")

print("\n")

print("Parsing Table (디버깅용)"
for state, actions in parsing_table.items():
    print(f"\nState {state}:")
    for symbol, act in actions.items():
        print(f"{symbol}: {act}")
'''        