import re
from collections import defaultdict


# ------------------------------------ Regex to NFA ------------------------------------------
class State:
    _id_counter = 0  # статическая переменная

    def __init__(self, name=None):
        if name is None:
            self.name = f"q{State._id_counter}"
            State._id_counter += 1
        else:
            self.name = name

        self.transitions = defaultdict(set)
        self.is_final = False

    def add_transition(self, symbol, state):
        self.transitions[symbol].add(state)

    def __hash__(self):
        return hash(self.name)  # Используем имя для хеширования

    def __eq__(self, other):
        return self.name == other.name  # Сравниваем состояния по имени

    def __repr__(self):
        return f"State({self.name}, final={self.is_final})"


# Класс для НКА
class NFA:
    def __init__(self, start_state, final_state):
        self.start_state = start_state
        self.final_state = final_state
        self.final_state.is_final = True

    def add_transition(self, from_state, symbol, to_state):
        from_state.add_transition(symbol, to_state)


# Раскрытие диапазонов в квадратных скобках
def expand_square_brackets(regex):
    def expand_char_class(char_class):
        result = []
        i = 0
        while i < len(char_class):
            if i + 2 < len(char_class) and char_class[i + 1] == '-':
                start = char_class[i]
                end = char_class[i + 2]
                result.extend([chr(c) for c in range(ord(start), ord(end) + 1)])
                i += 3
            else:
                result.append(char_class[i])
                i += 1
        return result

    pattern = re.compile(r'\[([^\]]+)\]')
    while match := pattern.search(regex):
        chars = match.group(1)
        expanded_chars = expand_char_class(chars)
        expanded = '(' + '|'.join(expanded_chars) + ')'
        regex = regex[:match.start()] + expanded + regex[match.end():]
    return regex


# Раскрытие {n}, {n,m}
def expand_braces(regex):
    pattern = re.compile(r'(\([^)]+\)|\([^()]+\)|\[[^\]]+\]|[^(){}\[\]])\{(\d+)(?:,(\d+))?\}')
    while True:
        match = pattern.search(regex)
        if not match:
            break
        base = match.group(1)
        n = int(match.group(2))
        m = int(match.group(3)) if match.group(3) else None

        if base.startswith('(') and base.endswith(')'):
            base = base
        else:
            base = f'({base})'

        if m is None:
            replacement = base * n
        else:
            options = [''.join([base] * i) for i in range(n, m + 1)]
            replacement = '(' + '|'.join(options) + ')'

        regex = regex[:match.start()] + replacement + regex[match.end():]

    return regex


# Вставка явной конкатенации (.) между элементами
def insert_concatenation(regex):
    result = ""
    for i in range(len(regex) - 1):
        result += regex[i]
        if (regex[i].isalnum() or regex[i] in ')*+?}') and (
                regex[i + 1].isalnum() or regex[i + 1] == '(' or regex[i + 1] == '['):
            result += '.'
    result += regex[-1]
    return result


# Алгоритм сортировочной станции для преобразования в постфиксную запись
def shunting_yard(regex):
    precedence = {'*': 3, '+': 3, '?': 3, '.': 2, '|': 1}
    output = []
    operators = []
    for char in regex:
        if char.isalnum():
            output.append(char)
        elif char in precedence:
            while operators and operators[-1] != '(' and precedence[operators[-1]] >= precedence[char]:
                output.append(operators.pop())
            operators.append(char)
        elif char == '(':
            operators.append(char)
        elif char == ')':
            while operators and operators[-1] != '(':
                output.append(operators.pop())
            operators.pop()
    while operators:
        output.append(operators.pop())
    return ''.join(output)


# Построение НКА из постфиксного выражения
def regex_to_nfa(regex):
    # print(f"\n[DEBUG] Исходное выражение: {regex}")
    regex = expand_square_brackets(regex)
    # print(f"[DEBUG] После expand_square_brackets: {regex}")
    regex = expand_braces(regex)
    # print(f"[DEBUG] После expand_braces: {regex}")
    regex = insert_concatenation(regex)
    # print(f"[DEBUG] После insert_concatenation: {regex}")
    regex = shunting_yard(regex)
    # print(f"[DEBUG] После shunting_yard (постфикс): {regex}")

    stack = []
    for char in regex:
        if char.isalnum():
            start = State()
            end = State()
            start.add_transition(char, end)
            stack.append(NFA(start, end))
        elif char == '*':
            nfa = stack.pop()
            start = State()
            end = State()
            start.add_transition('', nfa.start_state)
            start.add_transition('', end)
            nfa.final_state.add_transition('', nfa.start_state)
            nfa.final_state.add_transition('', end)
            stack.append(NFA(start, end))
        elif char == '+':
            nfa = stack.pop()
            start = State()
            end = State()
            start.add_transition('', nfa.start_state)
            nfa.final_state.add_transition('', nfa.start_state)
            nfa.final_state.add_transition('', end)
            stack.append(NFA(start, end))
        elif char == '?':
            nfa = stack.pop()
            start = State()
            end = State()
            start.add_transition('', nfa.start_state)
            start.add_transition('', end)
            nfa.final_state.add_transition('', end)
            stack.append(NFA(start, end))
        elif char == '|':
            nfa2 = stack.pop()
            nfa1 = stack.pop()
            start = State()
            end = State()
            start.add_transition('', nfa1.start_state)
            start.add_transition('', nfa2.start_state)
            nfa1.final_state.add_transition('', end)
            nfa2.final_state.add_transition('', end)
            stack.append(NFA(start, end))
        elif char == '.':
            nfa2 = stack.pop()
            nfa1 = stack.pop()
            nfa1.final_state.add_transition('', nfa2.start_state)
            stack.append(NFA(nfa1.start_state, nfa2.final_state))

    return stack.pop()


# Удалить все флаги is_final кроме настоящего финального состояния
def clear_is_final(nfa):
    visited = set()
    stack = [nfa.start_state]
    while stack:
        state = stack.pop()
        if state in visited:
            continue
        visited.add(state)
        state.is_final = False
        for transitions in state.transitions.values():
            stack.extend(transitions)
    nfa.final_state.is_final = True


def epsilon_closure(states):
    stack = list(states)
    closure = set(states)
    while stack:
        state = stack.pop()
        epsilon_targets = state.transitions.get('', set())

        # Обработка только если это действительно множество (для NFA)
        if isinstance(epsilon_targets, set):
            for next_state in epsilon_targets:
                if next_state not in closure:
                    closure.add(next_state)
                    stack.append(next_state)
    return closure


# Проверка строки на соответствие НКА
def match_nfa(nfa, string):
    current_states = epsilon_closure({nfa.start_state})
    for char in string:
        next_states = set()
        for state in current_states:
            for next_state in state.transitions.get(char, set()):
                next_states.update(epsilon_closure({next_state}))
        current_states = next_states
    return any(state.is_final for state in current_states)


# -------------------------------------# NFA to DFA---------------------------------------------------------------------


class DFAState:
    def __init__(self, arg, is_final=False):
        if isinstance(arg, (set, frozenset)):
            # Создание из множества состояний НКА
            self.nfa_states = frozenset(arg)
            self.name = "_".join(sorted(str(id(s)) for s in arg))  # уникальное имя по id состояний
            self.is_final = any(s.is_final for s in arg)
        else:
            # Создание по имени
            self.name = arg
            self.nfa_states = None
            self.is_final = is_final

        self.transitions = {}

    def add_transition(self, symbol, state):
        self.transitions[symbol] = state

    def __repr__(self):
        return f"DFAState({self.name}, final={self.is_final})"


class DFA:
    def __init__(self, start_state):
        self.start_state = start_state
        self.states = []


def get_alphabet(automaton):
    alphabet = set()
    visited = set()
    stack = [automaton.start_state]

    while stack:
        state = stack.pop()
        if state in visited:
            continue
        visited.add(state)

        for symbol, next_states in state.transitions.items():
            if symbol != '':  # исключаем ε-переходы
                alphabet.add(symbol)

            # Поддержка и NFA, и DFA
            if isinstance(next_states, set):  # NFA: множество состояний
                for next_state in next_states:
                    stack.append(next_state)
            else:  # DFA: одно состояние
                stack.append(next_states)

    return alphabet


def nfa_to_dfa(nfa):
    from collections import deque

    def get_start_closure(start_state):
        # Безопасная проверка на поддержку ε-переходов
        if hasattr(start_state, 'transitions') and isinstance(start_state.transitions.get('', set()), set):
            return epsilon_closure({start_state})
        else:
            return {start_state}

    start_set = get_start_closure(nfa.start_state)
    start_state = DFAState(start_set)
    dfa_states = {frozenset(start_set): start_state}
    queue = deque([start_set])
    alphabet = get_alphabet(nfa)

    while queue:
        current_set = queue.popleft()
        current_dfa_state = dfa_states[frozenset(current_set)]

        for symbol in alphabet:
            if symbol == '':
                continue  # ε не обрабатываем в ДКА

            next_set = set()
            for state in current_set:
                targets = state.transitions.get(symbol, set())
                next_set.update(targets)

            closure = epsilon_closure(next_set)
            closure_frozen = frozenset(closure)

            if closure_frozen not in dfa_states:
                dfa_states[closure_frozen] = DFAState(closure)
                queue.append(closure)

            current_dfa_state.transitions[symbol] = dfa_states[closure_frozen]

    dfa = type('DFA', (), {})()
    dfa.start_state = start_state
    dfa.states = set(dfa_states.values())
    return dfa


def match_dfa(dfa, string):
    current_state = dfa.start_state
    for char in string:
        if char not in current_state.transitions:
            return False
        current_state = current_state.transitions[char]
    return current_state.is_final


# -----------------------------------Minimisation-----------------------------------------------------------------------


def reverse_dfa(dfa):
    all_states = set()
    reversed_transitions = defaultdict(set)
    final_states = []

    # Создаём все состояния и реверсируем переходы
    for state in dfa.states:
        all_states.add(state)
        for symbol, target in state.transitions.items():
            reversed_transitions[target].add((symbol, state))
        if state.is_final:
            final_states.append(state)

    # Создаём новое стартовое состояние, соединённое ε-переходами с финальными
    new_start = State("new_start")
    for fs in final_states:
        new_start.add_transition('', fs)  # ε-переходы

    # Старое начальное состояние становится финальным
    #  print(f"[DEBUG reverse_dfa] Старое стартовое состояние теперь финальное: {id(dfa.start_state)}")

    # Обновляем переходы
    for state in all_states:
        new_trans = defaultdict(set)
        for dest, srcs in reversed_transitions.items():
            if dest == state:
                for symbol, src in srcs:
                    new_trans[symbol].add(src)
        state.transitions = new_trans

    all_states.add(new_start)

    # Сброс финальности у всех
    for state in all_states:
        state.is_final = False
    dfa.start_state.is_final = True

    return NFA(new_start, dfa.start_state)


def brzozowski_minimization(dfa):
    # print("[DEBUG] ⬅ Первый реверс и детерминизация")
    reversed_nfa = reverse_dfa(dfa)  # ✔ пусть остаётся
    reversed_dfa = nfa_to_dfa(reversed_nfa)

    # print("[DEBUG] ➡ Второй реверс и детерминизация")
    reversed_nfa2 = reverse_dfa(reversed_dfa)
    minimized_dfa = nfa_to_dfa(reversed_nfa2)

    return minimized_dfa


# -------------------------------------------------------------------------


# 🧪 Примеры
regexes = {
    '[a-z]{3}': ["abc", "xyz", "ab", "abcd"],
    '(a|b)+c': ["aac", "abc", "bbbc", "c", "abb"],
    '[0-9]{3,4}': ["12", "123", "1234", "1", "12345"],
    '[A-Ca-c]*': ["", "AaBbCc", "abcAC", "D", "abcx"],
    '[13579]?0': ["0", "10", "30", "11", "00", "21"],
    'a*': ['', 'a', 'aa', 'aaa', 'b'],
    'b+': ['', 'b', 'bb', 'bbb', 'bbj'],
    'c?': ['', 'c', 'cc', 'ccc', 'c1'],
    '(a|b)*abb':['abb', 'ababb', 'ab']
}

for regex, strings in regexes.items():
    print("----------------------------------------------------------------------------------")
    print(f"\nРегулярное выражение: '{regex}'")

    nfa = regex_to_nfa(regex)
    clear_is_final(nfa)
    # print("NFA построен успешно")

    dfa = nfa_to_dfa(nfa)
    # print("DFA построен успешно")

    min_dfa = brzozowski_minimization(dfa)

    for s in strings:
        result = match_nfa(nfa, s)
        # print(f"Строка '{s}' соответствует автомату NFA: {result}")
        # #
        # result_dfa = match_dfa(dfa, s)
        # print(f"Строка '{s}' соответствует автомату DFA: {result_dfa}")

        result_min_dfa = match_dfa(min_dfa, s)
        print(f"Строка '{s}' соответствует MIN автомату DFA: {result_min_dfa}")
