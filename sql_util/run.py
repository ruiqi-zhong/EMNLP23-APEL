import os
from typing import Any, Tuple
import sqlite3
from sql_util.dbpath import EXEC_TMP_DIR
import time
import subprocess
import pickle as pkl
import random
import threading
import re
from typing import Union, List, Tuple, Set, Dict, Any
from collections import defaultdict
from itertools import product
from collections import Counter
import numpy as np

threadLock = threading.Lock()


TIMEOUT = 60
# for evaluation's sake, replace current year always with 2020
CUR_YEAR = 2020


import signal
from contextlib import contextmanager

def make_display(exec_result):
    return tuple(tuple(round_element(element) for element in row) for row in exec_result)


class TimeoutException(Exception): pass
@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def permute_tuple(element: Tuple, perm: Tuple) -> Tuple:
    assert len(element) == len(perm)
    return tuple([element[i] for i in perm])



def get_cursor_path(sqlite_path: str):
    try:
        if not os.path.exists(sqlite_path):
            print('Openning a new connection %s' % sqlite_path)
        connection = sqlite3.connect(sqlite_path)
        connection.execute("PRAGMA foreign_keys = ON;")
        connection.execute('PRAGMA writable_schema=1; ')
    except Exception as e:
        print(sqlite_path)
        raise e
    connection.text_factory = lambda b: b.decode(errors='ignore')
    cursor = connection.cursor()
    return cursor


def can_execute_path(sqlite_path: str, q: str) -> bool:
    try:
        flag, result = exec_db_path_(sqlite_path, q)
        return flag == 'result'
    except:
        return False


def clean_tmp_f(f_prefix: str):
    with threadLock:
        for suffix in ('.in', '.out'):
            f_path = f_prefix + suffix
            if os.path.exists(f_path):
                os.unlink(f_path)

def exec_db_path(sqlite_path: str, query: str, timeout: int = TIMEOUT, nocase=True) -> Tuple[str, Any]:
    query = replace_cur_year(query)
    if nocase:
        query = add_collate_nocase(query)
    cursor = get_cursor_path(sqlite_path)
    try:
        with time_limit(timeout):
            cursor.execute(query)
            result = cursor.fetchall()
            cursor.close()
            cursor.connection.close()
            return 'result', result
    except Exception as e:
        cursor.close()
        cursor.connection.close()
        return 'exception', e

def replace_cur_year(query: str) -> str:
    return re.sub('YEAR\s*\(\s*CURDATE\s*\(\s*\)\s*\)\s*', '2020', query, flags=re.IGNORECASE)


def add_collate_nocase(query: str):
    value_regexps = ['"[^"]*"', "'[^']*'"]
    value_strs = []
    for regex in value_regexps:
        value_strs += re.findall(regex, query)
    for str_ in set(value_strs):
        query = query.replace(str_, str_ + " COLLATE NOCASE ")
    return query


def exec_db_path_(sqlite_path: str, query: str, nocase=True) -> Tuple[str, Any]:
    query = replace_cur_year(query)
    if nocase:
        query = add_collate_nocase(query)
    cursor = get_cursor_path(sqlite_path)
    try:
        with time_limit(TIMEOUT):
            cursor.execute(query)
            result = cursor.fetchall()
            cursor.close()
            cursor.connection.close()
            return 'result', result
    except Exception as e:
        cursor.close()
        cursor.connection.close()
        return 'exception', e


# return whether two bag of relations are equivalent
def multiset_eq(l1: List, l2: List) -> bool:
    if len(l1) != len(l2):
        return False
    d = defaultdict(int)
    for e in l1:
        d[e] = d[e] + 1
    for e in l2:
        d[e] = d[e] - 1
        if d[e] < 0:
            return False
    return True



# unorder each row in the table
# [result_1 and result_2 has the same bag of unordered row]
# is a necessary condition of
# [result_1 and result_2 are equivalent in denotation]
def quick_rej(result1: List[Tuple], result2: List[Tuple], order_matters: bool) -> bool:
    s1 = [unorder_row(row) for row in result1]
    s2 = [unorder_row(row) for row in result2]
    if order_matters:
        return s1 == s2
    else:
        return set(s1) == set(s2)


def unorder_row(row: Tuple) -> Tuple:
    return tuple(sorted(row, key=lambda x: str(x) + str(type(x))))

def get_constraint_permutation(tab1_sets_by_columns: List[Set], result2: List[Tuple]):
    num_cols = len(result2[0])
    perm_constraints = [{i for i in range(num_cols)} for _ in range(num_cols)]
    if num_cols <= 3:
        return product(*perm_constraints)

    # we sample 20 rows and constrain the space of permutations
    for _ in range(20):
        random_tab2_row = random.choice(result2)

        for tab1_col in range(num_cols):
            for tab2_col in set(perm_constraints[tab1_col]):
                if random_tab2_row[tab2_col] not in tab1_sets_by_columns[tab1_col]:
                    perm_constraints[tab1_col].remove(tab2_col)
    prod = 1
    for vs in perm_constraints:
        prod *= len(vs)
    return product(*perm_constraints)


# check whether two denotations are correct
def result_eq(result1: List[Tuple], result2: List[Tuple], order_matters: bool) -> bool:
    if len(result1) == 0 and len(result2) == 0:
        return True

    # if length is not the same, then they are definitely different bag of rows
    if len(result1) != len(result2):
        return False

    num_cols = len(result1[0])

    # if the results do not have the same number of columns, they are different
    if len(result2[0]) != num_cols:
        return False

    # unorder each row and compare whether the denotation is the same
    # this can already find most pair of denotations that are different
    if not quick_rej(result1, result2, order_matters):
        return False

    # the rest of the problem is in fact more complicated than one might think
    # we want to find a permutation of column order and a permutation of row order,
    # s.t. result_1 is the same as result_2
    # we return true if we can find such column & row permutations
    # and false if we cannot
    tab1_sets_by_columns = [{row[i] for row in result1} for i in range(num_cols)]

    # on a high level, we enumerate all possible column permutations that might make result_1 == result_2
    # we decrease the size of the column permutation space by the function get_constraint_permutation
    # if one of the permutation make result_1, result_2 equivalent, then they are equivalent
    for perm in get_constraint_permutation(tab1_sets_by_columns, result2):
        #if len(perm) != len(set(perm)):
        #    continue
        if num_cols == 1:
            result2_perm = result2
        else:
            result2_perm = [permute_tuple(element, perm) for element in result2]
        if order_matters:
            if result1 == result2_perm:
                return True
        else:
            # in fact the first condition must hold if the second condition holds
            # but the first is way more efficient implementation-wise
            # and we use it to quickly reject impossible candidates
            if set(result1) == set(result2_perm) and multiset_eq(result1, result2_perm):
                return True
    return False


def frozen_multiset(x):
    return frozenset(Counter(x).items())

def make_set(x):
    return tuple(sorted({tuple(sorted([str(a) for a in r])) for r in x}))


class SQLResult(object):
    def __init__(self, result, order_matters=False, no_rep=False):
        self.result = result
        self.order_matters = order_matters
        self.no_rep = no_rep
        assert not (order_matters and no_rep)
        if self.order_matters:
            # list denotation
            self.my_hash = hash(tuple(map(frozen_multiset, self.result)))
        elif not no_rep:
            # multiset denotation
            self.my_hash = hash(frozenset(map(frozen_multiset, self.result)))
        else:
            # standard set denotation
            self.result = make_set(self.result)
            self.my_hash = hash(self.result)

    def __eq__(self, other):
        assert self.order_matters == other.order_matters
        if not self.no_rep:
            return result_eq(self.result, other.result, self.order_matters)
        else:
            return other.result == self.result

    def __hash__(self):
        return self.my_hash

eps = 1e-1
# def round_element(e):
#     if e is None:
#         return str(e)
#     try:
#         x1 = float(e)
#         x2 = int(x1)
#         if np.abs(x1 - x2) < eps:
#             return str(int(x2))
#         else:
#             return str(e)
#     except KeyboardInterrupt as interrupt:
#         print(interrup)
#         exit(0)
#     except Exception as ex:
#         return str(e)


def round_element(e):
    if e is None:
        return str(e)
    try:
        x1 = float(e)
        start = int(x1) - 1
        for x2 in range(start, start+3):
            if np.abs(x1 - x2) < eps:
                return str(int(x2))
        return str('%.2f' % x1)
    except KeyboardInterrupt as interrupt:
        print(interrupt)
        exit(0)
    except Exception as ex:
        return str(e)
    
    
    
def setify_row(r):
    return tuple(sorted([round_element(x) for x in r]))


def listify_result(l):
    if type(l) != list:
        return random.random()
    return tuple([setify_row(row) for row in l])


def setify_result(l):
    if type(l) != list:
        return random.random()
    return tuple(sorted(set(setify_row(row) for row in l)))

    
def get_result_set(queries: List[str],
                  testcase_path: Union[str, None], 
                   distinguish_criteria='list') -> Dict[Any, List[int]]:

    results = defaultdict(list)
    for query_id, query in enumerate(queries):
        exec_flag, exec_result = exec_db_path_(testcase_path, query)
        if exec_flag == 'exception':
            results[-1].append(query_id)
#             print('get_result_set error', query, testcase_path)
#             exit(0)
            return None
        else:
            if distinguish_criteria == 'list':
                results[listify_result(exec_result)].append(query_id)
            elif distinguish_criteria == 'multiset':
                results[SQLResult(exec_result, order_matters=False)].append(query_id)
            elif distinguish_criteria == 'set':
                results[setify_result(exec_result)].append(query_id)
    return results
