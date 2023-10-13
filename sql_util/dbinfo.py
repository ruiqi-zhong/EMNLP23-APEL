import pickle as pkl
from typing import Set, List, Tuple, Dict, Any, TypeVar
from collections import OrderedDict, defaultdict, Counter
from itertools import chain
from sql_util.run import exec_db_path_, get_cursor_path
from sql_util.dbpath import get_value_path
from collections import defaultdict
import random
import os
import re
from pprint import pprint
import subprocess
import sqlparse
import time


table_name_query = "SELECT name FROM sqlite_master WHERE type='table';"
column_type_query = "pragma table_info('%s');"
foreign_key_query = "pragma foreign_key_list('%s')"
table_schema_query = "select sql from sqlite_master where type='table' and name='%s'"
select_all_query = "SELECT * from %s;"
INCLUDE_ALL = 'include_all'
MAX_DISPLAY = 10


def get_values(db_name: str) -> Set[str]:
    values = pkl.load(open(get_value_path(db_name), 'rb'))
    return values


def get_tc2values(db_id):
    db_path = 'data/database/{db_id}/{db_id}.sqliteempty'.format(db_id=db_id)
    result = {}
    p, d, e = get_all_db_info_path(db_path)
    for t, c in p:
        val_path = 'synthetic_vals/{db_id}-{t}-{c}.pkl'.format(db_id=db_id, t=t, c=c)
        result[(t, c)] = pkl.load(open(val_path, 'rb'))
    return result


def get_schema_path(sqlite_path: str, table_name: str) -> str:
    _, schema = exec_db_path_(sqlite_path, table_schema_query % table_name, nocase=False)
    schema = schema[0][0]
    return schema


def get_unique_keys(schema: str) -> Set[str]:
    schema_by_list = schema.split('\n')
    unique_keys = set()
    for r in schema_by_list:
        if 'unique' in r.lower():
            unique_keys.add(r.strip().split()[0].upper().replace("\"", '').replace('`', ''))
    return unique_keys


def get_checked_keys(schema: str) -> Set[str]:
    schema_by_list = schema.split('\n')
    checked_keys = set()
    for r in schema_by_list:
        if 'check (' in r or 'check(' in r:
            checked_keys.add(r.strip().split()[0].upper().replace("\"", '').replace('`', ''))
    return checked_keys


def get_table_names_path(sqlite_path: str) -> List[str]:
    table_names = [x[0] for x in exec_db_path_(sqlite_path, table_name_query, nocase=False)[1]]
    return table_names


def extract_table_column_properties_path(sqlite_path: str) \
        -> Tuple[Dict[str, Dict[str, Any]], Dict[Tuple[str, str], Tuple[str, str]]]:
    table_names = get_table_names_path(sqlite_path)
    table_name2column_properties = OrderedDict()
    child2parent = OrderedDict()
    for table_name in table_names:
        schema = get_schema_path(sqlite_path, table_name)
        unique_keys, checked_keys = get_unique_keys(schema), get_checked_keys(schema)
        table_name = table_name.lower()
        column_properties = OrderedDict()
        result_type, result = exec_db_path_(sqlite_path, column_type_query % table_name, nocase=False)
        for (
                columnID, column_name, columnType,
                columnNotNull, columnDefault, columnPK,
        ) in result:
            column_name = column_name.upper()
            column_properties[column_name] = {
                'ID': columnID,
                'name': column_name,
                'type': columnType,
                'notnull': columnNotNull,
                'default': columnDefault,
                'PK': columnPK,
                'unique': column_name in unique_keys,
                'checked': column_name in checked_keys
            }
        table_name2column_properties[table_name.lower()] = column_properties

        # extract foreign keys and population child2parent
        result_type, result = exec_db_path_(sqlite_path, foreign_key_query % table_name, nocase=False)
        for (
                keyid, column_seq_id, other_tab_name, this_column_name, other_column_name,
                on_update, on_delete, match
        ) in result:
            # these lines handle a foreign key exception in the test set
            # due to implicit reference
            if other_column_name is None:
                other_column_name = this_column_name

            table_name, other_tab_name = table_name.lower(), other_tab_name.lower()
            this_column_name, other_column_name = this_column_name.upper(), other_column_name.upper()

            # these lines handle a foreign key exception in the test set
            # due to typo in the column name
            if other_tab_name == 'author' and other_column_name == 'IDAUTHORA':
                other_column_name = 'IDAUTHOR'

            child2parent[(table_name, this_column_name)] = (other_tab_name, other_column_name)

    # make sure that every table, column in the dependency are in the table.
    dep_table_columns = set(child2parent.keys()) | set(child2parent.values())
    for table_name, column_name in dep_table_columns:
        assert table_name.lower() == table_name, "table name should be lower case"
        assert column_name.upper() == column_name, "column name should be upper case"
        assert table_name in table_name2column_properties, "table name %s missing." % table_name
        assert column_name in table_name2column_properties[table_name], \
            "column name %s should be present in table %s" % (column_name, table_name)

    return table_name2column_properties, child2parent


T = TypeVar('T')
# collapse a two level dictionary into a single level dictionary
def collapse_key(d: Dict[str, Dict[str, T]]) -> Dict[Tuple[str, str], T]:
    result = OrderedDict()
    for k1, v1 in d.items():
        for k2, v2 in v1.items():
            result[(k1, k2)] = v2
    return result


E = TypeVar('E')
def process_order_helper(dep: Dict[E, Set[E]], all: Set[E]) -> List[Set[E]]:
    dep_ks = set(dep.keys())
    for k in dep.values():
        dep_ks |= set(k)
    # assert that all the elements in the dependency relations are in the universe set
    assert len(dep_ks - all) == 0, dep_ks - all
    order = list(my_top_sort({k: v for k, v in dep.items()}))
    if len(order) == 0:
        order.append(set())
    for k in all:
        if k not in dep_ks:
            order[0].add(k)
    s = set()
    for o in order:
        s |= set(o)
    assert len(s) == len(all), (s - all, all - s)
    return order


def my_top_sort(dep: Dict[E, Set[E]]) -> List[Set[E]]:
    order = []
    elements_left = set()
    for child, parents in dep.items():
        elements_left.add(child)
        elements_left |= parents

    while len(elements_left) != 0:
        level_set = set()
        for e in elements_left:
            if e not in dep.keys():
                level_set.add(e)
            else:
                if all(parent not in elements_left for parent in dep[e]):
                    level_set.add(e)
        for e in level_set:
            elements_left.remove(e)
        order.append(level_set)
    return order


# order the columns/tables by foreign key references
def get_process_order(child2parent: Dict[Tuple[str, str], Tuple[str, str]],
                      table_column_properties: Dict[Tuple[str, str], Dict[str, Any]])\
        -> Tuple[List[Set[Tuple[str, str]]], List[Set[str]]]:
    all_table_column = set(table_column_properties.keys())
    dep_child2parent = {c: {p} for c, p in child2parent.items()}
    table_column_order = process_order_helper(dep_child2parent, all_table_column)

    all_table = set([k[0] for k in all_table_column])
    table_child2parent = defaultdict(set)
    for k1, k2 in child2parent.items():
        if k1[0] == k2[0]:
            continue
        table_child2parent[k1[0]].add(k2[0])
    table_order = process_order_helper(table_child2parent, all_table)
    return table_column_order, table_order

def get_all_table2ancestor(tc2_, child2parent):
    table_child2parent = defaultdict(set)
    for k1, k2 in child2parent.items():
        if k1[0] == k2[0]:
            continue
        table_child2parent[k1[0]].add(k2[0])
    
    t2s = {}
    def h(t):
        if t in t2s:
            return t2s[t]
        ancestors = set()
        parents = table_child2parent[t]
        ancestors |= parents
        for p in parents:
            ancestors |= h(p)
        if t not in t2s:
            t2s[t] = ancestors
        return ancestors
    for t, c in tc2_:
        h(t)
    return t2s
    


# load information from the database
# including:
# 1. column_properties: (table_name, column_name) -> column properties
#   where column properties are a map from property_name (str) -> value
# 2. foreign key relations: (table_name, column_name) -> (table_name, column_name)
# 3. column_content: (table_name, column_name) -> list, list of element types.
def get_all_db_info_path(sqlite_path: str) \
        -> Tuple[
            Dict[Tuple[str, str], Dict[str, Any]],
            Dict[Tuple[str, str], Tuple[str, str]],
            Dict[Tuple[str, str], List],
        ]:
    table_name2column_properties, child2parent = extract_table_column_properties_path(sqlite_path)

    table_name2content = OrderedDict()
    for table_name in table_name2column_properties:
        column_names = [col_name for col_name in table_name2column_properties[table_name]]
        query = 'select ' + ', '.join([' \"{c}\" '.format(c=column_name) for column_name in column_names]) + (
            ' from {t}'.format(t=table_name))
        result_type, result = exec_db_path_(sqlite_path, query, nocase=False)
        # ensure that table retrieval succeeds
        if result_type == 'exception':
            raise result
        table_name2content[table_name] = result

    table_name2column_name2elements = OrderedDict()
    for table_name in table_name2column_properties:
        column_properties, content = table_name2column_properties[table_name], table_name2content[table_name]
        # initialize the map from column name to list of elements
        table_name2column_name2elements[table_name] = OrderedDict((column_name, []) for column_name in column_properties)
        # ensure that the number of columns per row
        # is the number of columns
        if len(content) > 0:
            assert len(content[0]) == len(column_properties)
        for row in content:
            for column_name, element in zip(column_properties, row):
                table_name2column_name2elements[table_name][column_name].append(element)

    return collapse_key(table_name2column_properties), child2parent, collapse_key(table_name2column_name2elements)


def get_table_size(table_column_elements: Dict[Tuple[str, str], List]) -> Dict[str, int]:
    table_name2size = OrderedDict()
    for k, elements in table_column_elements.items():
        table_name = k[0]
        if table_name not in table_name2size:
            table_name2size[table_name] = len(elements)
    return table_name2size


def get_primary_keys(table_column_properties: Dict[Tuple[str, str], Dict[str, Any]]) -> Dict[str, List[str]]:
    table_name2primary_keys = OrderedDict()
    for (table_name, column_name), property in table_column_properties.items():
        if table_name not in table_name2primary_keys:
            table_name2primary_keys[table_name] = []
        if property['PK'] != 0:
            table_name2primary_keys[table_name].append(column_name)
    return table_name2primary_keys


def get_indexing_from_db(db_path: str, shuffle=True) -> Dict[str, List[Dict[str, Any]]]:
    table_column_properties, _, _ = get_all_db_info_path(db_path)
    all_tables_names = {t_c[0] for t_c in table_column_properties}

    table_name2indexes = {}
    for table_name in all_tables_names:
        column_names = [t_c[1] for t_c in table_column_properties if t_c[0] == table_name]
        selection_query = 'select ' + ', '.join(['"%s"' % c for c in column_names]) + ' from "' + table_name + '";'
        retrieved_results = exec_db_path_(db_path, selection_query, nocase=False)[1]
        table_name2indexes[table_name] = [{name: e for name, e in zip(column_names, row)} for row in retrieved_results]
        if shuffle:
            random.shuffle(table_name2indexes[table_name])
    return table_name2indexes


def print_table(table_name, column_names, rows):
    print('table:', table_name)
    num_cols = len(column_names)
    template = " ".join(['{:20}'] * num_cols)
    print(template.format(*column_names))
    for row in rows:
        print(template.format(*[str(r) for r in row]))

        
def wildcard_selection(q):
    p1 = '\*'
    p2 = 'count\((\s)*\*(\s)*\)'
    c1 = len(re.findall(p1, q))
    c2 = len(re.findall(p2, q, flags=re.IGNORECASE))
    return c1 - c2 > 0


def database_pprint_w_query(path, queries, print_empty=True, max_display=MAX_DISPLAY):
    tc2_, _, _ = get_all_db_info_path(path)
    tc = set(tc2_)
    queries = [q.lower() for q in queries]
    relevant_tables = set()
    for t, c in tc:
        for q in queries:
            if t.lower() in q:
                relevant_tables.add(t)
    
    tc = {(t, c) for t, c in tc2_ if t in relevant_tables}
    result = set()
    for t, c in tc:
        for q in queries:
            if c.lower() in q or wildcard_selection(q):
                result.add(c)
    return database_pprint(path, print_empty, include_cols=' '.join(result), max_display=max_display, 
                           include_tables=' '.join(relevant_tables))


def get_relevant_tables2print(path, queries):
    tc2_, _, _ = get_all_db_info_path(path)
    tc = set(tc2_)
    queries = [q.lower() for q in queries]
    relevant_tables = set()
    for t, c in tc:
        for q in queries:
            if t.lower() in q:
                relevant_tables.add(t)
    return relevant_tables


def get_all_src_tgt_path(d):
    tab_name2parents = defaultdict(dict)
    for (t1, c1), (t2, c2) in d.items():
        tab_name2parents[t1][t2] = (t1, c1), (t2, c2)
    src_tgt_paths = defaultdict(list)

    for src_node in list(tab_name2parents):
        # frontier is a set of paths
        # a path is a list of ((t1, c1), (t2, c2))
        frontier = [[tab_name2parents[src_node][n]] for n in tab_name2parents[src_node]]

        while len(frontier) != 0:
            new_frontier = []
            for path in frontier:
                n = path[-1][-1][0]

                src_tgt_paths[(src_node, n)].append(path)
                for new_parent in tab_name2parents[n]:
                    new_path = list(path)
                    new_path.append(tab_name2parents[n][new_parent])
                    new_frontier.append(new_path)
            frontier = new_frontier
    return src_tgt_paths


def simplify_dict(row):
    removed_keys = set()
    """
    pprint(row.keys())
    for (path, k), v in row.items():
        if len(path) > 0:
            prev_path = path[:-1]
            prev_k = path[-1][0]
            removed_keys.add((prev_path, prev_k))

    for k in removed_keys:
        del row[k]
        """

    # k_counter = Counter([k for (path, k) in row])
    # new_dict = OrderedDict([((((), k) if k_counter[k] == 1 else (path, k)), v) for (path, k), v in row.items()])
    new_dict = OrderedDict([(((path, k)), v) for (path, k), v in row.items()])
    return new_dict


def get_displayed_col_name(col):
    if col[0] == ():
        return '.'.join(col[1])
    else:
        path, k = col
        displayed_chain_keys = [x[0] for x in path]
        displayed_chain_keys.append(k)
        return '\n'.join('.'.join(x) for x in displayed_chain_keys)


def dedup_id_cols(simplified_dicts):
    es2ks = defaultdict(set)
    for k in simplified_dicts[0]:
        es = tuple([d[k] for d in simplified_dicts])
        es2ks[es].add(k)
    kept_ks = [list(ks)[0] for es, ks in es2ks.items()]
    return [{k: d[k] for k in kept_ks} for d in simplified_dicts]

def table_pprint_w_query(path, table_name, queries, max_display=MAX_DISPLAY, merged_tables=None):
    queries = [q.lower() for q in queries]
    tc2_, d, e = get_all_db_info_path(path)
    for k in e:
        e[k] = [str(x) for x in e[k]]
    table2size = {}
    for tc, elements in e.items():
        table2size[tc[0]] = len(elements)
    num_rows = table2size[table_name]

    if merged_tables is None:
        merged_tables = set()
    elif merged_tables == 'all':
        merged_tables = {tc[0] for tc in tc2_}
    expanded = len(merged_tables) != 0
    included_cols = set()
    for t, c in tc2_:
        if t == table_name or t in merged_tables:
            for q in queries:
                if c.lower() in q or wildcard_selection(q):
                    included_cols.add((t, c))

    pks = [k for k, v in tc2_.items() if k[0] == table_name and v['PK'] != 0]
    def find_anc(k):
        result = [k]
        while result[-1] in d:
            result.append(d[result[-1]])
        return result
    prioritized_ks = list(chain(*[find_anc(pk) for pk in pks]))

    rows = [
        OrderedDict([(((), k), e[k][i]) for k in e.keys()
                     if (k[0] == table_name and k in included_cols) or (k in pks)])
        for i in range(num_rows)
    ]

    all_src_tgt_path = get_all_src_tgt_path(d)
    all_paths = []
    for tgt_tab in merged_tables:
        for path in all_src_tgt_path[(table_name, tgt_tab)]:
            for i in range(1, len(path) + 1):
                all_paths.append(tuple(path[:i]))
    all_paths = sorted(set(all_paths), key=lambda l: len(l))

    for p in all_paths:
        for i, row in enumerate(rows):
            cur_position = (table_name, i)
            for (edge_start, edge_end) in p:
                pivot = e[edge_start][cur_position[1]]
                cur_position = (edge_end[0], e[edge_end].index(pivot))
            for k in e.keys():
                if k[0] == cur_position[0] and k in included_cols:
                    rows[i][(p, k)] = e[k][cur_position[-1]]

    simplified_rows = [simplify_dict(row) for row in rows]
    if len(simplified_rows) != 0:
        simplified_rows = dedup_id_cols(simplified_rows)
        column_names = {c: (1, idx) if c[1] not in prioritized_ks else (0, idx) for idx, c in enumerate(simplified_rows[0])}
        sorted_col_names = sorted(column_names, key=lambda c: column_names[c])
        rows = [[row[k] for k in sorted_col_names] for row in simplified_rows]
        displayed_col_names = [get_displayed_col_name(col) for col in sorted_col_names]
    else:
        displayed_col_names = ['.'.join(k) for k in included_cols if k[0] == table_name]
        rows = []

    for n in displayed_col_names:
        print(n)
        print('=====')
    print_table(table_name, displayed_col_names, rows)
    return {
        'table_name': table_name,
        'column_names': displayed_col_names,
        'rows': rows,
        'size': len(rows),
        'expanded': expanded
    }

def database_pprint(path, print_empty=True, include_cols=INCLUDE_ALL, max_display=MAX_DISPLAY, include_tables=INCLUDE_ALL):
    include_cols = include_cols.lower()
    returned_val = OrderedDict()
    tc2_, d, _ = get_all_db_info_path(path)
    _, tab_order = get_process_order(d, tc2_)
    table_column_names = [tc for tc in tc2_.keys()]
    table_names = [t for tables in tab_order for t in tables]

    for table_name in table_names:
        column_names = [c for t, c in table_column_names if t == table_name]
        if include_tables != INCLUDE_ALL and table_name.lower() not in include_tables.lower():
            continue
        c2idx = {c: idx for idx, c in enumerate(column_names)}
        
        if include_cols != INCLUDE_ALL:
            # only include relevant columns but still print at least 2 entries
            # per table
            proposed_column_names = {
                c for c in column_names if c.lower() in
                include_cols or tc2_[(table_name, c)]['PK'] != 0}
            for c in sorted(c2idx, key=lambda c: c2idx[c] if tc2_[(table_name, c)]['PK'] == 0 else -1):
                if len(proposed_column_names) < 2:
                    proposed_column_names.add(c)
                else:
                    break
            column_names = sorted(proposed_column_names,
                                  key=lambda c: c2idx[c] if tc2_[(table_name, c)]['PK'] == 0 else -1)
        column_names = sorted(column_names,
                              key=lambda c: c2idx[c]
                              if tc2_[(table_name, c)]['PK'] == 0 else -1)

        query = 'select ' + ', '.join([' \"{c}\" '.format(c=column_name) for column_name in column_names]) + (
            ' from {t}'.format(t=table_name))
        _, rows = exec_db_path_(path, query, nocase=False)
        num_rows = len(rows)

        randomly_selected_idxes = [i for i in range(num_rows)]
        random.shuffle(randomly_selected_idxes)
        randomly_selected_idxes = randomly_selected_idxes[:max_display]
        randomly_selected_idxes = sorted(randomly_selected_idxes)

        rows = [list(rows[i]) for i in randomly_selected_idxes]
        if len(rows) > 0 or print_empty:
            print_table(table_name, column_names, rows)
            returned_val[table_name] = {
                'table_name': table_name,
                'column_names': column_names,
                'rows': rows,
                'size': len(rows)
            }
    return returned_val

def get_total_size_from_indexes(table_name2indexes: Dict[str, List[Dict[str, Any]]]) -> int:
    return sum([len(v) for _, v in table_name2indexes.items()])


def get_total_size_from_path(path):
    _, _, table_column2elements = get_all_db_info_path(path)
    return sum([v for t, v in get_table_size(table_column2elements).items() if t != 'sqlite_sequence'])


def get_total_printed_size_from_path(path, queries):
    p, _, table_column2elements = get_all_db_info_path(path)
    if type(queries) != str:
        queries = ' '.join(queries)
    queries = queries.lower()
    
    return sum([v for t, v in get_table_size(table_column2elements).items() if t.lower() in queries.lower()])


def get_table2size_from_path(path):
    _, _, table_column2elements = get_all_db_info_path(path)
    return get_table_size(table_column2elements)


def get_database_schema_by_table(db_path: str) -> Dict[str, str]:
    table_names = get_table_names_path(db_path)
    table_name2schema = {}
    cursor = get_cursor_path(db_path)
    for table_name in table_names:
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='%s';" % table_name)
        table_schema = cursor.fetchall()[0][0]
        table_name2schema[table_name] = table_schema
    return table_name2schema


def round_float_in_insert(l):
    sqlparse.parse(l)
    toks = sqlparse.parse(l)
    if len(toks) > 0:
        toks = sqlparse.parse(l)[0].flatten()
        return ''.join(['%.2f' % float(t.value) if 'float' in str(t.ttype).lower() else t.value for t in toks])
    else:
        return ''
    

def get_init_sql(db_path):
    db_creation_str = subprocess.check_output("sqlite3 %s .dump" % db_path, shell=True).decode()
    ls = db_creation_str.split('\n')
    assert ls[0] == 'PRAGMA foreign_keys=OFF;'
    assert ls[1] == 'BEGIN TRANSACTION;'
    assert ls[-2] == 'COMMIT;'
    schema = '\n'.join(l for l in ls[2:-2] if 'INSERT INTO' not in l)
    content = '\n'.join(round_float_in_insert(l) for l in ls[2:-2] if 'INSERT INTO' in l)
    return {
        'schema': schema,
        'content': content
    }


def create_db(db_dict, db_path):
    
    schema = db_dict['schema']
    content = db_dict['content']
    db_creation_str = 'PRAGMA foreign_keys=ON;\nBEGIN TRANSACTION;\n'
    db_creation_str += schema + '\n'
    db_creation_str += content + '\n'
    db_creation_str += 'COMMIT;\n'
    
    intended_size = len(content.split('\n'))
    
    if os.path.exists(db_path):
        os.unlink(db_path)

    tmp_sql_file = 'tmp/%s.sql' % str(time.time())
    with open(tmp_sql_file, 'w') as out_file:
        out_file.write(db_creation_str)

    status = os.system('sqlite3 -init %s %s ""' % (tmp_sql_file, db_path))
    
    actual_size = get_total_size_from_path(db_path)
    
    os.unlink(tmp_sql_file)
    
    return {
        'db_creation_str': db_creation_str,
        'intended_size': intended_size,
        'actual_size': actual_size,
        'status': status
    }


def get_testsuite(db_path):
    db_dir = os.path.dirname(db_path)
    sqlite_files = [os.path.join(db_dir, f) for f in os.listdir(db_dir)
                    if '.sqlite' in f and 'EMPTYRARE' not in f]
    return sqlite_files


def get_nonempty_db_path(db_id):
    if db_id == 'wta_1':
        return 'data/database/wta_1/wta_1randomdrop0.sqlite'
    else:
        return 'data/database/{db_id}/{db_id}.sqlite'.format(db_id=db_id)

    
def get_empty_db_path(db_id):
    db_path = 'data/database/{db_id}/{db_id}.sqliteempty'.format(db_id=db_id)
    if os.path.exists(db_path):
        return db_path
    else:
        return 'data/database/{db_id}/{db_id}.sqlite'.format(db_id=db_id)
    

def get_random_db_path(db_id):
    k = random.randint(0, 99)
    return 'db_rand_init/%s/rand%d.sqlite' % (db_id, k)


def get_tc2values(db_id):
    db_path = 'data/database/{db_id}/{db_id}.sqliteempty'.format(db_id=db_id)
    result = {}
    p, d, e = get_all_db_info_path(db_path)
    for t, c in p:
        val_path = 'synthetic_vals/{db_id}-{t}-{c}.pkl'.format(db_id=db_id, t=t, c=c)
        elements = pkl.load(open(val_path, 'rb'))
        elements = [e if type(e) != str else e.replace('"', '').replace("'", '') for e in elements]
        result[(t, c)] = elements
    return result