from typing import List, Dict, Tuple, Any, Set
from sql_util.dbinfo import get_all_db_info_path, get_process_order, get_primary_keys, get_total_size_from_path, database_pprint
from sql_util.writedb import write_db_path
from sql_util.parse import extract_typed_value_in_comparison_from_query
from sql_util.value_typing import type_values_w_db
from sql_util.fuzz.fuzz import filter_by_primary, filter_by_unique_keys, restore_order, random_choices
import random
from itertools import chain
from collections import defaultdict
import os
import re
import time

FIRST_LEVEL_TABLE_SIZE = 200
TABLE_SIZE_POWER = 2


def mixture_samples(fuzzer, vals, target_size: int, primary_key=False, type=None):
    results = []
    for i in range(target_size):
        if primary_key and type != 'int':
            if len(vals) != 0:
                results.append(random.choice(vals))
            else:
                results.append(str(random.randint(0, target_size)))
        else:
            results.append(fuzzer.one_sample())
    return results


def perturb_single(x):
    if type(x) == int:
        r = random.random()
        if r < 0.2:
            return x - 1
        elif r > 0.8:
            return x + 1
    elif type(x) == float:
        r = random.random()
        if r < 0.2:
            return x + 0.5
        elif r > 0.8:
            return x - 0.5
    elif type(x) == str:
        if not re.search(r'\d', x):
            r = random.random()
            if r < 0.2:
                return x.replace('%', '') + ' [suffix]'
            elif r < 0.4:
                return '[prefix] ' + x.replace('%', '')
            elif r < 0.6:
                return '[prefix] ' + x.replace('%', '') + ' [suffix]'
            else:
                return x
        else:
            r = random.random()
            if r < 0.5:
                return x
            else:
                s = re.sub(r'\d', str(random.choice([0, 1])), x, random.randint(0, 3))
                return s
    else:
        return x

    
def perturb_set(xs):
    return {perturb_single(x) for x in xs}
        

def generate_table2column2elements(orig_path, target_values_by_column,
                                   table_column2natural_values=None, dbinfo=None, rep_loose=True, table_name2size=None):
    if table_column2natural_values is None:
        table_column2natural_values = defaultdict(set)
    if dbinfo is None:
        table_column_properties, child2parent, table_column2elements = get_all_db_info_path(orig_path)
    else:
        table_column_properties, child2parent, table_column2elements = dbinfo
    table_column_order, table_order = get_process_order(table_column_properties=table_column_properties,
                                                        child2parent=child2parent)
    orig_table_size = {}
    for (t, c), es in table_column2elements.items():
        orig_table_size[t] = len(es)

    if table_name2size is None:
        table_name2size = {}

        #if not shuffle_add_only:
        base = int(FIRST_LEVEL_TABLE_SIZE * random.random()) + 1
        variable_size = random.random() < 0.5
        for level, table_names in enumerate(table_order):
            for t in table_names:
                table_name2size[t] = base * (TABLE_SIZE_POWER ** level)
                if level != 0 and variable_size:
                    table_name2size[t] = 1 + int(random.random() * table_name2size[t])
    #             table_name2size[t] = min(table_name2size[t], orig_table_size[t])

    table_primary_keys = get_primary_keys(table_column_properties)
    new_table2column2elements = {}

    for table_names in table_order:
        for table_name in table_names:
            column2elements = {}
            new_table2column2elements[table_name] = column2elements
            primary_keys = table_primary_keys[table_name]

            # the column names in order
            # sometimes a column might refer to (foreign key) another column in the same table
            fuzz_order_column_names = []
            for table_columns in table_column_order:
                for table_column in table_columns:
                    if table_column[0] == table_name:
                        fuzz_order_column_names.append(table_column[1])

            orig_order_column_names = []
            for t, column_name in table_column_properties:
                if t == table_name:
                    orig_order_column_names.append(column_name)

            unique_keys = set([k for k in orig_order_column_names
                               if table_column_properties[(table_name, k)]['unique']])

            # table_size = 1 + int(random.random() * table_name2table_size[table_name])
            table_size = table_name2size[table_name]
            # print('hypothesis size', table_size)
            for column_name in fuzz_order_column_names:
                table_column = (table_name, column_name)

                if table_column in child2parent:
                    parent_t, parent_c = child2parent[table_column]
                    column2elements[column_name] = random_choices(new_table2column2elements[parent_t][parent_c],
                                                                  table_name2size[table_name])
                    continue

                # get the original values and add in target values
                # which result in fuzz_values
                orig_values = table_column2elements[(table_name, column_name)]
                target_values_to_add, o_set = set(), set(orig_values)
                for tv in target_values_by_column[(table_name, column_name)]:
                    if tv not in o_set:
                        target_values_to_add.add(tv)
                target_values_to_add = perturb_set(target_values_to_add)
                fuzz_values = target_values_to_add | set(orig_values) | set(table_column2natural_values[(table_name, column_name)])
                other_values = fuzz_values - target_values_to_add

                # check whether there are repated elements so that we won't break the "distinct" constraint, if any
                # has_rep = len(o_set) != len(orig_values)
                # allow reptition unless specified otherwise
                has_rep = rep_loose or (len(o_set) != len(orig_values))
                
                upweight = random.random() < 0.5
                # print(table_name2size[table_name])
                # if the original column already has repetition, then sample with repetition, otherwise not
                if has_rep:
                    random_list = random_choices(list(other_values | target_values_to_add),
                                                                      table_name2size[table_name])
                    if not upweight:
                        column2elements[column_name] = random_list
                    else:
                        l = [e for _ in range(3) for e in target_values_to_add]
                        l.extend(random_list)
                        l = l[:table_name2size[table_name]]
                        random.shuffle(l)
                        column2elements[column_name] = l
                else:
                    if upweight:
                        fuzz_vals_list = list(other_values)
                        target_values_to_add = list(set(target_values_to_add))
                        random.shuffle(fuzz_vals_list)
                        fuzz_vals_list = target_values_to_add + fuzz_vals_list
                        fuzz_vals_list = fuzz_vals_list[:table_name2size[table_name]]
                        random.shuffle(fuzz_vals_list)
                    else:
                        fuzz_vals_list = list(fuzz_values)[:table_name2size[table_name]]
                        random.shuffle(fuzz_vals_list)
                    assert len(fuzz_vals_list) ==  table_name2size[table_name], table_name2size[table_name]
                    fuzz_vals_list = list(set(fuzz_vals_list))[:table_name2size[table_name]]

                    column2elements[column_name] = fuzz_vals_list[:table_name2size[table_name]]
            
            # filter by unique primary key constraint
            # and restore the original column order
            # and filter unique columns to avoid repitition
            # print('column 2 elements', list(map(len, column2elements.values())))
            transformations = [
                (filter_by_primary, primary_keys),
                (restore_order, orig_order_column_names),
                (filter_by_unique_keys, unique_keys)
            ]
            # assert all(len(elements) == num_elements for elements in column2elements.values()), ([len(elements) for elements in column2elements.values()], num_elements)

            for f, arg in transformations:
                column2elements = f(column2elements, arg)
                # print('column 2 elements', list(map(len, column2elements.values())))
            new_table2column2elements[table_name] = column2elements
    return new_table2column2elements


def generate_natural_db(orig_path: str, target_path: str,
                        typed_values: List[Tuple[Tuple[str, str], str]],
                        table_column2natural_values: Dict[Tuple[str, str], Set[Any]] = None,
                        loose: bool = False,
                        rep_loose=True,
                        overwrite=False):
    target_values_by_column = type_values_w_db(orig_path, typed_values, loose)
    new_table2column2elements = generate_table2column2elements(orig_path, target_values_by_column,
                                                               table_column2natural_values, rep_loose=rep_loose)
    write_db_path(orig_path, target_path, new_table2column2elements, overwrite=overwrite)


def generate_natural_db_with_queries(orig_path: str, target_path: str, queries: List[str],
                                     table_column2natural_values: Dict[Tuple[str, str], Set[Any]] = None,
                                     loose: bool = False, rep_loose=True, overwrite: bool = False):
    typed_values = list(set(chain(*[extract_typed_value_in_comparison_from_query(query) for query in queries])))
    generate_natural_db(orig_path, target_path, typed_values,
                        table_column2natural_values=table_column2natural_values,
                        loose=loose, overwrite=overwrite, rep_loose=rep_loose)
    
    
def is_number(v):
    if v is None:
        return False
    try:
        float(v)
        return True
    except ValueError:
        return False
        

class NaturalFuzzGenerator:

    def __init__(self, orig_path: str, table_column2natural_values: Dict[Tuple[str, str], Set[Any]] = None,
                 loose: bool = False, overwrite: bool = False):
        self.orig_path, self.table_column2natural_values, self.loose, self.overwrite = orig_path, table_column2natural_values, loose, overwrite
        self.dbinfo = get_all_db_info_path(self.orig_path)
        self.cls_extract_typed_value_in_comparison_from_query_cache = {}
        table_level_set = get_process_order(table_column_properties=self.dbinfo[0], child2parent=self.dbinfo[1])[1]
        
        self.tab_order = []
        for l_set in table_level_set:
            for t in l_set:
                self.tab_order.append(t)
    
    def cls_extract_typed_value_in_comparison_from_query(self, query):
        if query not in self.cls_extract_typed_value_in_comparison_from_query_cache:
            self.cls_extract_typed_value_in_comparison_from_query_cache[query] =  extract_typed_value_in_comparison_from_query(query)
        return self.cls_extract_typed_value_in_comparison_from_query_cache[query]
        

    def generate_one_db(self, queries, target_path, target_size=None, rep_loose=True):
        a = time.time()
        typed_values = list(set(chain(*[self.cls_extract_typed_value_in_comparison_from_query(query) for query in queries])))
        b = time.time()
        assert self.orig_path != target_path
        target_values_by_column = type_values_w_db(self.orig_path, typed_values, self.loose, self.dbinfo)
        c = time.time()
        
        table_name2size = None
        if target_size is not None:
            
            table_name2size = {t: 0 for t in self.tab_order}
            all_count = 0
            while all_count < target_size:
                for t in self.tab_order:
                    table_name2size[t] += 1
                    all_count += 1
                    if all_count >= target_size:
                        break

        new_table2column2elements = generate_table2column2elements(self.orig_path, target_values_by_column,
                                                                   self.table_column2natural_values, self.dbinfo, 
                                                                   rep_loose=rep_loose, table_name2size=table_name2size)
        d = time.time()
        
        o = self.orig_path + 'empty'
        if not os.path.exists(o):
            o = self.orig_path
        if os.path.exists(target_path):
            os.unlink(target_path)

        write_db_path(o, target_path, new_table2column2elements, overwrite=self.overwrite)
        e = time.time()
        # print(b - a, c - b, d - c, e - d)
        return {'size': get_total_size_from_path(target_path)}


