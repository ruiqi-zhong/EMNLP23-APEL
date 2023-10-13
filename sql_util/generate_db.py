from typing import List, Dict, Tuple
from sql_util.dbinfo import get_all_db_info_path, get_process_order, get_primary_keys
from sql_util.writedb import write_db_path
from sql_util.parse import extract_typed_value_in_comparison_from_query
from sql_util.value_typing import type_values_w_db, is_num, is_int
from sql_util.fuzz.fuzz import filter_by_primary, filter_by_unique_keys, restore_order, random_choices, get_fuzzer_from_type_str
import random
from itertools import chain

FIRST_LEVEL_TABLE_SIZE = 50
TABLE_SIZE_POWER = 2


def generate_variant(value: str) -> List[str]:
    if is_int(value):
        num = int(value)
        return [str(i) for i in [num - 1, num, num + 1]]
    if is_num(value):
        num = float(value)
        return [str(i) for i in [num - .1, num, num + .1]]
    if '%' in value:
        core = value.replace('%', '')
        return [core, '[text] ' + core, core + ' [text]', '[text] ' + core + ' [text]']
    return [value]


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


def generate_table2column2elements(orig_path, target_values_by_column, orig_fraction, shuffle_add_only=False):
    table_column_properties, child2parent, table_column2elements = get_all_db_info_path(orig_path)
    table_column_order, table_order = get_process_order(table_column_properties=table_column_properties,
                                                        child2parent=child2parent)
    if not shuffle_add_only:
        for key in target_values_by_column:
            table_column2elements[key].extend([v_ for v in target_values_by_column[key] for v_ in generate_variant(v)])
    fuzzers = {k: get_fuzzer_from_type_str(table_column_properties[k]['type'], table_column2elements[k], p=1-orig_fraction) for k in table_column_properties}
    table_name2size = {}

    if not shuffle_add_only:
        base = int(FIRST_LEVEL_TABLE_SIZE * random.random()) + 1
        variable_size = random.random() < 0.5
        for level, table_names in enumerate(table_order):
            for t in table_names:
                table_name2size[t] = base * (TABLE_SIZE_POWER ** level)
                if level != 0 and variable_size:
                    table_name2size[t] = 1 + int(random.random() * table_name2size[t])
    else:
        for (tab_name, _), elements in table_column2elements.items():
            table_name2size[tab_name] = len(elements)

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
                if not shuffle_add_only:
                    if table_column in child2parent:
                        parent_table, parent_column = child2parent[table_column]
                        assert parent_table in new_table2column2elements.keys(), "table %s should have been fuzzed" % parent_table
                        parent_elements = new_table2column2elements[parent_table][parent_column]
                        column2elements[column_name] = random_choices(parent_elements, k=table_size)
                    else:
                        # the type of the data might be wrong, but the sqlite interface are able to deal with that
                        column2elements[column_name] = mixture_samples(fuzzers[(table_name, column_name)], table_column2elements[(table_name, column_name)], table_size, table_column_properties[(table_name, column_name)]['PK'] == 1, table_column_properties[(table_name, column_name)]['type'])
                else:
                    # get the original values and add in target values
                    # which result in fuzz_values
                    orig_values = table_column2elements[(table_name, column_name)]
                    target_values_to_add, o_set = set(), set(orig_values)
                    for tv in target_values_by_column[(table_name, column_name)]:
                        if tv not in o_set:
                            target_values_to_add.add(tv)
                    fuzz_values = target_values_to_add | set(orig_values)

                    # check whether there are repated elements so that we won't break the "distinct" constraint, if any
                    has_rep = len(o_set) != len(orig_values)

                    # if the original column already has repetition, then sample with repetition, otherwise not
                    if has_rep:
                        column2elements[column_name] = random_choices(list(fuzz_values), table_name2size[table_name])
                    else:
                        fuzz_vals_list = list(fuzz_values)
                        random.shuffle(fuzz_vals_list)
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

            for f, arg in transformations:
                column2elements = f(column2elements, arg)
                # print('column 2 elements', list(map(len, column2elements.values())))
            new_table2column2elements[table_name] = column2elements
    return new_table2column2elements


def generate_natural_db(orig_path: str, target_path: str, typed_values: List[Tuple[Tuple[str, str], str]],
                        loose: bool = False, orig_fraction: float = .7, overwrite=False, shuffle_add_only=False):
    target_values_by_column = type_values_w_db(orig_path, typed_values, loose)
    new_table2column2elements = generate_table2column2elements(orig_path, target_values_by_column, orig_fraction, shuffle_add_only=shuffle_add_only)
    write_db_path(orig_path, target_path, new_table2column2elements, overwrite=overwrite)


def generate_natural_db_with_queries(orig_path: str, target_path: str, queries: List[str],
                                     loose: bool = False, orig_fraction: float = .7, overwrite: bool = False, shuffle_add_only: bool=False):
    typed_values = list(set(chain(*[extract_typed_value_in_comparison_from_query(query) for query in queries])))
    generate_natural_db(orig_path, target_path, typed_values, loose=loose,
                        orig_fraction=orig_fraction, overwrite=overwrite, shuffle_add_only=shuffle_add_only)


# generate naturalistic databases
def generate_natural_db_with_queries_wrapper(args: Tuple[str, str, List[str], Dict[str, str]]):
    orig_path, target_path, queries, kwargs = args
    generate_natural_db_with_queries(orig_path, target_path, queries, **kwargs)
