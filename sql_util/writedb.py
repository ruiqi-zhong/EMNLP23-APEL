import os
from typing import List, Tuple, Dict
from sql_util.run import get_cursor_path
from shutil import copyfile
from sql_util.dbinfo import table_name_query, get_table_names_path, extract_table_column_properties_path, \
    get_process_order, get_all_db_info_path, get_total_size_from_path, get_database_schema_by_table
import random
from typing import Any
import re
from pprint import pprint


def insert_row(cursor, table_name: str, column_names: List[str], row: Tuple) -> str:
    assert len(row) == len(column_names), "number of elements per row needs to be the same as number of columns"
    dummy_args = " ,".join(["?"] * len(column_names))
    q = ''' INSERT INTO {table_name}({column_names}) VALUES ({dummy_args}) '''\
        .format(column_names=','.join(['[%s]' % c for c in column_names]), table_name=table_name, dummy_args=dummy_args)
    try:
        row = tuple([r if type(r) != str else r.strip() for r in row])
        cursor.execute(q, row)
        return 'success'
    except Exception as e:
        print('unable to insert the following')
        print(q)
        print(row)
        print(e)
        return "fails"


def delete_random_fraction(orig_path: str, target_path: str, table_name: str, fraction: float):
    if orig_path != target_path:
        os.system('cp {orig_path} {target_path}'.format(orig_path=orig_path, target_path=target_path))
    orig_size = get_total_size_from_path(orig_path)
    cursor = get_cursor_path(target_path)

    r = int((2 ** 64) * (fraction - 0.5))
    cursor.execute('DELETE FROM {table_name} WHERE random() < {r}'.format(table_name=table_name, r=r))
    cursor.connection.commit()
    cursor.connection.close()
    new_size = get_total_size_from_path(target_path)
    num_deleted_rows = orig_size - new_size
    return num_deleted_rows


def insert_table(cursor, table_name: str, column2elements: Dict[str, List]) -> None:
    column_names = list(column2elements.keys())
    num_rows = len(column2elements[column_names[0]])

    one_success = False
    all_success = []
    for row_id in range(num_rows):
        row = tuple([column2elements[column_name][row_id] for column_name in column_names])
        insertion_result = insert_row(cursor, table_name, column_names, row)
        if insertion_result == 'success':
            one_success = True
        all_success.append(insertion_result == 'success')
    if not one_success and num_rows != 0:
        print('no successful insertion for table %s' % table_name)
    if not all(all_success):
        print('%d/%d success.' % (sum(all_success), len(all_success)))
        
        
# we write a new data base
# by copying from an empty database
# and insert columns
def write_db_path(orig_path: str, new_db_path: str, table2column2elements: Dict[str, Dict[str, List]],
             overwrite: bool = False) -> None:
    if os.path.exists(new_db_path) and not overwrite:
        print('new database already exists.')
        return
    assert orig_path != new_db_path

    
    empty_db_path = init_empty_db_from_orig_(orig_path, empty_db_path=new_db_path)
    
#     empty_db_path = init_empty_db_from_orig_(orig_path)
#     os.system('cp %s %s' % (empty_db_path, new_db_path))
#     os.unlink(empty_db_path)

    cursor = get_cursor_path(new_db_path)
    p, d, e = get_all_db_info_path(new_db_path)

    _, table_process_order = get_process_order(d, p)

    table_name2column_properties, _ = extract_table_column_properties_path(orig_path)
    
    for table_names in table_process_order:
        for table_name in table_names:
            column2elements = table2column2elements[table_name]
            # the order of the column should stay the same
            columns = list(column2elements.keys())
            orig_columns = list(table_name2column_properties[table_name].keys())
            assert columns == orig_columns, (columns, orig_columns)
            insert_table(cursor, table_name, column2elements)
    cursor.connection.commit()
    cursor.connection.close()


remove_query = 'delete from %s;'
EMPTY = 'EMPTYRARE'


def init_empty_db_from_orig_(sqlite_path: str, verbose: bool = False, empty_db_path=None) -> str:
    if empty_db_path is None:
        empty_db_path = sqlite_path + EMPTY + str(random.randint(0, 10000000000))

    assert empty_db_path != sqlite_path

    # copy the old database
    # initialize a new one and get the cursor
    copyfile(sqlite_path, empty_db_path)
    cursor = get_cursor_path(empty_db_path)
    table_names = get_table_names_path(sqlite_path)
    for table_name in table_names:
        cursor.execute(remove_query % table_name)
    if verbose:
        cursor.execute(table_name_query)
        result = cursor.fetchall()
        print('Tables created: ')
        print(result)
    cursor.connection.commit()
    cursor.connection.close()
    return empty_db_path


def subsample_db(orig_path: str, target_path: str,
                 delete_fraction: float = 0.5, overwrite: bool = False):
    if os.path.exists(target_path) and not overwrite:
        raise Exception('Path %s exists, do not overwrite.' % target_path)
    copyfile(orig_path, target_path)
    cursor = get_cursor_path(target_path)

    table_column_properties, child2parent, _ = get_all_db_info_path(target_path)
    _, table_order = get_process_order(child2parent, table_column_properties)
    for table in table_order:
        cursor.execute('DELETE TOP (%d) PERCENT FROM %s;' % (int(delete_fraction * 100), table))
    cursor.connection.commit()
    cursor.connection.close()


# delete an entry from the original path and store the result in the target path
def delete_entry_from_db(orig_path: str, target_path: str, table_name: str, entry: Dict[str, Any]):
    if orig_path != target_path:
        os.system('cp {orig_path} {target_path}'.format(orig_path=orig_path, target_path=target_path))
    deletion_query = 'delete from "{table_name}" where '.format(table_name=table_name)
    for column_name, val in entry.items():
        deletion_query += '"{column_name}" = "{val}" AND '.format(column_name=column_name, val=val)
    deletion_query += ';'
    deletion_query = deletion_query.replace('AND ;', ';')

    cursor = get_cursor_path(target_path)
    cursor.execute(deletion_query)
    cursor.connection.commit()
    cursor.connection.close()


def add_on_delete_cascade(lines):
    delete_suffix = ' ON DELETE CASCADE'
    new_lines = []
    for l in lines.split('\n'):
        if 'REFERENCES ' in l.upper() and delete_suffix not in l:
            if l[-1] == ',':
                l = l[:-1] + delete_suffix + ','
            else:
                l += delete_suffix
        new_lines.append(l)
    return '\n'.join(new_lines)


def add_on_delete_cascade_to_db(db_path: str):
    table_names = get_table_names_path(db_path)
    cursor = get_cursor_path(db_path)
    for table_name in table_names:
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='%s';" % table_name)
        table_schema = cursor.fetchall()[0][0]
        print(table_schema)
        added_cascade_schema = add_on_delete_cascade(table_schema)
        print(added_cascade_schema)
        if "'" in added_cascade_schema:
            cursor.execute('UPDATE sqlite_master SET sql="%s" WHERE type="table" AND name="%s";' % (added_cascade_schema, table_name))
        else:
            cursor.execute("UPDATE sqlite_master SET sql='%s' WHERE type='table' AND name='%s';" % (added_cascade_schema, table_name))
    cursor.connection.commit()
    
def remove_all_fk(schema: str) -> str:
    tmp = 'CREATE TABLE rankings("ranking_date" DATE,"ranking" INT,"player_id" INT,"ranking_points" INT,"tours" INT,FOREIGN KEY(player_id) REFERENCES players(player_id) ON DELETE CASCADE)'
    if schema == tmp:
        return 'CREATE TABLE rankings("ranking_date" DATE,"ranking" INT,"player_id" INT,"ranking_points" INT,"tours" INT)'
    lines = schema.split('\n')
    return '\n'.join([l for l in lines if 'foreign key' not in l.lower()])
    
def remove_all_fk_db(db_path: str):
    table_names = get_table_names_path(db_path)
    cursor = get_cursor_path(db_path)
    for table_name in table_names:
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='%s';" % table_name)
        table_schema = cursor.fetchall()[0][0]
#         print(table_schema)
        new_schema = re.sub(r',[\n\s]+\)', '\n)', remove_all_fk(table_schema))
#         print(new_schema)
        if "'" in new_schema:
            cursor.execute('UPDATE sqlite_master SET sql="%s" WHERE type="table" AND name="%s";' % (new_schema, table_name))
        else:
            cursor.execute("UPDATE sqlite_master SET sql='%s' WHERE type='table' AND name='%s';" % (new_schema, table_name))
    cursor.connection.commit()
    cursor.connection.close()
    
    
def write_new_fk(db_path: str, new_fk: Dict[Tuple[str, str], Tuple[str, str]]):
    table_names = get_table_names_path(db_path)
    cursor = get_cursor_path(db_path)
    for table_name in table_names:
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='%s';" % table_name)
        table_schema = cursor.fetchall()[0][0]
        new_schema = str(table_schema)

        new_table_fks = [k for k in new_fk.items() if k[0][0].lower() == table_name.lower()]
        for fk in new_table_fks:
            new_schema = re.sub(
                r'[\n\s]*\)$',
                ',\nFOREIGN KEY (`%s`) REFERENCES `%s`("%s") ON DELETE CASCADE\n)' % (fk[0][1], fk[1][0], fk[1][1]), 
                new_schema
            ).replace('`', "'").replace('"', "'")

        if "'" in new_schema:
            cursor.execute('UPDATE sqlite_master SET sql="%s" WHERE type="table" AND name="%s";' % (new_schema, table_name))
        else:
            cursor.execute("UPDATE sqlite_master SET sql='%s' WHERE type='table' AND name='%s';" % (new_schema, table_name))
    cursor.connection.commit()
    cursor.connection.close()


def retype_int_w_str(db_path, table_name, column_name):
    in_db_table_name = None
    t2schema = get_database_schema_by_table(db_path)
    for t in t2schema:
        if t.lower() == table_name.lower():
            in_db_table_name = t
    
    table_schema = t2schema[in_db_table_name]

    segments = table_schema.split(',')
    #print(column_name)
    for i, seg in enumerate(segments):
        if re.search(r'\b%s\b' % column_name, seg, flags=re.IGNORECASE):
            new_seg = re.sub(r'\binteger\b', 'text', seg, flags=re.IGNORECASE)
            new_seg = re.sub(r'\bint\b', 'text', new_seg, flags=re.IGNORECASE)
            segments[i] = new_seg
    new_table_schema = ','.join(segments)
    new_table_schema = new_table_schema.replace('AUTOINCREMENT', '')
    #print(new_table_schema)
    # print(new_table_schema)
    
    cursor = get_cursor_path(db_path)
    if "'" in new_table_schema:
        cursor.execute('UPDATE sqlite_master SET sql="%s" WHERE type="table" AND name="%s";' % (new_table_schema, in_db_table_name))
    else:
        cursor.execute("UPDATE sqlite_master SET sql='%s' WHERE type='table' AND name='%s';" % (new_table_schema, in_db_table_name))
    cursor.connection.commit()
    cursor.connection.close()
    
    table_schema = {k.lower(): result for k, result in get_database_schema_by_table(db_path).items()}[table_name]
    if new_table_schema != table_schema:
        print(table_schema)
        print('update fails')
        exit(0)
    


def delete_entries_by_mod(orig_path: str, target_path: str, table_name: str, period: int, remainder: int):
    if orig_path != target_path:
        os.system('cp {orig_path} {target_path}'.format(orig_path=orig_path, target_path=target_path))
    orig_size = get_total_size_from_path(orig_path)
    cursor = get_cursor_path(target_path)
    cursor.execute('DELETE FROM {table_name} WHERE ROWID % {period} = {remainder}'.format(table_name=table_name,
                                                                                       period=period, remainder=remainder))
    cursor.connection.commit()
    cursor.connection.close()
    new_size = get_total_size_from_path(target_path)
    num_deleted_rows = orig_size - new_size
    if num_deleted_rows == 0:
        print(orig_path)
        print(target_path)
        print(table_name)
        print(period)
        print(remainder)
    return orig_size - new_size

import pickle as pkl
import os
ss_path = 'special/schema_special.pkl'
special_schema = {}
if os.path.exists(ss_path):
    special_schema = pkl.load(open('special/schema_special.pkl', 'rb'))
    special_schema = {(os.path.dirname(k[0]), k[1]): v for k, v in special_schema.items()}

def create_db_with_ordered_schema(target_path, ordered_t2s):
    if os.path.exists(target_path):
        os.unlink(target_path)
    c = get_cursor_path(target_path)
    for t, creation in ordered_t2s.items():
        c.execute(creation)
    c.connection.commit()
    
    
def create_db_with_cascade_deletion(db_path: str, target_path: str, overwrite: bool = True):
    if os.path.exists(target_path) and not overwrite:
        print('WARNING: path %s already exists' % target_path)
    elif os.path.exists(target_path):
        os.unlink(target_path)

    schema_by_table = get_database_schema_by_table(db_path)
    for tab in schema_by_table:
        if (os.path.dirname(db_path), tab) in special_schema:
            schema_by_table[tab] = special_schema[(os.path.dirname(db_path), tab)]
        else:
            schema_by_table[tab] = add_on_delete_cascade(schema_by_table[tab])
    c = get_cursor_path(target_path)
    for tab, creation in schema_by_table.items():
        print(creation)
        if tab == 'sqlite_sequence':
            continue
        c.execute(creation)
    c.connection.commit()
    
    _, _, e = get_all_db_info_path(db_path)
    p, d, _ = get_all_db_info_path(target_path)
    _, table_order = get_process_order(d, p)
    
    for tables in table_order:
        for tab in tables:
            col2elements = {col: e[(tab, col)] for (tab_, col) in p if tab_ == tab}
            insert_table(c, tab, col2elements)
    c.connection.commit()
    c.connection.close()


