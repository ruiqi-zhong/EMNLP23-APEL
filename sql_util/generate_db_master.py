from sql_util.generate_db import generate_natural_db_with_queries
from typing import List
import os
from sql_util.dbinfo import database_pprint
from sql_util.writedb import create_db_with_cascade_deletion


generation_types = ['copy', 'shuffle_add_only', 'natural']


def generate_db_master_func(orig_path: str, target_path: str, queries: List[str],
                            loose: bool = False, orig_fraction: float = .7, overwrite: bool = False,
                            generation_type='copy'):
    cascade_tmp = orig_path + 'cascade.sqlite'
    create_db_with_cascade_deletion(orig_path, cascade_tmp, overwrite=True)
    result = generate_db_master_func_(cascade_tmp, target_path, queries,
                                      loose, orig_fraction, overwrite, generation_type = 'copy'
    )
    os.unlink(cascade_tmp)
    return result


def generate_db_master_func_(orig_path: str, target_path: str, queries: List[str],
                            loose: bool = False, orig_fraction: float = .7, overwrite: bool = False,
                            generation_type='copy'):
    if generation_type not in generation_type:
        raise Exception('Generation type %s not understood' % generation_type)
    if generation_type == 'copy':
        os.system('cp %s %s' % (orig_path, target_path))
    elif generation_type == 'shuffle_add_only':
        return generate_natural_db_with_queries(orig_path=orig_path, target_path=target_path, queries=queries,
                                   loose=loose, orig_fraction=orig_fraction, overwrite=overwrite, shuffle_add_only=True)
    elif generation_type == 'natural':
        return generate_natural_db_with_queries(orig_path=orig_path, target_path=target_path, queries=queries,
                                   loose=loose, orig_fraction=orig_fraction, overwrite=overwrite, shuffle_add_only=False)


if __name__ == '__main__':
    target_path = 'tmp.sqlite'
    orig_path = 'database/concert_singer/concert_singer.sqlite'
    queries = ['select name from singer where singer_id < 10 group by country having count(*) > 3 ']

    for generation_type in generation_types:
        generate_db_master_func(orig_path, target_path, queries, overwrite=True, generation_type=generation_type)
        database_pprint(target_path)
        input()



