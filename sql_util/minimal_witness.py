import os
from sql_util.dbinfo import get_indexing_from_db, get_total_size_from_path, database_pprint, get_table2size_from_path, get_total_printed_size_from_path
from sql_util.run import get_result_set, listify_result, setify_result, exec_db_path_
from typing import List
import numpy as np
from sql_util.writedb import delete_entries_by_mod, delete_random_fraction
from shutil import copyfile
import time
import random


drop_block_count = 20
min_result_size = 2


def compute_entropy(probabilities: List[float]) -> float:
    probabilities = np.array(probabilities)
    # smoothing
    probabilities += 1e-10
    probabilities /= np.sum(probabilities)
    probabilities = np.array(probabilities)
    entropy = -np.sum(np.log2(probabilities) * probabilities)
    return entropy


def add_s_before_format_suffix(path, s):
    return path.replace('.sqlite', s + '.sqlite')


def compute_expected_entropy(probabilities_groups: List[List[float]]) -> float:
    return np.sum([sum(probabilities) * compute_entropy(probabilities) for probabilities in probabilities_groups])


# remove the databases that are not on the pareto frontier
def update_db2size_entropy_frontier(db2size_entropy_frontier, exempted_db_path):
    deleted_dbs = set()
    for db1 in db2size_entropy_frontier:
        for db2 in db2size_entropy_frontier:
            if db1 != db2:
                s1, e1 = db2size_entropy_frontier[db1]
                s2, e2 = db2size_entropy_frontier[db2]
                if s1 >= s2 and e1 >= e2:
                    if s1 == s2 and e1 == e2:
                        if db1 < db2:
                            deleted_dbs.add(db1)
                    else:
                        deleted_dbs.add(db1)

    for db in deleted_dbs:
        if db != exempted_db_path:
            del db2size_entropy_frontier[db]
            if os.path.exists(db):
                os.unlink(db)

                
def get_expected_remaining_entropy(db_path, queries, probabilities=None, distinguish_criteria='list'):
    if probabilities is None:
        n = len(queries)
        probabilities = np.ones(n) / n
    else:
        probabilities = probabilities / np.sum(probabilities)
    assert len(probabilities) == len(queries)
    results = get_result_set(queries, db_path, distinguish_criteria)
    expected_entropy = compute_expected_entropy([[probabilities[idx] for idx in results[key]] for key in results])
    return expected_entropy


def get_expected_remaining_entropy_all_info(db_path, q2prob, distinguish_criteria='list', gold=None):
    queries = sorted(q2prob, key=lambda q: q2prob[q], reverse=True)
    probabilities = [q2prob[q] for q in queries]
    
    f = listify_result if distinguish_criteria == 'list' else setify_result
    
    if probabilities is None:
        n = len(queries)
        probabilities = np.ones(n) / n
    else:
        probabilities = probabilities / np.sum(probabilities)
    assert len(probabilities) == len(queries)
    results = get_result_set(queries, db_path, distinguish_criteria)
    if results is None:
        e = compute_entropy(probabilities)
        return {
            'remaining_entropy': e,
            'information_gain': 0.,
            'q2pobs': [
                q2prob
            ],
            'q2prob_left': q2prob,
            'actual_entropy': e
        }
    expected_entropy = compute_expected_entropy([[probabilities[idx] for idx in results[key]] for key in results])
    
    returned_dict = {
        'remaining_entropy': expected_entropy,
        'information_gain': compute_entropy(probabilities) - expected_entropy,
        'q2pobs': [
            normalize_q2prob({queries[idx]: probabilities[idx] for idx in results[key]})
            for key in results
        ]
    }
    
    if gold is not None:
        gold_result = f(exec_db_path_(db_path, gold)[1])
        if gold_result in results:
            q2probs_left = normalize_q2prob({queries[idx]: probabilities[idx] for idx in results[gold_result]})
        else:
            q2probs_left = None
        returned_dict['q2prob_left'] = q2probs_left
        if q2probs_left is None:
            returned_dict['actual_entropy'] = 0
        else:
            returned_dict['actual_entropy'] = compute_entropy([v for v in q2probs_left.values()])
    return returned_dict
    


def normalize_q2prob(q2prob):
    norm = sum(q2prob.values())
    return {q: q2prob[q] / norm for q in q2prob}


def minimize_distinguishing_for_set(queries: List[str], distinguishing_db_path: str,
                                    num_minimization_restart=1, verbose=False, 
                                    max_total_row=float('inf'), 
                                    distinguish_criteria='list', probabilities=None):
    assert distinguishing_db_path.endswith('.sqlite')
    if probabilities is None:
        probabilities = np.ones(len(queries)) / len(queries)
    else:
        probabilities = probabilities / np.sum(probabilities)
    step_count = 0
    db2size_entropy_frontier = {}
    if verbose:
        print('original size of the database at %s is %d' % (distinguishing_db_path, get_total_size_from_path(distinguishing_db_path)))
        print('queries: ')
        for a in queries[:20]:
            print(a)
        if len(queries) > 20:
            print('...')

    # reinitialize to the database to the original one
    for restart_idx in range(num_minimization_restart):
        db_target_path = add_s_before_format_suffix(distinguishing_db_path, '_generated_restart%d' % restart_idx)
        db_target_lookahead_path = add_s_before_format_suffix(db_target_path, 'lookahead')

        # make a copy of the original database
        os.system('touch %s' % (db_target_path))
        copyfile(distinguishing_db_path, db_target_path)
        #os.system('cp %s %s' % (distinguishing_db_path, db_target_path))
        cur_size = get_total_size_from_path(db_target_path)

        epoch_idx = 0
        if verbose:
            print('computing splits based on the starting database')

        results = get_result_set(queries, db_target_path)

        curr_entropy = compute_expected_entropy([[probabilities[idx] for idx in results[key]] for key in results])
        # stop dropping entries if the size of the database does not change.
        while cur_size > min_result_size:
            table2size = get_table2size_from_path(db_target_path)
            max_tab_name = max(table2size, key=lambda x: table2size[x])
            best_drop_tab_size_if_entropy_must_increase, best_entropy_if_must_increase = (max_tab_name, table2size[max_tab_name]), \
                                                                                         float('inf')
            entry_dropped_for_epoch = False

            # trying to drop entries without increasing the resulting expected entropy
            table_name2indexes = get_indexing_from_db(db_target_path, shuffle=True)

            # deleting entries from the largest table first
            for table_name in sorted(table2size, key=lambda k: table2size[k], reverse=True):
                entries = table_name2indexes[table_name]
                if len(entries) == 0:
                    continue
                period = min(len(entries), drop_block_count)

                for shot in range(period):
                    if cur_size <= min_result_size:
                        break

                    if verbose:
                        print('dropping table %s with period %d and shot %d' % (table_name, period, shot))
                    # tentatively drop an entry, and put the resulting database in a new path
                    os.system('touch %s' % (db_target_lookahead_path))
                    # delete_entries_from_db(db_target_path, db_target_lookahead_path, table_name, entries)
                    # num_deleted_rows = delete_entries_by_mod(db_target_path, db_target_lookahead_path, table_name, period, remainder)
                    num_deleted_rows = delete_random_fraction(db_target_path, db_target_lookahead_path, table_name, 1. / period)

                    # check whether the new database decreases entropy
                    #results = get_result_set(queries, db_target_lookahead_path)
                    #expected_entropy = compute_expected_entropy([[probabilities[idx] for idx in results[key]] for key in results])
                    expected_entropy = get_expected_remaining_entropy(db_target_lookahead_path, queries, 
                                                                      probabilities, distinguish_criteria=distinguish_criteria)
                    if expected_entropy > curr_entropy or num_deleted_rows == 0:
                        os.unlink(db_target_lookahead_path)
                        # if there are entries that are indeed deleted
                        # keep track of the table name, period and remainder
                        if expected_entropy < best_entropy_if_must_increase and num_deleted_rows != 0:
                            best_entropy_if_must_increase = expected_entropy
                            best_drop_tab_size_if_entropy_must_increase = table_name, period
                    else:
                        cur_size = get_total_size_from_path(db_target_path)
                        if verbose:
                            print('successfully dropped one block, now size', cur_size)
                        os.rename(db_target_lookahead_path, db_target_path)
                        curr_entropy = expected_entropy
                        entry_dropped_for_epoch = True

                    if cur_size <= min_result_size:
                        break

                    step_count += 1

            # at the end of the epoch
            # copy and add the database to the frontier
            end_of_epoch_db_path = db_target_path + '_end_of_epoch%d' % epoch_idx
            
            # os.system('cp %s %s' % (db_target_path, end_of_epoch_db_path))
            copyfile(db_target_path, end_of_epoch_db_path)
            db2size_entropy_frontier[end_of_epoch_db_path] = \
                (get_total_size_from_path(end_of_epoch_db_path), curr_entropy)

            if not entry_dropped_for_epoch:
                if verbose:
                    print('force dropping a subset')
                # delete an entry if necessary
                # copy and add the database to the frontier
                # table_name, (period, remainder) = best_drop_tab_pr_if_entropy_must_increase
                table_name, period = best_drop_tab_size_if_entropy_must_increase
                end_of_epoch_delete_1_db_path = db_target_path + '_end_of_epoch%d_delete_1' % epoch_idx
                delete_random_fraction(db_target_path, end_of_epoch_delete_1_db_path, table_name, 1./period)

                db2size_entropy_frontier[end_of_epoch_delete_1_db_path] = \
                    (get_total_size_from_path(end_of_epoch_delete_1_db_path), best_entropy_if_must_increase)

                # delete one entry and move on
                # os.system('cp %s %s' % (end_of_epoch_delete_1_db_path, db_target_path))
                copyfile(end_of_epoch_delete_1_db_path, db_target_path)
                curr_entropy = best_entropy_if_must_increase

            epoch_idx += 1
            cur_size = get_total_size_from_path(db_target_path)
            update_db2size_entropy_frontier(db2size_entropy_frontier, exempted_db_path=db_target_path)

            if verbose:
                print('Minimizing DB Restart %d, Epoch %d, the current total size is %d, the current entropy is %f, dumped minimized db into %s'
                      % (restart_idx + 1, epoch_idx, cur_size, curr_entropy, db_target_path))
        if verbose:
            print('Minimizing DB Restart %d complete after %d epochs, total size is %d, the entropy is %f, dumped minimized db into %s'
                  % (restart_idx + 1, epoch_idx, cur_size, curr_entropy, db_target_path))

        if os.path.exists(db_target_path):
            os.unlink(db_target_path)

    update_db2size_entropy_frontier(db2size_entropy_frontier, exempted_db_path='')
    # select the largest table satisfying the table size constraint
    # this has the lowest expected entropy possible because all remaining databases are already at the pareto frontier
    best_db_path = max([db_path for db_path, (s, e) in db2size_entropy_frontier.items() if s <= max_total_row],
                       key=lambda k: db2size_entropy_frontier[k][0])

    for db_path in db2size_entropy_frontier:
        if db_path != best_db_path:
            os.unlink(db_path)
            
    return {
        'best_db_path': best_db_path,
        'best_size': get_total_printed_size_from_path(best_db_path, queries),
        'best_expected_entropy': db2size_entropy_frontier[best_db_path][1],
        'frontier': db2size_entropy_frontier,
        "step_count": step_count
    }


def drop_random_to_target_size_approximate(db_path, target_size, target_path):
    cur_path = target_path
    tmp_path = cur_path + '_'
    os.system('cp %s %s' % (db_path, cur_path))
    os.system('chmod 777 %s' % cur_path)

    cur_size = get_total_size_from_path(cur_path)
    continue_dropping = True
    num_attempt = 20
    
    while num_attempt > 0:
        
        table2size = get_table2size_from_path(cur_path)
        table = random.choice(list(table2size))
        if table2size[table] == 0:
            continue
        
        fraction = max(0.2, 1. / table2size[table])
        num_r_dropped = delete_random_fraction(cur_path, tmp_path, table, fraction)
        new_size = get_total_size_from_path(tmp_path)

        if new_size <= target_size * 0.8:
            num_attempt -= 1
            continue
        elif new_size <= target_size:
            break
        else:
            os.system('mv %s %s' % (tmp_path, cur_path))
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)
        
    return cur_path


def drop_random_to_target_size(db_path, target_size):
    cur_path = 'tmp/' + str(time.time())
    tmp_path = cur_path + '_'
    os.system('cp %s %s' % (db_path, cur_path))
    os.system('chmod 777 %s' % cur_path)

    cur_size = get_total_size_from_path(cur_path)
    continue_dropping = True
    num_attempt = 10
    
    while num_attempt > 0:
        
        table2size = get_table2size_from_path(cur_path)
        table = random.choice(list(table2size))
        if table2size[table] == 0:
            continue
        
        fraction = max(0.2, 1. / table2size[table])
        num_r_dropped = delete_random_fraction(cur_path, tmp_path, table, fraction)
        new_size = get_total_size_from_path(tmp_path)

        if new_size < target_size:
            os.unlink(tmp_path)
            num_attempt -= 1
        else:
            os.system('mv %s %s' % (tmp_path, cur_path))
        if new_size == target_size:
            return cur_path
    return cur_path



