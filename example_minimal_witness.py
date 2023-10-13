from sql_util.minimal_witness import minimize_distinguishing_for_set, get_expected_remaining_entropy
from sql_util.dbinfo import database_pprint
from natural_fuzz import NaturalFuzzGenerator
import os

database_path = 'dog_kennels.sqlite'
random_db_path = 'random_db.sqlite'
tmp_random_db_path = 'tmp_random_db.sqlite'

fuzzer = NaturalFuzzGenerator(database_path, None, overwrite=True)

# a sql distribution
sqls = [
    'select max(FIRST_NAME) from professionals;',
    'select min(FIRST_NAME) from professionals;',
]
probablities = [0.5, 0.5]

# using the fuzzer to generate a large number of large random database with high information gain
# and keep the one with the smallest expected remaining entropy
smallest_remaining_entropy = float('inf')
for _ in range(1000):
    fuzzer.generate_one_db(sqls, tmp_random_db_path)
    remaining_entropy = get_expected_remaining_entropy(tmp_random_db_path, sqls, probabilities=probablities)
    if smallest_remaining_entropy > remaining_entropy:
        smallest_remaining_entropy = remaining_entropy
        os.system('mv {} {}'.format(tmp_random_db_path, random_db_path))
    if remaining_entropy > 1e-10:
        break

if os.path.exists(tmp_random_db_path):
    os.remove(tmp_random_db_path)


minimization_result = minimize_distinguishing_for_set(
    sqls, distinguishing_db_path=database_path, num_minimization_restart=1, verbose=True, 
    max_total_row=float('inf'), 
    distinguish_criteria='list', 
    probabilities=probablities
)
new_db_path = minimization_result['best_db_path']

database_pprint(database_path, print_empty=False)
print('^^^^^^ The original database before  ^^^^^^')
database_pprint(random_db_path, print_empty=False)
print('^^^^^^ The random database before minimization ^^^^^^')
print('We want to find a small database that can make the two sqls return different results ')
print('vvvvvv The database after minimization vvvvvv')
database_pprint(new_db_path, print_empty=False)

# tidying up the temporary files
# remove these lines if you want to inspect the database
if os.path.exists(random_db_path):
    os.remove(random_db_path)
if os.path.exists(new_db_path):
    os.remove(new_db_path)