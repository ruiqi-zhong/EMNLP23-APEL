import re
import sqlparse
from typing import List, Tuple, Set, Iterator, Dict, Any, Union
from sqlparse.sql import Comparison, Identifier
from sqlparse.tokens import Whitespace
import itertools
from collections import namedtuple
import re
from pprint import pprint


Token = namedtuple('Token', ['ttype', 'value'])
VALUE_NUM_SYMBOL = 'VALUERARE'
QUOTE_CHARS = {'`', '\'', '"'}


def tokenize(query: str) -> List[Token]:
    tokens = list([Token(t.ttype, t.value) for t in sqlparse.parse(query)[0].flatten()])
    return tokens


def join_tokens(tokens: List[Token]) -> str:
    return ''.join([x.value for x in tokens]).strip().replace('  ', ' ')


def round_trip_test(query: str) -> None:
    tokens = tokenize(query)
    reconstructed = ''.join([token.value for token in tokens])
    assert query == reconstructed, "Round trip test fails for string %s" % query


def postprocess(query: str) -> str:
    query = query.replace('> =', '>=').replace('< =', '<=').replace('! =', '!=')
    return query


def drop_where(query: str) -> str:
    result_query = query
    parse = sqlparse.parse(query)
    if len(parse) < 1:
        return query
    for t in parse[0].tokens:
        if isinstance(t, sqlparse.sql.Where):
            result_query = result_query.replace(str(t), '')
    return result_query

def replace_where(query: str) -> str:
    result_query = query
    parse = sqlparse.parse(query)
    if len(parse) < 1:
        return query
    for t in parse[0].tokens:
        if isinstance(t, sqlparse.sql.Where):
            orig_where = str(t)
            new_where = ''
            if '!=' in orig_where:
                new_where = orig_where.replace('!=', '=')
            elif '=' in orig_where:
                new_where = orig_where.replace('=', '!=')
            result_query = result_query.replace(orig_where, new_where)
    return result_query


# strip_query, reformat_query and replace values
# were implemented by Yu Tao for processing CoSQL
def strip_query(query: str) -> Tuple[List[str], List[str]]:
    query_keywords, all_values = [], []

    # then replace all stuff enclosed by "" with a numerical value to get it marked as {VALUE}

    # Tao's implementation is commented out here.
    """
    str_1 = re.findall("\"[^\"]*\"", query)
    str_2 = re.findall("\'[^\']*\'", query)
    values = str_1 + str_2
        """

    toks = sqlparse.parse(query)[0].flatten()
    values = [t.value for t in toks if t.ttype == sqlparse.tokens.Literal.String.Single or t.ttype == sqlparse.tokens.Literal.String.Symbol]


    for val in values:
        all_values.append(val)
        query = query.replace(val.strip(), VALUE_NUM_SYMBOL)

    query_tokenized = query.split()
    float_nums = re.findall("[-+]?\d*\.\d+", query)
    all_values += [qt for qt in query_tokenized if qt in float_nums]
    query_tokenized = [VALUE_NUM_SYMBOL if qt in float_nums else qt for qt in query_tokenized]

    query = " ".join(query_tokenized)
    int_nums = [i.strip() for i in re.findall("[^tT]\d+", query)]

    all_values += [qt for qt in query_tokenized if qt in int_nums]
    query_tokenized = [VALUE_NUM_SYMBOL if qt in int_nums else qt for qt in query_tokenized]
    # print int_nums, query, query_tokenized

    for tok in query_tokenized:
        if "." in tok:
            table = re.findall("[Tt]\d+\.", tok)
            if len(table) > 0:
                to = tok.replace(".", " . ").split()
                to = [t.lower() for t in to if len(t) > 0]
                query_keywords.extend(to)
            else:
                query_keywords.append(tok.lower())

        elif len(tok) > 0:
            query_keywords.append(tok.lower())
    return query_keywords, all_values


def reformat_query(query: str) -> str:
    query = query.strip().replace(";", "").replace("\t", "")
    query = ' '.join([t.value for t in tokenize(query) if t.ttype != sqlparse.tokens.Whitespace])
    t_stars = ["t1.*", "t2.*", "t3.*", "T1.*", "T2.*", "T3.*"]
    for ts in t_stars:
        query = query.replace(ts, "*")
    return query


def replace_values(sql: str) -> Tuple[List[str], Set[str]]:
    sql = sqlparse.format(sql, reindent=False, keyword_case='upper')
    # sql = re.sub(r"(<=|>=|!=|=|<|>|,)", r" \1 ", sql)
    sql = re.sub(r"(T\d+\.)\s", r"\1", sql)
    query_toks_no_value, values = strip_query(sql)
    return query_toks_no_value, set(values)


# extract the non-value tokens and the set of values
# from a sql query
def extract_query_values(sql: str) -> Tuple[List[str], Set[str]]:
    reformated = reformat_query(query=sql)
    query_value_replaced, values = replace_values(reformated)
    return query_value_replaced, values


# plug in the values into query with value slots
def plugin(query_value_replaced: List[str], values_in_order: List[str]) -> str:
    q_length = len(query_value_replaced)
    query_w_values = query_value_replaced[:]
    value_idx = [idx for idx in range(q_length) if query_value_replaced[idx] == VALUE_NUM_SYMBOL.lower()]
    assert len(value_idx) == len(values_in_order)

    for idx, value in zip(value_idx, values_in_order):
        query_w_values[idx] = value
    return ' '.join(query_w_values)


# a generator generating all possible ways of
# filling values into predicted query
def plugin_all_permutations(query_value_replaced: List[str], values: Set[str]) -> Iterator[str]:
    num_slots = len([v for v in query_value_replaced if v == VALUE_NUM_SYMBOL.lower()])
    for values in itertools.product(*[list(values) for _ in range(num_slots)]):
        yield plugin(query_value_replaced, list(values))


# given the gold query and the model prediction
# extract values from the gold, extract predicted sql with value slots
# return 1) number of possible ways to plug in gold values and 2) an iterator of predictions with value plugged in
def get_all_preds_for_execution(gold: str, pred: str) -> Tuple[int, Iterator[str]]:
    _, gold_values = extract_query_values(gold)
    pred_query_value_replaced, _ = extract_query_values(pred)
    num_slots = len([v for v in pred_query_value_replaced if v == VALUE_NUM_SYMBOL.lower()])
    num_alternatives = len(gold_values) ** num_slots
    return num_alternatives, plugin_all_permutations(pred_query_value_replaced, gold_values)


def remove_distinct(s):
    toks = [t.value for t in list(sqlparse.parse(s)[0].flatten())]
    return ''.join([t for t in toks if t.lower() != 'distinct'])


def extract_all_comparison_from_node(node: Token) -> List[Comparison]:
    comparison_list = []
    if hasattr(node, 'tokens'):
        for t in node.tokens:
            comparison_list.extend(extract_all_comparison_from_node(t))
    if type(node) == Comparison:
        comparison_list.append(node)
    return comparison_list


def extract_all_comparison(query: str) -> List[Comparison]:
    tree_ = sqlparse.parse(query)
    if len(tree_) < 1:
        return []
    tree = tree_[0]
    comparison_list = extract_all_comparison_from_node(tree)
    return comparison_list


def extract_toks_from_comparison(comparison_node: Comparison) -> List[Token]:
    tokens = [t for t in comparison_node.tokens if t.ttype != Whitespace]
    return tokens


def extract_info_from_comparison(comparison_node: Comparison) -> Dict[str, Any]:
    tokens = extract_toks_from_comparison(comparison_node)
    left, op, right = tokens

    returned_dict = {
        'left': left,
        'op': op.value,
        'right': right
    }

    if type(left) != Identifier:
        return returned_dict

    table = None
    if len(left.tokens) == 3 and re.match('^[tT][0-9]$', left.tokens[0].value) is None:
        table = left.tokens[0].value.lower()
    col = left.tokens[-1].value

    if type(right) == Identifier:
        if len(right.tokens) == 1 and type(right.tokens[0]) == sqlparse.sql.Token:
            right_val = right.tokens[0].value
        else:
            return returned_dict
    elif type(right) == sqlparse.sql.Token:
        right_val = right.value
    else:
        return returned_dict

    returned_dict['table_col'], returned_dict['val'] = (rm_placeholder(table), rm_placeholder(col.upper())), rm_placeholder(process_str_value(right_val))

    return returned_dict


def extract_all_comparison_from_query(query: str) -> List[Dict[str, Any]]:
    comparison_list = extract_all_comparison(query)
    return [extract_info_from_comparison(c) for c in comparison_list]


def rm_placeholder(s: Union[str, None]) -> Union[str, None]:
    if s is None:
        return None
    return re.sub('placeholderrare', '', s, flags=re.IGNORECASE)


in_tuple_pattern = re.compile('(?:WHERE|OR|AND) (?:\w*\.)?([\w]*) IN \((.*?)\)')
def typed_values_in_tuples(query):
    groups = in_tuple_pattern.findall(query)
    typed_values = []
    for group in groups:
        if 'SELECT' in group[1].upper():
            continue
        tab_col = (None, rm_placeholder(group[0].upper()))
        vals = [x.strip().replace('"', '') for x in group[1].split(',')]
        for val in vals:
            typed_values.append((tab_col, val))
    return typed_values


def extract_typed_value_in_comparison_from_query(query: str) -> List[Tuple[Tuple[Union[str, None], str], str]]:
    query = re.sub(r'\byear\b', 'yearplaceholderrare', query, flags=re.IGNORECASE)
    query = re.sub(r'\bnumber\b', 'numberplaceholderrare', query, flags=re.IGNORECASE)
    query = re.sub(r'\blength\b', 'lengthplaceholderrare', query, flags=re.IGNORECASE)

    cmps = extract_all_comparison_from_query(query)
    typed_values = [(cmp['table_col'], cmp['val']) for cmp in cmps if 'table_col' in cmp]
    typed_values.extend(typed_values_in_tuples(query))
    for table, col, val1, val2 in re.findall('(?:([^\.\s]*)\.)?([^\.\s]+) (?:not )between ([^\s;]+) and ([^\s;]+)', query, re.IGNORECASE):
        if table == '':
            table = None
        else:
            table = table.lower()
        col = col.upper()
        for v in [val1, val2]:
            typed_values.append(((table, col), v))

    typed_values = [((t if t is None or not re.match(r'^T\d+$', t, flags=re.IGNORECASE) else None, c), v) for ((t, c), v) in typed_values]
    return typed_values


def process_str_value(v: str) -> str:
    if len(v) > 0 and v[0] in QUOTE_CHARS:
        v = v[1:]
    if len(v) > 0 and v[-1] in QUOTE_CHARS:
        v = v[:-1]
    for c in QUOTE_CHARS:
        v = v.replace(c + c, c)
    return v




AGG_OPS = ("none", "max", "min", "count", "sum", "avg")

def has_agg(unit):
    return unit[0] != AGG_OPS.index("none")


WHERE_OPS = (
    "not",
    "between",
    "=",
    ">",
    "<",
    ">=",
    "<=",
    "!=",
    "in",
    "like",
    "is",
    "exists",
)


def count_agg(units):
    return len([unit for unit in units if has_agg(unit)])


def count_component1(sql):
    count = 0
    if len(sql["where"]) > 0:
        count += 1
    if len(sql["groupBy"]) > 0:
        count += 1
    if len(sql["orderBy"]) > 0:
        count += 1
    if sql["limit"] is not None:
        count += 1
    if len(sql["from"]["table_units"]) > 0:  # JOIN
        count += len(sql["from"]["table_units"]) - 1

    ao = sql["from"]["conds"][1::2] + sql["where"][1::2] + sql["having"][1::2]
    count += len([token for token in ao if token == "or"])
    cond_units = sql["from"]["conds"][::2] + sql["where"][::2] + sql["having"][::2]
    count += len(
        [
            cond_unit
            for cond_unit in cond_units
            if cond_unit[1] == WHERE_OPS.index("like")
        ]
    )

    return count


def get_nestedSQL(sql):
    nested = []
    for cond_unit in sql["from"]["conds"][::2] + sql["where"][::2] + sql["having"][::2]:
        if type(cond_unit[3]) is dict:
            nested.append(cond_unit[3])
        if type(cond_unit[4]) is dict:
            nested.append(cond_unit[4])
    if sql["intersect"] is not None:
        nested.append(sql["intersect"])
    if sql["except"] is not None:
        nested.append(sql["except"])
    if sql["union"] is not None:
        nested.append(sql["union"])
    return nested

def count_component2(sql):
    nested = get_nestedSQL(sql)
    return len(nested)


def count_others(sql):
    count = 0
    # number of aggregation
    agg_count = count_agg(sql["select"][1])
    agg_count += count_agg(sql["where"][::2])
    agg_count += count_agg(sql["groupBy"])
    if len(sql["orderBy"]) > 0:
        agg_count += count_agg(
            [unit[1] for unit in sql["orderBy"][1] if unit[1]]
            + [unit[2] for unit in sql["orderBy"][1] if unit[2]]
        )
    agg_count += count_agg(sql["having"])
    if agg_count > 1:
        count += 1

    # number of select columns
    if len(sql["select"][1]) > 1:
        count += 1

    # number of where conditions
    if len(sql["where"]) > 1:
        count += 1

    # number of group by clauses
    if len(sql["groupBy"]) > 1:
        count += 1

    return count


def get_difficulty(sql):
    count_comp1_ = count_component1(sql)
    count_comp2_ = count_component2(sql)
    count_others_ = count_others(sql)

    if count_comp1_ <= 1 and count_others_ == 0 and count_comp2_ == 0:
        return "easy"
    elif (count_others_ <= 2 and count_comp1_ <= 1 and count_comp2_ == 0) or (
            count_comp1_ <= 2 and count_others_ < 2 and count_comp2_ == 0
    ):
        return "medium"
    elif (
            (count_others_ > 2 and count_comp1_ <= 2 and count_comp2_ == 0)
            or (2 < count_comp1_ <= 3 and count_others_ <= 2 and count_comp2_ == 0)
            or (count_comp1_ <= 1 and count_others_ == 0 and count_comp2_ <= 1)
    ):
        return "hard"
    else:
        return "extra"

    
def replace_agg(q):
    for agg_op in ['min', 'max', 'avg', 'count', 'sum']:
        pattern = re.compile(agg_op, re.IGNORECASE)
        q = pattern.sub("", q)
    return q




def perturb_limit(q):
    limit_subspan = re.search('LIMIT (\d+)', q, re.IGNORECASE)
    if limit_subspan is not None:
        orig_subspan_str = limit_subspan.group(0)
        orig_n = int(limit_subspan.group(1))
        if orig_n != 1:
            new_subspan_str = 'LIMIT ' + str(orig_n - 1)
            return q.replace(orig_subspan_str, new_subspan_str)
    

def get_natural_neighbors(q):
    result = set()
    for f in [replace_agg, perturb_limit, drop_where, replace_where]:
        neighbor = f(q)
        if neighbor is not None and neighbor != q:
            
            result.add(neighbor)
    return result

import sqlparse
import re

class RewriteException(Exception):
    pass
    

def recursive_rewrite(sql_query):
    sql_parse_ = sqlparse.parse(sql_query)
    if len(sql_parse_) == 0:
        return sql_query
    for tok in sql_parse_[0].tokens:
        if str(tok.ttype) == 'Token.Punctuation':
            inner = tok.value.strip()
            if 'SELECT' in tok.value.upper() and '(' == inner[0] and ')' == inner[-1]:
                inner_query = inner[1:-1]
                sql_query = sql_query.replace(inner_query, recursive_rewrite(inner_query))
    return rewrite_order_for_single_statement(sql_query)


def rewrite_order_for_single_statement(sql_query):
    sql_parse_ = sqlparse.parse(sql_query)
    if len(sql_parse_) == 0:
        return sql_query
    parsed_toks = sql_parse_[0].tokens
    
    order_by_tok_idx = None
    for i in range(len(parsed_toks)):
        tok = parsed_toks[i]
        if str(tok.ttype) == 'Token.Keyword' and tok.value.upper() == 'ORDER BY':
            order_by_tok_idx = i
            break
    
    if order_by_tok_idx is None:
        return sql_query
    
    limit_idx = None

    for i in range(order_by_tok_idx + 1, len(parsed_toks)):
        tok = parsed_toks[i]
        if str(tok.ttype) == 'Token.Keyword' and tok.value.upper() == 'LIMIT':
            limit_idx = i
            break
    if limit_idx is None:
        return sql_query
    
    limit_number = None
    for i in range(limit_idx + 1, len(parsed_toks)):
        tok = parsed_toks[i]
        if str(tok.ttype) != 'Token.Text.Whitespace':
            try:
                limit_number = int(tok.value)
            except ValueError:
                continue
            break
    if not limit_number:
        return sql_query
        #raise RewriteException('limit number not found in query %s' % sql_query)
    
    if limit_number != 1:
        return sql_query

    ranking_key = ''.join([str(parsed_toks[i]) for i in range(order_by_tok_idx + 1, limit_idx)])
    q3_agg = 'MAX' if 'DESC' in ranking_key.upper() else 'MIN'
    ranking_key = re.sub('ASC', '', ranking_key, flags=re.IGNORECASE)
    ranking_key = re.sub('DESC', '', ranking_key, flags=re.IGNORECASE)
    ranking_key = ranking_key.strip()
    
    without_order_query_toks = parsed_toks[:order_by_tok_idx]
    # print(ranking_key)
    select_idx = None
    for i in range(len(parsed_toks)):
        tok = parsed_toks[i]
        if str(tok.ttype) == 'Token.Keyword.DML' and tok.value.upper() == 'SELECT':
            select_idx = i
            break
    
    from_idx = None
    for i in range(select_idx + 1, len(parsed_toks)):
        tok = parsed_toks[i]
        if str(tok.ttype) == 'Token.Keyword' and tok.value.upper() == 'FROM':
            from_idx = i
            break
    
    if select_idx is None or from_idx is None:
        raise RewriteException('select or from not found in query %s' % sql_query)
    
    selection_string = ''.join([str(parsed_toks[i]) for i in range(select_idx + 1, from_idx)]).strip()
    distinct = selection_string[:8].upper() == 'DISTINCT'
    selection_string = re.sub('DISTINCT', '', selection_string, flags=re.IGNORECASE)
    selected_columns = [re.split(' as ', c.strip(), flags=re.IGNORECASE)[0].strip() for c in selection_string.split(',')]
    just_declared_names = []
    for c in selection_string.split(','):
        l = re.split('as', c.strip(), flags=re.IGNORECASE)
        if len(l) > 1:
            just_declared_names.append(l[1].strip())

    core_query_w_col_alias = ''.join([str(parsed_toks[i]) for i in range(from_idx)])
    mq_name = 'myquery_%d'
    selection_name = 'selected_col_%d'
    need_to_alias_ranking_key = True
    ranking_key_name = 'ranking_key'

    for n in just_declared_names:
        if n.lower() == ranking_key.lower():
            ranking_key_name = n
            need_to_alias_ranking_key = False
            break

    for i, c in enumerate(selected_columns):
        core_query_w_col_alias += ', ' + c + ' as {selection_name} '.format(selection_name = 'selected_col_%d' % i)
    if need_to_alias_ranking_key:
        core_query_w_col_alias += ', ' + ranking_key + ' as {ranking_key} '.format(ranking_key=ranking_key_name)
    core_query_w_col_alias += ''.join([str(parsed_toks[i]) for i in range(from_idx, order_by_tok_idx)])
    core_query_w_col_alias = re.sub('DISTINCT', '', core_query_w_col_alias, flags=re.IGNORECASE)
    
    q2 = '(%s) ' % core_query_w_col_alias + (mq_name % 2)
    q3 = '(select {agg}({q2_name}.{ranking_key}) as best_key from '.format(agg=q3_agg, q2_name=mq_name % 2, ranking_key=ranking_key_name) + q2 + ') ' + (mq_name % 3)
    
    returned_query = 'SELECT '
    if distinct:
        returned_query += 'DISTINCT '
    returned_query += ', '.join([(mq_name % 1) + '.' + (selection_name % i) for i in range(len(selected_columns))])
    returned_query += (' from (%s) ' % core_query_w_col_alias) + mq_name % 1
    returned_query += ', '
    returned_query += q3
    returned_query += ' where {q1_name}.{ranking_key} = {q3_name}.best_key'.format(ranking_key=ranking_key_name, q1_name=mq_name % 1, q3_name=mq_name % 3)
    return returned_query

def find_all(str_, sub_str, ignore_parents=None):
    if ignore_parents is None:
        ignore_parents = []
    idxs = []
    for i in range(len(str_)):
        found_parent = False
        for parent in ignore_parents:
            child_start_idx = parent.find(sub_str)
            if str_[(i-child_start_idx):(i+len(parent)-child_start_idx)] == parent:
                found_parent = True
                break
        if found_parent:
            continue
        if str_[i:(i+len(sub_str))]==sub_str:
            idxs.append(i)
    return idxs

def convert_join_recurse(sql_query, idxs, keywords):
    if len(idxs) == 0:
        return [sql_query]
    idx = idxs[0]
    new_queries = []
    for present_keyword in keywords:
        if sql_query[idx:(idx+len(present_keyword))] == present_keyword:
            for replace_keyword in keywords:
                new_q = sql_query[:idx]+replace_keyword+sql_query[(idx+len(present_keyword)):]
                new_idxs = [i-len(present_keyword)+len(replace_keyword) for i in idxs[1:]]
                new_queries += convert_join_recurse(new_q, new_idxs, keywords)
    return new_queries

def convert_join(sql_query):
    keywords = ['RIGHT JOIN', 'LEFT JOIN', 'JOIN']
    idxs = sum([find_all(sql_query, keywords[i], keywords[:i]) for i in range(len(keywords))], [])
    return convert_join_recurse(sql_query, idxs, keywords)
