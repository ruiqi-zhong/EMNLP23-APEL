from collections import OrderedDict

path2dep = {}

advising_dep_dict = OrderedDict([
    (('course_offering', 'COURSE_ID'), ('course', 'COURSE_ID')),
    (('course_offering', 'SEMESTER'), ('semester', 'SEMESTER_ID')),
    (('offering_instructor', 'INSTRUCTOR_ID'), ('instructor', 'INSTRUCTOR_ID')),
    (('offering_instructor', 'OFFERING_ID'), ('course_offering', 'OFFERING_ID')),
    (('student_record', 'COURSE_ID'), ('course', 'COURSE_ID')),
    (('student_record', 'STUDENT_ID'), ('student', 'STUDENT_ID')),
    (('program_requirement', 'PROGRAM_ID'), ('program', 'PROGRAM_ID')),
    (('program_course', 'COURSE_ID'), ('course', 'COURSE_ID')),
    (('program_course', 'PROGRAM_ID'), ('program', 'PROGRAM_ID')),
    (('area', 'COURSE_ID'), ('course_offering', 'COURSE_ID')),
    (('student_record', 'SEMESTER'), ('semester', 'SEMESTER_ID')),
    (('course_prerequisite', 'PRE_COURSE_ID'), ('course', 'COURSE_ID')),
    (('course_prerequisite', 'COURSE_ID'), ('course', 'COURSE_ID')),
    (('gsi', 'COURSE_OFFERING_ID'), ('course_offering', 'OFFERING_ID')),
    (('gsi', 'STUDENT_ID'), ('student', 'STUDENT_ID')),
])

path2dep['database/advising/advising.sqlite'] = advising_dep_dict

scholar_dep_dict = OrderedDict([
    (('paper', 'VENUEID'), ('venue', 'VENUEID')),
    (('paper', 'JOURNALID'), ('journal', 'JOURNALID')),
    (('cite', 'CITINGPAPERID'), ('paper', 'PAPERID')),
    (('cite', 'CITEDPAPERID'), ('paper', 'PAPERID')),
    (('paperkeyphrase', 'KEYPHRASEID'), ('keyphrase', 'KEYPHRASEID')),
    (('paperkeyphrase', 'PAPERID'), ('paper', 'PAPERID')),
    (('writes', 'AUTHORID'), ('author', 'AUTHORID')),
    (('writes', 'PAPERID'), ('paper', 'PAPERID')),
    (('paperdataset', 'DATASETID'), ('dataset', 'DATASETID')),
    (('paperdataset', 'PAPERID'), ('paper', 'PAPERID')),
    (('paperfield', 'FIELDID'), ('field', 'FIELDID')),
    (('paper', 'JOURNALID'), ('journal', 'JOURNALID')),
    (('paperfield', 'PAPERID'), ('paper', 'PAPERID'))
])

path2dep['database/scholar/scholar.sqlite'] = scholar_dep_dict

imdb_dep_dict = OrderedDict([
    (('cast', 'MSID'), ('copyright', 'MSID')),
    (('cast', 'AID'), ('actor', 'AID')),
    (('classification', 'MSID'), ('copyright', 'MSID')),
    (('classification', 'GID'), ('genre', 'GID')),
    (('directed_by', 'DID'), ('director', 'DID')),
    (('directed_by', 'MSID'), ('copyright', 'MSID')),
    (('made_by', 'PID'), ('producer', 'PID')),
    (('made_by', 'MSID'), ('copyright', 'MSID')),
    (('tags', 'KID'), ('keyword', 'ID')),
    (('tags', 'MSID'), ('copyright', 'MSID')),
    (('written_by', 'WID'), ('writer', 'WID')),
    (('written_by', 'MSID'), ('copyright', 'MSID')),
    (('movie', 'MID'), ('copyright', 'MSID')),
    (('copyright', 'CID'), ('company', 'ID')),
    (('tv_series', 'SID'), ('copyright', 'MSID'))
])

path2dep['database/imdb/imdb.sqlite'] = imdb_dep_dict

academic_dep_dict = OrderedDict([
    (('domain_author', 'DID'), ('domain', 'DID')),
     (('domain_author', 'AID'), ('author', 'AID')),
     (('domain_conference', 'DID'), ('domain', 'DID')),
     (('domain_conference', 'CID'), ('conference', 'CID')),
     (('domain_journal', 'DID'), ('domain', 'DID')),
     (('domain_journal', 'JID'), ('journal', 'JID')),
     (('domain_keyword', 'DID'), ('domain', 'DID')),
     (('domain_keyword', 'KID'), ('keyword', 'KID')),
     (('publication', 'CID'), ('conference', 'CID')),
     (('publication', 'JID'), ('journal', 'JID')),
     (('domain_publication', 'DID'), ('domain', 'DID')),
     (('domain_publication', 'PID'), ('publication', 'PID')),
     (('publication_keyword', 'KID'), ('keyword', 'KID')),
     (('publication_keyword', 'PID'), ('publication', 'PID')),
     (('writes', 'AID'), ('author', 'AID')),
     (('writes', 'PID'), ('publication', 'PID')),
     (('cite', 'CITING'), ('publication', 'PID')),
     (('cite', 'CITED'), ('publication', 'PID')),
    (('author', 'OID'), ('organization', 'OID'))
])

path2dep['database/academic/academic.sqlite'] = academic_dep_dict

atis_dep_dict = OrderedDict([
    (('flight', 'FROM_AIRPORT'), ('airport', 'AIRPORT_CODE')),
    (('flight', 'TO_AIRPORT'), ('airport', 'AIRPORT_CODE')),
    (('city', 'STATE_CODE'), ('state', 'STATE_CODE')),
    (('flight', 'FLIGHT_DAYS'), ('days', 'DAYS_CODE')),
    (('restriction', 'RESTRICTION_CODE'), ('fare', 'RESTRICTION_CODE')),
    (('flight', 'AIRLINE_CODE'), ('airline', 'AIRLINE_CODE'))
])

path2dep['database/atis/atis.sqlite'] = atis_dep_dict
