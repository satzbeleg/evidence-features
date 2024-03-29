import logging
import cassandra as cas
import cassandra.cluster
import cassandra.query
import cassandra.policies
import gc
import numpy as np
from typing import List
import itertools
from .transform_all import to_int, i2f
from .transform_sbert import sbert_i2b
from .transform_kshingle import kshingle_to_int32
import uuid
from .todisk import is_valid_uuid


# start logger
logger = logging.getLogger(__name__)


cas_exec_profile = cas.cluster.ExecutionProfile(
    load_balancing_policy=cas.policies.RoundRobinPolicy(),
    request_timeout=None,  # disable query timeout
    # retry_policy=,
    # consistency_level=,
    # serial_consistency_level=,
    # row_factory=
)


class CqlConn:
    def __init__(self,
                 keyspace: str,
                 reset_tables: bool = False,
                 port: int = 9042,
                 contact_points=['0.0.0.0']
                 ):
        """ connect to Cassandra cluster """
        # connect to cluster
        self.cluster = cas.cluster.Cluster(
            contact_points=contact_points,
            port=port,
            protocol_version=5,
            idle_heartbeat_interval=0,
            reconnection_policy=cas.policies.ConstantReconnectionPolicy(
                1, None),
            execution_profiles={
                cas.cluster.EXEC_PROFILE_DEFAULT: cas_exec_profile},
        )
        # open an connection
        self.session = self.cluster.connect(
            wait_for_all_pools=False)
        # create keyspace and its tables IF NOT EXISTS
        _cas_init_tables(self.session, keyspace, reset_tables)
        # set `USE keyspace;`
        self.session.set_keyspace(keyspace)

    def get_session(self) -> cas.cluster.Session:
        return self.session

    def shutdown(self) -> None:
        self.session.shutdown()
        self.cluster.shutdown()
        gc.collect()
        pass


def _isvalid_keyspace_name(keyspace: str) -> bool:
    """ helper function for `_cas_init_tables` """
    try:
        if keyspace.islower():
            if keyspace.isalpha():
                return True
    except Exception:
        pass
    return False


def _cas_init_tables(session: cas.cluster.Session,
                     keyspace: str,
                     reset: bool = False) -> None:
    """ Initialize the CQL keyspace and tables for the new dataset.
        (only called in `CqlConn`)
    Parameters:
    -----------
    session : cas.cluster.Session
        A Cassandra Session object, i.e., an existing DB connection.
    keyspace : str
        The CQL KEYSPACE. Must be a string of lower case letters.
        Use the dataset name as keyspace.
    reset : bool (Default: False)
        Flag to delete and recreate the keyspace
    Notes:
    ------
    The `lemma` is used as partition key for `GROUP BY` and `WHERE` clauses,
    i.e., we can only query the whole data partion for a lemma.
    Parameters:
    -----------
    keyspace : str
        The CQL KEYSPACE. Must be a string of lower case letters.
        Use the dataset name as keyspace.
    reset : bool (Default False)
        Will drop keyspace in CQL
    """
    # check input arguments
    if not _isvalid_keyspace_name(keyspace):
        msg = (f"keyspace='{keyspace}' is not valid. "
               "Please use lower case letters")
        logger.error(msg)
        raise Exception(msg)

    # drop keyspace
    if reset:
        session.execute(f"DROP KEYSPACE IF EXISTS {keyspace};")

    # create a keyspace for the dataset
    session.execute(f"""
    CREATE KEYSPACE IF NOT EXISTS {keyspace}
    WITH REPLICATION = {{
        'class': 'SimpleStrategy',
        'replication_factor': 1
    }};
    """)

    # create tables if not exist
    session.execute(f"""
    CREATE TABLE IF NOT EXISTS {keyspace}.tbl_features (
      headword  TEXT
    , example_id UUID
    , sentence  TEXT
    , sent_id   UUID
    , spans    frozen<list<frozen<list<SMALLINT>>>>
    , annot    TEXT
    , biblio   TEXT
    , license  TEXT
    , score    FLOAT
    , feats1   frozen<list<TINYINT>>
    , feats2   frozen<list<TINYINT>>
    , feats3   frozen<list<TINYINT>>
    , feats4   frozen<list<TINYINT>>
    , feats5   frozen<list<SMALLINT>>
    , feats6   frozen<list<SMALLINT>>
    , feats7   frozen<list<SMALLINT>>
    , feats8   frozen<list<TINYINT>>
    , feats9   frozen<list<TINYINT>>
    , feats12  frozen<list<SMALLINT>>
    , feats13  frozen<list<TINYINT>>
    , feats14  frozen<list<TINYINT>>
    , hashes15  frozen<list<INT>>
    , hashes16  frozen<list<INT>>
    , hashes18  frozen<list<INT>>
    , PRIMARY KEY ((headword), sentence)
    );
    """)
    pass


# error logger
def handle_err(err, headword=None):
    logger.error(
        f"Insertion error for headword='{headword}'")


def insert_sentences(session: cas.cluster.Session,
                     sentences: List[str],
                     max_chars: int = 2048,
                     # num_partitions: int = 128,
                     sent_ids: List[str] = None,
                     biblio: List[str] = None,
                     licensetext: List[str] = None,
                     scores: List[float] = None,
                     document_level=False):
    # encode features
    # if sbert_making=True then `len(f1) = product(l17.shapes)`
    (
        f1, f2, f3, f4, f5, f6, f7, f8,
        f9, f12, f13, f14, h15, h16,
        sentences_sbd, l17, spans, annot
    ) = to_int(sentences, sbert_masking=True, document_level=document_level)
    if document_level:
        sentences = sentences_sbd

    # sent ids
    if sent_ids is None:
        sent_ids = [str(uuid.uuid4()) for _ in range(len(sentences))]
    else:
        # if sent_ids are not UUID strings, then hash strings as UUID
        sent_ids = [
            x if is_valid_uuid(x) else str(uuid.UUID(x))
            for x in sent_ids]

    # encode bibliographic information if exists
    if biblio is not None:
        if isinstance(biblio, (list, tuple)):
            h18 = kshingle_to_int32(biblio)
        elif isinstance(biblio, str):
            h18 = np.array(
                kshingle_to_int32([biblio])[0].tolist() * len(sentences)
            ).astype(np.int32)
            biblio = [biblio] * len(sentences)
    else:
        h18 = np.array([[0] * 32] * len(sentences)).astype(np.int32)
        biblio = [""] * len(sentences)

    # check license
    if isinstance(licensetext, str):
        licensetext = [licensetext] * len(sentences)
    elif licensetext is None:
        licensetext = [""] * len(sentences)

    # check scores
    if scores is None:
        scores = [0.5] * len(sentences)

    # get headwords
    headwords = list(set(itertools.chain(*l17)))
    if len(headwords) == 0:
        logger.warning("No headwords found")
        pass

    # prepare statement
    stmt = session.prepare(f"""
    INSERT INTO {session.keyspace}.tbl_features
    (
        headword, example_id, sentence, sent_id,
        spans, annot, biblio, license, score,
        feats1, feats2, feats3, feats4,
        feats5, feats6, feats7, feats8,
        feats9, feats12, feats13, feats14,
        hashes15, hashes16, hashes18
    )
    VALUES (
        ?, ?, ?, ?,
        ?, ?, ?, ?, ?,
        ?, ?, ?, ?,
        ?, ?, ?, ?,
        ?, ?, ?, ?,
        ?, ?, ?
    )
    IF NOT EXISTS;
    """)

    # prepare batch statement
    batches = {
        k: cas.query.BatchStatement(
            consistency_level=cas.query.ConsistencyLevel.ANY)
        for k in headwords
    }

    # loop over each sentence
    j_mask = 0  # index for `f1[j_mask]`
    for i, text in enumerate(sentences):
        # chop sentence length to `max_chars`
        text = text[:max_chars]
        # skip all sentences with less than 3 tokens
        if len(text.split(" ")) < 3:
            logger.warning(f"Sentence to short: '{text}'")
            j_mask += len(l17[i])
            continue
        # skip if no headword was found
        if len(l17[i]) == 0:
            logger.warning(f"Sentence has no VERB, NOUN, ADJ: '{text}'")
            j_mask += len(l17[i])
            continue

        # save a row for each headword
        for k, headword in enumerate(l17[i]):
            # add to batch
            batches[headword].add(stmt, [
                headword, uuid.uuid4(), text, uuid.UUID(sent_ids[i]),
                [spans[i][k]], annot[i], biblio[i], licensetext[i], scores[i],
                f1[j_mask],  # masked embeddings!
                f2[i], f3[i], f4[i],
                f5[i], f6[i], f7[i], f8[i],
                f9[i], f12[i], f13[i], f14[i],
                h15[i], h16[i], h18[i]
            ])
            j_mask += 1  # masked embeddings!

    # execute
    for headword, batch in batches.items():
        if len(batch) > 0:
            fb = session.execute_async(batch)
            fb.add_errback(handle_err, headword=headword)
    # done
    pass


def get_headwords(session: cas.cluster.Session,
                  max_fetch_size: int = 1000000):
    """ Lookup all unique lemmata from CQL database

    Parameters:
    -----------
    session : cas.cluster.Session
        A Cassandra Session object, i.e., an existing DB connection.
        The session must have a default keyspace, i.e., `USE keyspace;`
        or `session.set_keyspace(keyspace)`.

    max_fetch_size : int (Default: int(1e6))
        Maximum number of partitions to retrieve from cassandra

    Example:
    --------
    import evidence_features as evf
    import evidence_features.cql
    conn = evf.cql.CqlConn(keyspace="evidence")
    headwords = evf.cql.cas_get_headwords(
        conn.get_session(), max_fetch_size=5000)
    """
    stmt = cas.query.SimpleStatement(
        f"SELECT DISTINCT headword FROM {session.keyspace}.tbl_features",
        fetch_size=max_fetch_size)
    # fetch from CQL
    headwords = []
    for row in session.execute(stmt):
        headwords.append(row.headword)
    headwords = list(set(headwords))
    # clean up
    del stmt
    gc.collect()
    # done
    return headwords


def download_similarity_features(session: cas.cluster.Session,
                                 headword: str,
                                 max_fetch_size: int = 1000000):
    # prepare statement
    stmt = cas.query.SimpleStatement(f"""
        SELECT sentence, biblio, score,
               feats1, hashes15, hashes16, hashes18
        FROM {session.keyspace}.tbl_features
        WHERE headword='{headword}';
        """, fetch_size=max_fetch_size)
    # read fetched rows
    sentences = []
    biblio = []
    scores = []
    feats_semantic = []
    hashes_grammar = []
    hashes_duplicate = []
    hashes_biblio = []
    for row in session.execute(stmt):
        sentences.append(row.sentence)
        biblio.append(row.biblio)
        scores.append(row.score)
        feats_semantic.append(row.feats1)
        hashes_grammar.append(row.hashes15)
        hashes_duplicate.append(row.hashes16)
        hashes_biblio.append(row.hashes18)
    # convert and enforce data type
    feats_semantic = sbert_i2b(np.array(feats_semantic, dtype=np.int8))
    hashes_grammar = np.array(hashes_grammar, dtype=np.int32)
    hashes_duplicate = np.array(hashes_duplicate, dtype=np.int32)
    hashes_biblio = np.array(hashes_biblio, dtype=np.int32)
    # clean up
    del stmt
    gc.collect()
    # done
    return (
        sentences, biblio, scores,
        feats_semantic, hashes_grammar, hashes_duplicate, hashes_biblio
    )


def download_scoring_features(session: cas.cluster.Session,
                              headword: str,
                              max_fetch_size: int = 1000000):
    # prepare statement
    stmt = cas.query.SimpleStatement(f"""
        SELECT feats1, feats2, feats3, feats4,
               feats5, feats6, feats7, feats8,
               feats9, feats12, feats13, feats14, sentence
        FROM {session.keyspace}.tbl_features
        WHERE headword='{headword}';
        """, fetch_size=max_fetch_size)
    # read fetched rows
    sentences = []
    feats1, feats2, feats3, feats4 = [], [], [], []
    feats5, feats6, feats7, feats8 = [], [], [], []
    feats9, feats12, feats13, feats14 = [], [], [], []
    for row in session.execute(stmt):
        sentences.append(row.sentence)
        feats1.append(row.feats1)
        feats2.append(row.feats2)
        feats3.append(row.feats3)
        feats4.append(row.feats4)
        feats5.append(row.feats5)
        feats6.append(row.feats6)
        feats7.append(row.feats7)
        feats8.append(row.feats8)
        feats9.append(row.feats9)
        feats12.append(row.feats12)
        feats13.append(row.feats13)
        feats14.append(row.feats14)
    # convert to float
    feats = i2f(
        feats1, feats2, feats3, feats4,
        feats5, feats6, feats7, feats8,
        feats9, feats12, feats13, feats14
    ).astype(np.float32)
    # clean up
    del stmt, feats1, feats2, feats3, feats4
    del feats5, feats6, feats7, feats8
    del feats9, feats12, feats13, feats14
    gc.collect()
    # done
    return sentences, feats
