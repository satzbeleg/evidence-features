import logging
import cassandra as cas
import cassandra.cluster
import cassandra.query
import cassandra.policies
import gc
import numpy as np

# start logger
logger = logging.getLogger(__name__)


class CqlConn:
    def __init__(self,
                 keyspace: str,
                 reset_tables: bool = False):
        """ connect to Cassandra cluster """
        # connect to cluster
        self.cluster = cas.cluster.Cluster(
            contact_points=['0.0.0.0'],
            port=9042,
            protocol_version=5,
            idle_heartbeat_interval=0,
            load_balancing_policy=cas.policies.RoundRobinPolicy(),
            reconnection_policy=cas.policies.ConstantReconnectionPolicy(
                1, None),
            # default_retry_policy=KeepTryingRetryPolicy(),
            # conviction_policy_factory=NeverConvictionPolicy  # doesn't work
        )
        # open an connection
        self.session = self.cluster.connect(
            wait_for_all_pools=False)
        # disable query timeout (` ResponseFuture.result()`, `cas.ReadTimeout`)
        self.session.default_timeout = None
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
    CREATE TABLE IF NOT EXISTS {keyspace}.dataset (
      lemma     TEXT
    , sentence  TEXT
    , feats1   frozen<list<TINYINT>>
    , feats2   frozen<list<TINYINT>>
    , feats3   frozen<list<TINYINT>>
    , feats4   frozen<list<TINYINT>>
    , feats5   frozen<list<TINYINT>>
    , feats6   frozen<list<TINYINT>>
    , feats7   frozen<list<TINYINT>>
    , feats8   frozen<list<TINYINT>>
    , feats9   frozen<list<TINYINT>>
    , feats12  frozen<list<TINYINT>>
    , feats13  frozen<list<TINYINT>>
    , feats14  frozen<list<TINYINT>>
    , PRIMARY KEY (lemma, sentence)
    );
    """)
    pass


def _cas_get_lemmata(session: cas.cluster.Session,
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
    import evidence_model as ev
    import evidence_model.dataset
    conn = ev.dataset.CqlConn(keyspace="toydata")
    lemmata = ev.datatset._cas_get_lemmata(
        conn.get_session(), max_fetch_size=5000)
    """
    stmt = cas.query.SimpleStatement(
        "SELECT DISTINCT lemma FROM dataset",
        fetch_size=max_fetch_size)
    # fetch from CQL
    ulemmata = []
    for row in session.execute(stmt):
        ulemmata.append(row.lemma)
    ulemmata = list(set(ulemmata))
    # clean up
    del stmt
    gc.collect()
    # done
    return ulemmata


def _cas_download_partitions(session: cas.cluster.Session,
                             lemmata: List[str],
                             max_lemmata: int = 100,
                             row_limit: int = 5000):
    """ Sample lemmata and download their CQL paritions into RAM

    Parameters:
    -----------
    session : cas.cluster.Session
        A Cassandra Session object, i.e., an existing DB connection.
        The session must have a default keyspace, i.e., `USE keyspace;`
        or `session.set_keyspace(keyspace)`.

    lemmata : List[str]
        List of available lemmata

    max_lemmata : int (Default: 100)
        The maximum number lemmata to use. It's also the max. number of
          CQL partitions to access.
        (ignored if set to max_lemmata='all')

    row_limit : int (Default: int(5e5))
        Maximum number of examples/rows to fetch for each partition.

    Example:
    --------
    import evidence_model as ev
    import evidence_model.dataset
    conn = ev.dataset.CqlConn(keyspace="demodata")
    lemmata = ev.dataset._cas_get_lemmata(
        conn.get_session(), max_fetch_size=5000)
    sampled_partions = ev.dataset._cas_download_partitions(
        conn.get_session(), lemmata, max_lemmata=60)
    assert len(sampled_partions) == min(len(lemmata), 60)
    """
    # prepare statement
    stmt = session.prepare(f"""
    SELECT sentence, feats1, feats2, feats3, feats4, feats5, feats6, feats9,
          , feats12, feats13, feats14
    FROM dataset WHERE lemma=?
    LIMIT {row_limit} ; """)

    # sample lemmata
    if max_lemmata == 'all':
        sampled_lemmata = lemmata.copy()
    else:
        sampled_lemmata = np.random.choice(
            lemmata, size=min(len(lemmata), max_lemmata),
            replace=False).tolist()

    # loop over responses
    sampled_partions = []
    err_counter = 0
    while err_counter < 5:
        if len(sampled_lemmata) == 0:
            break
        lemma = sampled_lemmata[0]
        sampled_lemmata.remove(lemma)
        try:
            # fetch lemma partition
            dat = session.execute(stmt, [lemma])
            # read data and convert directly to float-point representation
            # bucket, hashvalue, augm = [], [], []
            # for row in dat:
            #     if row.bucket is None:
            #         continue
            #     if not isinstance(row.bucket, int):
            #         continue
            #     if row.hashvalue is None:
            #         continue
            #     num_enc8 = len(row.hashvalue)
            #     if num_enc8 == 0:
            #         continue
            #     if row.augmentations is None:
            #         continue
            #     if len(row.augmentations) == 0:
            #         continue
            #     tmp = [a for a in row.augmentations if len(a) == num_enc8]
            #     if len(tmp) == 0:
            #         continue
            #     # cast and save
            #     bucket.append(int(row.bucket))
            #     hashvalue.append(np.array(row.hashvalue).astype(np.int8))
            #     augm.append(np.array(tmp).astype(np.int8))
            # append to variable in RAM
            if (len(bucket) > 0) and (len(hashvalue) > 0) and (len(augm) > 0):
                sampled_partions.append((bucket, hashvalue, augm))
            err_counter = 0
            del dat, bucket, hashvalue, augm
            gc.collect()
        except cas.ReadTimeout as e:
            logger.error(f"Read Timeout problems with '{lemma}': {e}")
            err_counter += 1
            time.sleep(2.0)
        except Exception as e:
            logger.error(f"Unknown problems with '{lemma}': {e}")
            err_counter += 1
            time.sleep(3.0)
    # clean up
    del stmt, sampled_lemmata
    gc.collect()
    # done
    return sampled_partions

