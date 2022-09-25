

## Start Cassandra Database
The preprocessed hashed embeddings incl. augmentations are stored in a Cassandra database.

```
docker-compose -f cassandra-container.yml up --build -d
```

Check if everything is running
```sh
# show all docker containers
docker ps -a
# show container logs
docker-compose -f cassandra-container.yml logs -t
```


## Init table and and insert demo data

```sh
cd ..
source .venv/bin/activate
python
```

```py
import evidence_features as evf
import evidence_features.cql

conn = evf.cql.CqlConn(keyspace="evidence", port=9042, contact_points=['0.0.0.0'])
# conn = evf.cql.CqlConn(keyspace="evidence", reset_tables=True, port=9042, contact_points=['0.0.0.0'])
session = conn.get_session()

sentences = [
    "Dieser Satz ist ein Beispiel, aber eher kurz.",
    "Die Kuh macht muh, der Hund wufft aber lauter.",
    "Eine Herde Kühe wird von Beispielen gehütet."
]
# encode and save sentences
evf.cql.insert_sentences(session, sentences)

# lookup all headwords in the db
headwords = evf.cql.get_headwords(session)

# download partition for variations
(
    sentences2, feats_semantic, feats_grammar, feats_duplicate
) = evf.cql.download_similarity_features(session, headword="Beispiel")
# np.mean(feats_duplicate[0] == feats_duplicate[1])

# download partition for scoring, e.g. convert to float
sentences3, feats = evf.cql.download_scoring_features(
    session, headword="Beispiel")

# close db connection
conn.shutdown()
```

## Access the database with cqlsh
```sh
# access the container
docker exec -it evidence-features /bin/sh
# open cqlsh
cqlsh -u cassandra -p cassandra
```

## Access the container
```sh
# access the container
docker exec -it evidence-features /bin/bash
# show table stats
nodetool tablestats evidence.tbl_features
# when done
cqlsh -u cassandra -p cassandra
SELECT COUNT(1) FROM evidence.tbl_features;
```


## Backup the database to CSV file
```sh
cd ./cql
# access the container
docker exec -it evidence-features /bin/bash
# open cqlsh
cqlsh -u cassandra -p cassandra
# export to table to CSV file
COPY evidence.tbl_features to '/export/ev_feats.csv';
# on the host machine
cd data/cassandra-export
gzip -9 -k ev_feats.csv
```

