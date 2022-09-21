

## Start Cassandra Database
The preprocessed hashed embeddings incl. augmentations are stored in a Cassandra database.

```
docker-compose -f cassandra-container.yml up --build -d
```

Check if everything is running
```sh
# show all docker containers
docker ps -a
# does the database exists on the host machine
ls data/cassandra
# show container logs
docker-compose -f cassandra-container.yml logs -t
```

## Access the database with cqlsh
```sh
# access the container
docker exec -it evidence-features /bin/sh
# open cqlsh
cqlsh -u cassandra -p cassandra
```


```sh
# access the container
docker exec -it evidence-features /bin/bash
# show table stats
nodetool tablestats dwdsdataset.dataset
# when done
cqlsh -u cassandra -p cassandra
SELECT COUNT(1) FROM dwdsdataset.dataset;
```


## Backup the database to CSV file
```sh
# access the container
docker exec -it evidence-features /bin/bash
# open cqlsh
cqlsh -u cassandra -p cassandra
# export to table to CSV file
COPY dwdsdataset.dataset to '/export/dwdsdataset.csv';
# on the host machine
cd data/cassandra-export
gzip -9 -k dwdsdataset.csv
```

