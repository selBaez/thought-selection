# Server instructions

## GraphDB

### Resources to set up

* To install, you can
  follow [this](https://graphdb.ontotext.com/documentation/10.6/graphdb-standalone-server.html#configuring-graphdb)
  and [this](https://graphdb.ontotext.com/documentation/9.8/enterprise/run-stand-alone-server.html)

* To attach remote UI to local,
  look [here](https://graphdb.ontotext.com/documentation/10.1/connecting-to-remote-graphdb-instance.html#)

* To create a repository,
  look [here](https://graphdb.ontotext.com/documentation/10.6/creating-a-repository.html#create-a-repository-in-a-remote-location)

### Running it on a server

To run graphdb as daemon (in background) and server-only (no UI)

``` bash
cd ~{INSTALLATION_PATH}/graphdb-10.6.3/bin/
bash graphdb -d -s
```

If there is an instance running, and it needs to be killed

``` bash
kill $(lsof -i:7200)
```

## Running experiments

You can use the provided bash file `experiment1.sh`. To run it on the background you can do

```
ssh {USERNAME}@{SERVER_IP}
cd {PATH_TO_REPOSITORY}/thought-selection/running_scripts
conda activate thought-selection
nohup source experiment1_server.sh > experiment1_server.log 2>&1 &
```

or 
``` bash
ssh {USERNAME}@{SERVER_IP} "cd {PATH_TO_REPOSITORY}/thought-selection/running_scripts && source activate thought-selection && nohup bash experiment1_server.sh > experiment1_server.log 2>&1 &"
```

Remember to:

* Create a conda virtual environment first
* Transfer the required files (user models)
* Adapt the paths to your installation 