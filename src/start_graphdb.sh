cd ~/data/sbs-graphs/graphdb-10.6.3/bin/

# run graphdb as daemon (in background) and server-only (no UI)
bash graphdb -d -s


# to install
#https://graphdb.ontotext.com/documentation/10.6/graphdb-standalone-server.html#configuring-graphdb
#https://graphdb.ontotext.com/documentation/9.8/enterprise/run-stand-alone-server.html

# to attach remote UI to local
#https://graphdb.ontotext.com/documentation/10.1/connecting-to-remote-graphdb-instance.html#

# to create a repository
#https://graphdb.ontotext.com/documentation/10.6/creating-a-repository.html#create-a-repository-in-a-remote-location

# if running and needs to be killed
#kill $(lsof -i:7200)
