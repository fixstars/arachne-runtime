#!/usr/bin/env bash

# overwrite uid and gid
usermod -u "$HOST_UID" developer
groupmod -g "$HOST_GID" developer

# keep some environments
{
    echo "export PYTHONPATH=${PYTHONPATH}"
    echo "export PYTHONIOENCODING=utf-8"
    echo "export TVM_LIBRARY_PATH=${TVM_LIBRARY_PATH}"
    # echo "export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
    echo "export PATH=${PATH}"
    echo "cd /workspaces/arachne-runtime"
} >> /home/developer/.bashrc

# change to the developer
chown developer:developer -R /home/developer
su - developer
