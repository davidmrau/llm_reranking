#!/bin/bash
USER=`accinfo | grep NRC | awk '{print $3}'`
#Loading modules
echo $USER

export PYTHONIOENCODING='utf-8'
cd ..
nvidia-smi
FILE_NAME=$1
export FILE_NAME
sleep 2 
source slurm/${1}


