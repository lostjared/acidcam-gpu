#!/bin/bash

LOCATION=$1
VALUE=$2


cat $LOCATION/index.txt | awk -v val="$VALUE" '{ if(index($1, val) > 0) print $1, ":", NR+1; }'







