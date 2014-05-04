#!/bin/bash
i=1000
while [ $i -lt 1500 ]
do
  echo "Seed: $i"
  python pacman.py -T ExampleTeam -p CollectAgent -m 1000 -t 0.0000001 -s $i -q >> /dev/null
i=$((i+1))
done