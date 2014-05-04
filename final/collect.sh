#!/bin/bash
i=2
while [ $i -lt 200 ]
do
  echo "Seed: $i"
  python pacman.py -T ExampleTeam -p CollectAgent -m 1000 -t 0.0000001 -s $i
done