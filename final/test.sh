#!/bin/bash
i=2
while [ $i -lt 200 ]
do
  echo "Seed: $i"
  python pacman.py -T CoequalizerTeam -d -q -m 100 -s $i > /dev/null
  i=$((i+1))
done