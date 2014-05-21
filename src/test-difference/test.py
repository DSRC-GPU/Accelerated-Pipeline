#!/usr/bin/env python2.7

import sys

f1 = open(sys.argv[1])
f2 = open(sys.argv[2])
nodes = int(sys.argv[3])

lines1 = f1.readlines()
lines2 = f2.readlines()

differences = []

for i in range(len(lines1)):
  if not lines1[i].strip():
    continue
  id1, x1, y1 = [float(x) for x in lines1[i].split()]
  id2, x2, y2 = [float(x) for x in lines2[i].split()]
  differences.append(abs(x1 - x2) + abs(y1 - y2))

it = 0
total = 0.0
for value in differences:
  total += value
  if it % nodes == nodes - 1:
    print total
    total = 0
  it += 1

