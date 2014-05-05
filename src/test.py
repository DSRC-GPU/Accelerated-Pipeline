#!/usr/bin/env python2.7

import sys

f1 = open(sys.argv[1])
f2 = open(sys.argv[2])

lines1 = f1.readlines()
lines2 = f2.readlines()

for i in range(len(lines1)):
    if not lines1[i].strip():
        print
        continue

    id1, x1, y1 = [float(x) for x in lines1[i].split()]
    id2, x2, y2 = [float(x) for x in lines2[i].split()]

    if id1 != id2:
        print("WTF");
        sys.exit(1)

    print(int(id1), abs(x1 - x2), abs(y1 - y2))


