#!/usr/bin/python

import sys

def main():
  f1 = sys.argv[1];
  f2 = sys.argv[2];
  with open(f1, 'r') as fi1, open(f2, 'r') as fi2:
    lines1 = fi1.readlines()
    lines2 = fi2.readlines()
    for i in range(len(lines1)):
      num1 = lines1[i].split(", ")[1]
      num2 = lines2[i].split(", ")[1]
      print float(num1) - float(num2)

if __name__ == "__main__":
  main()

