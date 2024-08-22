#!/usr/bin/env bash

./play.py --cuda -r 10 saves/t1/best_088_39300.dat saves/t1/best_025_09900.dat saves/t1/best_022_08200.dat \
  saves/t1/best_021_08100.dat saves/t1/best_009_03400.dat saves/t1/best_014_04700.dat saves/t1/best_008_02700.dat \
  saves/t1/best_010_03500.dat saves/t1/best_029_11800.dat saves/t1/best_007_02300.dat \
  saves/t2/best_069_41500.dat saves/t2/best_070_42200.dat saves/t2/best_066_38900.dat saves/t2/best_071_42600.dat \
  saves/t2/best_059_33700.dat saves/t2/best_049_27500.dat saves/t2/best_068_41300.dat saves/t2/best_048_26700.dat \
  saves/t2/best_058_32100.dat saves/t2/best_076_45200.dat > final.txt
