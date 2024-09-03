#!/bin/sh

./solver.py -e cube2x2 -m saves/cube2x2-paper-d200-2x2-d200-t1/best_3.2572e-02.dat --max-steps 30000 --cuda -o c2x2-paper-d200-t1.csv &
./solver.py -e cube2x2 -m saves/cube2x2-zero-goal-d200-2x2-d200-zg-t2/best_1.3816e-02.dat --max-steps 30000 --cuda -o c2x2-zero-goal-d200-t1.csv &
./solver.py -e cube3x3 --cuda --max-steps 30000 -m saves/cube3x3-paper-d200-3x3-paper-d200-t1/best_3.1818e-02.dat -o c3x3-paper-d200-t1.csv &
./solver.py -e cube3x3 --cuda --max-steps 30000 -m saves/cube3x3-zero-goal-d200-3x3-zg-d200-t1/best_2.0891e-02.dat -o c3x3-zero-goal-d200-t1.csv &
#./solver.py -e cube3x3 --cuda --max-steps 30000 -m saves/cube3x3-zero-goal-d200-no-decay/best_2.1798e-02.dat -o c3x3-zero-goal-d200-no-decay-v2.csv &
