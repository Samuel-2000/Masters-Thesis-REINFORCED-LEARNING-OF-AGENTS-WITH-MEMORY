Maze environment inspired by: https://github.com/michal-hradis/maze-rl



python run.py test --play --task-class complex --complexity-level 0.7 --n-doors 5 --n-buttons-per-door 4
python run.py test --model .\models\lstm\lstm_64b_0.0005lr_2026-01-14_08-33-02_dynamic_best.pt --dynamic --episodes 1
python run.py test --model models/lstm_64b_0.0005lr_2026-01-20_14-57-16_dynamic_best.pt --dynamic --episodes 1 --save-video



run.py (check parser.py for args)

run.py train
run.py test --model .\models\lstm_64b_0.0005lr_2025-12-10_14-59-38_best.pt

run.py train --batch-size 64 --network-type multimemory --epochs 10000
run.py test --model .\models\multimemory_64b_0.0005lr_2025-12-10_21-02-37_best.pt --save-video


# train and Test dynamic complexity in a short training run
python run.py train --network-type lstm --epochs 100 --batch-size 64 --dynamic-complexity --performance-window 20 --adjustment-interval 10 --save-dir test_dynamic







# Default: basic, complexity=0.0
python run.py test --model models/lstm_best.pt --save-video
# Video: lstm_best_ep_0.mp4

# Specific configuration
python run.py test --model models/lstm_best.pt --task-class doors --complexity-level 0.5 --save-video
# Video: lstm_best_doors_comp_0_50_ep_0.mp4

# Dynamic test (all stages × 5 complexities)
python run.py test --model models/lstm_best.pt --dynamic --save-video
# Videos: lstm_best_basic_comp_0_00_ep_0.mp4, lstm_best_doors_comp_0_25_ep_0.mp4, etc.

# Custom dynamic test
python run.py test --model models/lstm_best.pt --dynamic --stages basic doors --complexities 0.0 0.5 1.0 --save-video

# Test without video saving
python run.py test --model models/lstm_best.pt --dynamic --visualize

# Test with custom number of episodes
python run.py test --model models/lstm_best.pt --dynamic --episodes 5 --save-video



train --network-type lstm --epochs 10000 --batch-size 64 --dynamic-complexity --performance-window 20 --adjustment-interval 1 --save-dir test_dynamic
train --network-type lstm --epochs 10000 --batch-size 64 --task-class buttons --complexity-level 1.0 --n-doors 4








batch_train.py
test_benchmark.py