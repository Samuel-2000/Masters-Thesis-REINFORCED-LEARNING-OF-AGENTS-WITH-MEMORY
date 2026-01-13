Heavily Inspired by: https://github.com/michal-hradis/maze-rl



run.py (check parser.py for args)

run.py train
run.py test --model .\models\lstm_64b_0.0005lr_2025-12-10_14-59-38_best.pt

run.py train --batch-size 64 --network-type multimemory --epochs 10000
run.py test --model .\models\multimemory_64b_0.0005lr_2025-12-10_21-02-37_best.pt --save-video


# Test dynamic complexity in a short training run
python run.py train --network-type lstm --epochs 100 --batch-size 64 --dynamic-complexity --performance-window 20 --adjustment-interval 10 --save-dir test_dynamic


batch_train.py
test_benchmark.py