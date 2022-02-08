#!/bin/bash
log_dir=$1
python main.py --num_point 1024 --use_single_patch True --log_dir $log_dir --transform_dim $2 --test_dir data/test/gt_2048 --out_dir 基准上采样测试 --n_upsample 2 --phase test --up_module $3 --K1 $4 --K2 $5 --checkpoint_path $6

python main.py --num_point 1024 --use_single_patch True --log_dir $log_dir --transform_dim $2 --test_dir data/test/gt_2048_0.01 --out_dir 0.01噪声测试 --n_upsample 2 --phase test --up_module $3 --K1 $4 --K2 $5 --checkpoint_path $6
python main.py --num_point 1024 --use_single_patch True --log_dir $log_dir --transform_dim $2 --test_dir data/test/gt_2048_0.001 --out_dir 0.001噪声测试 --n_upsample 2 --phase test --up_module $3 --K1 $4 --K2 $5 --checkpoint_path $6
python main.py --num_point 1024 --use_single_patch True --log_dir $log_dir --transform_dim $2 --test_dir data/test/gt_2048_0.02 --out_dir 0.02噪声测试 --n_upsample 2 --phase test --up_module $3 --K1 $4 --K2 $5 --checkpoint_path $6
python main.py --num_point 1024 --use_single_patch True --log_dir $log_dir --transform_dim $2 --test_dir data/test/gt_2048_0.005 --out_dir 0.005噪声测试 --n_upsample 2 --phase test --up_module $3 --K1 $4 --K2 $5 --checkpoint_path $6

python main.py --num_point 1024 --use_single_patch True --log_dir $log_dir --transform_dim $2 --test_dir data/test/real_0.01_2048 --out_dir 0.01_blender噪声测试 --n_upsample 2 --gt_dir data/test/gt_realscan_32768 --phase test --up_module $3 --K1 $4 --K2 $5 --checkpoint_path $6
python main.py --num_point 1024 --use_single_patch True --log_dir $log_dir --transform_dim $2 --test_dir data/test/real_0.02_2048 --out_dir 0.02_blender噪声测试 --n_upsample 2 --gt_dir data/test/gt_realscan_32768 --phase test --up_module $3 --K1 $4 --K2 $5 --checkpoint_path $6
python main.py --num_point 1024 --use_single_patch True --log_dir $log_dir --transform_dim $2 --test_dir data/test/real_0.05_2048 --out_dir 0.05_blender噪声测试 --n_upsample 2 --gt_dir data/test/gt_realscan_32768 --phase test --up_module $3 --K1 $4 --K2 $5 --checkpoint_path $6
python main.py --num_point 1024 --use_single_patch True --log_dir $log_dir --transform_dim $2 --test_dir data/test/real_0.005_2048 --out_dir 0.005_blender噪声测试 --n_upsample 2 --gt_dir data/test/gt_realscan_32768 --phase test --up_module $3 --K1 $4 --K2 $5 --checkpoint_path $6

python main.py --num_point 1024 --use_single_patch True --log_dir $log_dir --transform_dim $2 --test_dir data/test/Outliers_5_2048 --out_dir 5离群点测试 --n_upsample 2 --phase test --up_module $3 --K1 $4 --K2 $5 --checkpoint_path $6
python main.py --num_point 1024 --use_single_patch True --log_dir $log_dir --transform_dim $2 --test_dir data/test/Outliers_10_2048 --out_dir 10离群点测试 --n_upsample 2  --phase test --up_module $3 --K1 $4 --K2 $5 --checkpoint_path $6
python main.py --num_point 1024 --use_single_patch True --log_dir $log_dir --transform_dim $2 --test_dir data/test/Outliers_20_2048 --out_dir 20离群点测试 --n_upsample 2  --phase test --up_module $3 --K1 $4 --K2 $5 --checkpoint_path $6

python main.py --num_point 1024 --use_single_patch True --log_dir $log_dir --transform_dim $2 --test_dir data/test/unseen_2048 --out_dir 泛化性测试  --phase test --gt_dir data/test/unseen_8192 --up_module $3 --K1 $4 --K2 $5 --checkpoint_path $6

python eval.py --gt data/test/gt_8192 --out_folder $log_dir/评估结果 --pred $log_dir/基准上采样测试 $log_dir/0.01噪声测试 $log_dir/0.001噪声测试 $log_dir/0.02噪声测试 $log_dir/0.005噪声测试 $log_dir/5离群点测试 $log_dir/10离群点测试 $log_dir/20离群点测试
python eval.py --gt data/test/gt_realscan_8192 --out_folder $log_dir/评估结果 --pred $log_dir/0.01_blender噪声测试 $log_dir/0.02_blender噪声测试 $log_dir/0.05_blender噪声测试 $log_dir/0.005_blender噪声测试
python eval.py --gt data/test/gt_32768 --out_folder $log_dir/评估结果 --pred $log_dir/基准上采样测试_up_2 $log_dir/0.01噪声测试_up_2 $log_dir/0.001噪声测试_up_2 $log_dir/0.02噪声测试_up_2 $log_dir/0.005噪声测试_up_2 $log_dir/5离群点测试_up_2 $log_dir/10离群点测试_up_2 $log_dir/20离群点测试_up_2
python eval.py --gt data/test/gt_realscan_32768 --out_folder $log_dir/评估结果 --pred $log_dir/0.01_blender噪声测试_up_2 $log_dir/0.02_blender噪声测试_up_2 $log_dir/0.05_blender噪声测试_up_2 $log_dir/0.005_blender噪声测试_up_2
python eval.py --gt data/test/unseen_8192 --out_folder $log_dir/评估结果 --pred $log_dir/泛化性测试