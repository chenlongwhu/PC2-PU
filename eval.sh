python main.py --log_dir log/PC2_PU_64 --transform_dim 64 --checkpoint_path model_best.pth.tar --test_dir data/test/gt_2048 --out_dir 基准上采样测试 --n_upsample 2 --phase test

python main.py --log_dir log/PC2_PU_64 --transform_dim 64 --checkpoint_path model_best.pth.tar --test_dir data/test/gt_2048_0.01 --out_dir 0.01噪声测试 --n_upsample 2 --phase test
python main.py --log_dir log/PC2_PU_64 --transform_dim 64 --checkpoint_path model_best.pth.tar --test_dir data/test/gt_2048_0.001 --out_dir 0.001噪声测试 --n_upsample 2 --phase test
python main.py --log_dir log/PC2_PU_64 --transform_dim 64 --checkpoint_path model_best.pth.tar --test_dir data/test/gt_2048_0.02 --out_dir 0.02噪声测试 --n_upsample 2 --phase test
python main.py --log_dir log/PC2_PU_64 --transform_dim 64 --checkpoint_path model_best.pth.tar --test_dir data/test/gt_2048_0.005 --out_dir 0.005噪声测试 --n_upsample 2 --phase test

python main.py --log_dir log/PC2_PU_64 --transform_dim 64 --checkpoint_path model_best.pth.tar --test_dir data/test/real_0.01_2048 --out_dir 0.01_blender噪声测试 --n_upsample 2 --gt_dir data/test/gt_realscan_32768 --phase test
python main.py --log_dir log/PC2_PU_64 --transform_dim 64 --checkpoint_path model_best.pth.tar --test_dir data/test/real_0.02_2048 --out_dir 0.02_blender噪声测试 --n_upsample 2 --gt_dir data/test/gt_realscan_32768 --phase test
python main.py --log_dir log/PC2_PU_64 --transform_dim 64 --checkpoint_path model_best.pth.tar --test_dir data/test/real_0.05_2048 --out_dir 0.05_blender噪声测试 --n_upsample 2 --gt_dir data/test/gt_realscan_32768 --phase test
python main.py --log_dir log/PC2_PU_64 --transform_dim 64 --checkpoint_path model_best.pth.tar --test_dir data/test/real_0.005_2048 --out_dir 0.005_blende噪声测试 --n_upsample 2 --gt_dir data/test/gt_realscan_32768 --phase test

python eval.py --gt data/test/gt_8192 --out_folder log/PC2_PU_64/评估结果 --pred log/PC2_PU_64/基准上采样测试 log/PC2_PU_64/0.01噪声测试 log/PC2_PU_64/0.001噪声测试 log/PC2_PU_64/0.02噪声测试 log/PC2_PU_64/0.005噪声测试
python eval.py --gt data/test/gt_realscan_8192 --out_folder log/PC2_PU_64/评估结果 --pred log/PC2_PU_64/0.01_blender噪声测试 log/PC2_PU_64/0.02_blender噪声测试 log/PC2_PU_64/0.05_blender噪声测试 log/PC2_PU_64/0.005_blender噪声测试
python eval.py --gt data/test/gt_32768 --out_folder log/PC2_PU_64/评估结果 --pred log/PC2_PU_64/基准上采样测试_up_2 log/PC2_PU_64/0.01噪声测试_up_2 log/PC2_PU_64/0.001噪声测试_up_2 log/PC2_PU_64/0.02噪声测试_up_2 log/PC2_PU_64/0.005噪声测试_up_2
python eval.py --gt data/test/gt_realscan_32768 --out_folder log/PC2_PU_64/评估结果 --pred log/PC2_PU_64/0.01_blender噪声测试_up_2 log/PC2_PU_64/0.02_blender噪声测试_up_2 log/PC2_PU_64/0.05_blender噪声测试_up_2 log/PC2_PU_64/0.005_blender噪声测试_up_2