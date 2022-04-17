# To replicate the number reported in the paper
exp_name=baseline
save_path=./pretrained_model
arch=Classifier_CNN
epoch=1
python main.py --arch $arch\
               --exp_name $exp_name\
               --save_path $save_path\
               --no_stage1\
               --no_stage2\
               --no_stage3\
               --stage1_test\
               --stage1_test_musan\
               --stage1_test_qut\
               2>&1 | tee $save_path/$exp_name/log_test.log 