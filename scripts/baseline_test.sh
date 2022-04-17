exp_name=baseline
save_path=./saved_model
[[ ! -d $save_path/$exp_name ]] && mkdir -p $save_path/$exp_name
arch=Classifier_CNN
# To evaluate on  GSC test set, use --stage1_test.
# Similarly, to evaluate on GSC-Musan and GSC-QUT test set
# use --stage1_test_musan and --stage1_test_qut, respectively. 
python main.py --arch $arch\
               --exp_name $exp_name\
               --no_stage1\
               --no_stage2\
               --no_stage3\
               --stage1_test\
               --stage1_test_musan\
               --stage1_test_qut\
               2>&1 | tee $save_path/$exp_name/log_test.log 