save_path=./saved_model
save_path_clf=$save_path/baseline/classifier
arch=Classifier_CNN
noise_type=musan
num_workers=4
target_SNR_dB=15
echo $target_SNR_dB
exp_name=conventional_noise_$target_SNR_dB
log_name=log_test
python main_conventional_noise_aug.py --vanillaNoiseAug\
                          --target_SNR_dB $target_SNR_dB\
                          --save_path $save_path\
                          --noise_type $noise_type\
                          --arch $arch\
                          --exp_name $exp_name\
                          --save_path_clf $save_path_clf\
                          --no_stage1\
                          --no_stage2\
                          --stage3\
                          --stage4\
                          --stage5\
                          --num_workers $num_workers\
                          2>&1 | tee $save_path/$exp_name/$log_name.log