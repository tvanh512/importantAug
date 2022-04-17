save_path=./saved_model
save_path_clf=$save_path/baseline/classifier
arch=Classifier_CNN
noise_type=musan
num_workers=8
exp_name=nullImportantAug
[[ ! -d $save_path/$exp_name ]] && mkdir -p $save_path/$exp_name
network=1
target_SNR_dB=-12.5
# To test on GSC test set, GSC-Musan and GSC-QUT, use --stage3 --stage4 --stage5
python main_conventional_noise_aug.py --target_SNR_dB $target_SNR_dB\
                          --vanillaNoiseAug --network $network\
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
                          2>&1 | tee $save_path/$exp_name/log.log