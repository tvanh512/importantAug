save_path=./saved_model
save_path_clf=$save_path/baseline/classifier
arch=Classifier_CNN
noise_type=musan
num_workers=4
for target_SNR_dB in -10 -5 0 5 10 15 20 25 30 35 40
do
	echo $target_SNR_dB
    exp_name=conventional_noise_$target_SNR_dB
    [[ ! -d $save_path/$exp_name ]] && mkdir -p $save_path/$exp_name
	python main_conventional_noise_aug.py --vanillaNoiseAug\
                              --target_SNR_dB $target_SNR_dB\
                              --save_path $save_path\
                              --noise_type $noise_type\
                              --arch $arch\
                              --exp_name $exp_name\
                              --save_path_clf $save_path_clf\
                              --no_stage1\
                              --num_workers $num_workers\
                              2>&1 | tee $save_path/$exp_name/log.log
done