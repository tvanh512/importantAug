exp_name=importantAugment
save_path=./saved_model
save_path_clf=$save_path/baseline/classifier
[[ ! -d $save_path/$exp_name ]] && mkdir $save_path/$exp_name
arch=Classifier_CNN
arch_gen=MaskGen_CNN2
noise_type=musan
num_workers=4
weight_loss_mask_freq_var=3
weight_loss_mask_time_var=3
weight_loss_classifier=1
weight_loss_mask_entropy=3
mask_option=3
target_SNR_dB=-12.5
python main.py --target_SNR_dB $target_SNR_dB\
               --mask_option $mask_option\
               --arch_gen $arch_gen\
               --save_path $save_path\
               --noise_type $noise_type\
               --weight_loss_mask_freq_var $weight_loss_mask_freq_var\
               --weight_loss_mask_time_var $weight_loss_mask_time_var\
               --weight_loss_classifier $weight_loss_classifier\
               --weight_loss_mask_entropy $weight_loss_mask_entropy\
               --arch $arch\
               --exp_name $exp_name\
               --save_path_clf $save_path_clf\
               --no_stage1 --no_stage3\
               --num_workers $num_workers\
               2>&1 | tee $save_path/$exp_name/log_train_generator.log 