exp_name=importantAugment
save_path=./saved_model
save_path_clf=$save_path/baseline/classifier
#mkdir $save_path/$exp_name
arch=Classifier_CNN
arch_gen=MaskGen_CNN2
noise_type=musan
num_workers=4
weight_loss_mask_freq_var=3
weight_loss_mask_time_var=3
weight_loss_classifier=1
weight_loss_mask_entropy=3
gen_aug_type=5
mask_shift=30
mask_option=3
target_SNR_dB=-12.5
log_name=stage3
# To test on GSC test set, GSC-Musan and GSC-QUT, use --stage4 --stage5 --stage6 
python main.py --target_SNR_dB $target_SNR_dB\
               --gen_augment\
               --gen_aug_type $gen_aug_type\
               --mask_shift $mask_shift\
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
               --no_stage1\
               --no_stage2\
               --num_workers $num_workers\
                2>&1 | tee $save_path/$exp_name/$log_name.log 