save_path=./pretrained_model
save_path_clf=$save_path/baseline/classifier
save_path_clf_frozen=$save_path/importantAugment/clf_frozen
arch=Classifier_CNN
arch_gen=MaskGen_CNN2
noise_type=musan
num_workers=8
weight_loss_mask_freq_var=3
weight_loss_mask_time_var=3
weight_loss_classifier=1
weight_loss_mask_entropy=3
gen_aug_type=3
mask_shift=30
mask_option=4 
target_SNR_dB=-12.5
log_name=stage3

for q in 1 5 10 20 40 50 70
do
    echo ----------------------------------
    echo q is $q
    exp_name=binarized_importantAug_$q
    [[ ! -d $save_path/$exp_name ]] &&  mkdir -p $save_path/$exp_name
    python main.py --target_SNR_dB $target_SNR_dB\
               --q $q\
               --save_path_clf $save_path_clf\
               --save_path_clf_frozen $save_path_clf_frozen\
               --no_stage1\
               --no_stage2\
               --no_stage3\
               --stagedev\
               --stage4\
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
               --num_workers $num_workers\
               2>&1 | tee $save_path/$exp_name/$log_name.log
done
