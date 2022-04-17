exp_name=baseline
save_path=./saved_model
[[ ! -d $save_path/$exp_name ]] && mkdir -p $save_path/$exp_name
arch=Classifier_CNN
python main.py --arch $arch\
               --exp_name $exp_name\
               --no_stage2\
               --no_stage3\
               2>&1 | tee $save_path/$exp_name/log.log 