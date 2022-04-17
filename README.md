## 1.  Introduction

We introduced ImportantAug, a technique to augment training data for speech classification and recognition models by adding noise to unimportant regions of the speech and not to important regions.   
This technique could also be used in computer vision tasks, such as image recognition by predicting importance maps for image

Trinh, Viet Anh, Hassan Salami Kavaki, and Michael I. Mandel. "ImportantAug: a data augmentation agent for speech."  In ICASSP 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2022.

We compare our proposed ImportantAug with other methods: baseline (no augmentation), conventional noise augmentation. We also do ablation study by removing importance map and call this experiment Null ImportantAug. 

In the baseline, we train a recognizer that does not utilize any data augmentation. All other methods are trained by initializing their parameters to those of this pre-trained baseline recognizer.

## 2. Initial set up and data preparation
The best way to train and reprocedure the result is to use the same environment. The custom dataset GSC-QUT is different if create using different environment.
conda env create -f environment.yml

After create the environment, please activate it:
conda activate py39torch19

To prepare data:
cd importantAug
importantAug$./data/prepare_data.sh

## 3. To train from scratch
a. Baseline: 

To train:  
cd importantAug     
importantAug$./scripts/baseline.sh   

To test:   
importantAug$./scripts/baseline_test.sh

b. Conventional noise:   
First, find the best SNR on the dev set:    
importantAug$./scripts/hyper_search_conventional_noise.sh      
To test, use the best SNR in the previous step as the target_SNR_dB in conventional_noise_test.sh and execute:     
importantAug$./scripts/conventional_noise_test.sh

c. ImportantAug:     
To train the generator:     
importantAug$./scripts/importantAug_train_generator.sh      
To do augmentation:      
importantAug$./scripts/importantAug.sh     
To test:      
.importantAug$/scripts/importantAug_test.sh

d. Null ImportantAug:     
To train:     
importantAug$./scripts/null_importantAug.sh     
To test:     
importantAug$./scripts/null_importantAug_test.sh    

e. Binarized ImportantAug
To train:     
importantAug$./scripts/binarized_importantAug.sh   


## 4. To reproduce the result

importantAug$./scripts/pretrained_baseline_test.sh      
importantAug$./scripts/pretrained_conventional_noise_dev.sh     
importantAug$./scripts/pretrained_conventional_noise_test.sh    
importantAug$./scripts/pretrained_importantAugment_test.sh    
importantAug$./scripts/pretrained_null_importantAug_test.sh    
importantAug$./scripts/pretrained_binarized_importantAug_test.sh 







 








