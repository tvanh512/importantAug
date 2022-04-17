import os
from tools.data_utils import NoisyDataset,data_processing_noisy, data_processing, make_relativepath_index
from torch.utils.data import TensorDataset, DataLoader
from evaluate import evaluate

def evaluate_noisy(args, criterion,
                        device,
                        gen_flag,
                        labels,
                        model,
                        stft_transform,
                        mask_add_aug,
                        mask_forward_eval,
                        mask_loss_eval,
                        save_path,
                        noisy_folder_name_base,
                        noisy_sc_path,
                        SNRdB_list = [-12.5,-10,0,10,20,30,40]):
    print('Evaluation for ',noisy_folder_name_base)
    path_to_index_GSC_dict,_ = make_relativepath_index(args.google_sc_test_txt_path)
    for SNRdB in SNRdB_list:
        noisy_folder_name = noisy_folder_name_base+ '_' + str(SNRdB) +'_dB'
        noisy_sc_root_dir = os.path.join(noisy_sc_path,noisy_folder_name)
        noisy_dataset = NoisyDataset(path_to_index_GSC_dict,noisy_sc_root_dir,stft_transform)
        noisy_loader =  DataLoader(noisy_dataset, args.batch_size,shuffle=False,collate_fn=lambda x:data_processing_noisy(x,labels))
        noisy_acc_test,noisy_loss_test = evaluate(args=args,
                            criterion=criterion,
                            data_loader=noisy_loader,
                            device=device,
                            gen_flag=gen_flag,
                            labels=labels,
                            model=model,
                            stft_transform=stft_transform,
                            mask_add_aug=mask_add_aug,
                            mask_forward_eval=mask_forward_eval,
                            mask_loss_eval = mask_loss_eval,
                            save_path=save_path,
                            )
        print("At SNR %.1f SNRdB, Noisy test acc %.3f, test err %.2f, test loss  %.3f "%(SNRdB,noisy_acc_test,100.0-noisy_acc_test,noisy_loss_test))
    print('End evaluation')