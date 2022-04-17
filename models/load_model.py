from models.classifier import Classifier_CNN
from models.generator import MaskGen_CNN, MaskGen_CNN2

def load_classifier(args):
    if args.arch == 'Classifier_CNN':
        print('Load Classifier configuration')
        model = Classifier_CNN(args.n_class,args.n_feat,args.n_layer_cls,args.kernel_size_cls,args.dropout)
    return model,args

def load_generator(args):
    if args.arch_gen == 'MaskGen_CNN':
        print('Load model')
        gen = MaskGen_CNN(
                args.n_layer_gen,
                1,
                1,
                args.kernel_size_gen,
                stride=1,
                dilation=1)
    elif args.arch_gen == 'MaskGen_CNN2':
        print('Load model')
        gen = MaskGen_CNN2(
                1,
                1,
                args.kernel_size_gen,
                stride=1,
                dilation=1)
    return gen,args