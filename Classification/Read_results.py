import argparse
import numpy as np
import os
parser = argparse.ArgumentParser(description='Direct label Transform')
parser.add_argument('--tweak',  type=int, default=0,help=' 0:dirichlet   1:tweak one  2: minority class' )
parser.add_argument('--dataset',  type=str, default='mnist',
                        help='mnist or cifar10')
parser.add_argument('--savingroot', default='./result', help='path to saving.')
args = parser.parse_args()
f = open("./output"+args.dataset+str(args.tweak)+"_Results.txt", 'w+')
if args.tweak == 0:
    alphas_list = [10,1,0.1,0.01]
elif args.tweak==1:
    alphas_list = [0.5,0.6,0.7,0.8,0.9]
else:
    alphas_list = [0.2, 0.3,0.4,0.5]
for alpha in alphas_list:
    classifier_list=['model','baseline','BBSE','RLLS','DLT','BEST']
    for name in classifier_list:
        model_accuracies = np.load(os.path.join(args.savingroot, args.dataset,name+"Accuracy" + 'Alpha' + str(alpha) + str(args.dataset) + ".npy"))
        model_f1s = np.load(os.path.join(args.savingroot, args.dataset,name+"f1" + 'Alpha' + str(alpha) + str(args.dataset) + ".npy"))
        model_recalls = np.load(os.path.join(args.savingroot, args.dataset,name+"recall" + 'Alpha' + str(alpha) + str(args.dataset) + ".npy"))
        print(name+" accuracy  Alpha" + str(alpha) + "  ", model_accuracies,file=f)
        print(name+" accuracy  Alpha" + str(alpha) + "  ", model_accuracies)
        print(name+ "  f1" + str(alpha) + "  ", model_f1s,file=f)
        print(name+ "  f1" + str(alpha) + "  ", model_f1s)
        print(name+ "  recall" + str(alpha) + "  ", model_recalls,file=f)
        print(name+ "  recall" + str(alpha) + "  ", model_recalls)
        model_accuracies = np.load(os.path.join(args.savingroot, args.dataset,"balanced"+name+"Accuracy" + 'Alpha' + str(alpha) + str(args.dataset) + ".npy"))
        model_f1s = np.load(os.path.join(args.savingroot, args.dataset,"balanced"+name+"f1" + 'Alpha' + str(alpha) + str(args.dataset) + ".npy"))
        model_recalls = np.load(os.path.join(args.savingroot, args.dataset,"balanced"+name+"recall" + 'Alpha' + str(alpha) + str(args.dataset) + ".npy"))