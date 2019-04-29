chmod +x /home/oliver/Documents/workplace/project/DAM/src/train_cnli.py  
#CUDA_VISIBLE_DEVICES="" python2 /home/oliver/Documents/workplace/project/DAM/src/train_cnli.py  --model mlp -r 0.0004 --optim adam -e 30 -b 32 --save /home/oliver/Documents/workplace/project/DAM/model_weights/cnli_mlp_new/ 2>/home/oliver/Documents/workplace/project/DAM/log/log_cnli_notrain_lr00004_b32.txt &
CUDA_VISIBLE_DEVICES="" /home/oliver/software/anaconda2/envs/py2.7_tensorflow/bin/python /home/oliver/Documents/workplace/project/DAM/src/train_cnli.py  --model mlp -r 0.0004 --optim adam -e 30 -b 32 --save /home/oliver/Documents/workplace/project/DAM/model_weights/cnli_mlp_basic/ 2>/home/oliver/Documents/workplace/project/DAM/log/log_cnli_lr00004_b32_basic.txt

