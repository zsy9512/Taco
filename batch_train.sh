for i in {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}
do
for j in {64,128,256,512,1024}
do
    python /home/zsy/workplace/taco/train_vq.py  \
    --BATCH_SIZE 32 \
    --N_EPOCHS 500 \
    --seed 123 \
    --save_dir /home/zsy/workplace/taco/output_001/logs  \
    --PATH  /home/zsy/workplace/taco/output_001/checkpoints \
    --train_data_path /home/zsy/workplace/taco/dataset/pwq_or/train \
    --valid_data_path /home/zsy/workplace/taco/dataset/pwq_or/val \
    --log \
    --tensorboard \
    --group_num 2 \
    --bins $j \
    --nq $i  \
    --LAMBDA_SPEC 0.02
done
done