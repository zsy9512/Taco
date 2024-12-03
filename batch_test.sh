for i in {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}
do
for j in {64,128,256,512,1024}
do
echo "test when n_q=$i,bins=$j"
python /home/zsy/workplace/taco/test_vq.py  \
    --resume_path=/home/zsy/workplace/taco/output_001/checkpoints/nq_$i"_"$j/latest.pth --nq $i \
    --bins $j
echo "test done"
done
done
#python3 /home/zsy/workplace/vibstream/vq_test.py --resume_path=/home/zsy/workplace/vibstream/output/checkpoints/vq_20/latest.pth --nq 20