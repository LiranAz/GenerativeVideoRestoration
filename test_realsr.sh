echo "------------------ LSDIR Dataset ------------------"
python inference.py --task realsr --scale 4 \
    --chop_size 512 --chop_stride 448 --bs 16  \
    --config_path configs/realsr_DiT.yaml \
    --ckpt_path weights/realsr.pth \
    -i testdata/LSDIR_Test2K_Center512/lq \
    -r testdata/LSDIR_Test2K_Center512/gt \
    -o results/LSDIR_Test2K_Center512

echo "------------------ RealSR Dataset ------------------"
python inference.py --task realsr --scale 4 \
    --chop_size 512 --chop_stride 448  \
    --config_path configs/realsr_DiT.yaml \
    --ckpt_path weights/realsr.pth \
    -i testdata/RealSR  \
    -o results/RealSR

echo "------------------ RealSet65 Dataset ------------------"
python inference.py --task realsr --scale 4 \
    --chop_size 512 --chop_stride 448 \
    --config_path configs/realsr_DiT.yaml \
    --ckpt_path weights/realsr.pth \
    -i testdata/RealSet65 \
    -o results/RealSet65
