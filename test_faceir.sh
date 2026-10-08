echo "------------------ CelebA-Test Dataset ------------------"
python inference.py --task faceir \
    --scale 1 --chop_size 256 --chop_stride 256 --bs 16  \
    --config_path configs/faceir_DiT.yaml \
    --ckpt_path weights/faceir.pth \
    -i testdata/CelebA-Test/lq \
    -r testdata/CelebA-Test/hq \
    -o results/CelebA-Test
python metrices/calculate_landmark_distance.py \
    -restored_folder results/CelebA-Test \
    -gt_folder testdata/CelebA-Test/hq
python metrices/calculate_cos_dist.py \
    -restored_folder results/CelebA-Test \
    -gt_folder testdata/CelebA-Test/hq
python metrices/calculate_fid_folder.py \
    -restored_folder results/CelebA-Test \
    --fid_stats testdata/CelebA-Test/hq

echo "------------------ LFW-Test Dataset ------------------"
python inference.py --task faceir \
    --scale 1 --chop_size 256 --chop_stride 256 --bs 16 \
    --config_path configs/faceir_DiT.yaml \
    --ckpt_path weights/faceir.pth \
    -i testdata/LFW-Test/cropped_faces  \
    -o results/LFW-Test

echo "------------------ WebPhoto_Test Dataset ------------------"
python inference.py --task faceir \
    --scale 1 --chop_size 256 --chop_stride 256 --bs 16 \
    --config_path configs/faceir_DiT.yaml \
    --ckpt_path weights/faceir.pth \
    -i testdata/WebPhoto_Test/  \
    -o results/WebPhoto_Test

echo "------------------ WIDER_Test Dataset ------------------"
python inference.py --task faceir \
    --scale 1 --chop_size 256 --chop_stride 256 --bs 16 \
    --config_path configs/faceir_DiT.yaml \
    --ckpt_path weights/faceir.pth \
    -i testdata/WIDER_Test/  \
    -o results/WebPhoto_Test