IMG_H5="/srv/workplace/tjurica/asphalt_cnn_alearn/asphalt-images.h5"

OUT_H5_0="/srv/workplace/tjurica/asphalt_cnn_alearn_inference/asphalt-inference-0.h5"
OUT_H5_1="/srv/workplace/tjurica/asphalt_cnn_alearn_inference/asphalt-inference-1.h5"

CUDA_VISIBLE_DEVICES=0 python3.5 mains/infer_h5.py -c configs/config_parallel_0.json --in_h5 ${IMG_H5} --out_h5 ${OUT_H5_0} &
P0=$!

CUDA_VISIBLE_DEVICES=1 python3.5 mains/infer_h5.py -c configs/config_parallel_1.json --in_h5 ${IMG_H5} --out_h5 ${OUT_H5_1} &
P1=$!

wait $P0 $P1