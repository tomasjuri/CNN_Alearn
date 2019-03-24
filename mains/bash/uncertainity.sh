INF_H5_0="/srv/workplace/tjurica/asphalt_cnn_alearn_inference/asphalt-inference-0.h5"
INF_H5_1="/srv/workplace/tjurica/asphalt_cnn_alearn_inference/asphalt-inference-1.h5"

UNC_OUT_H5="/srv/workplace/tjurica/asphalt_cnn_alearn_inference/asphalt-uncertainity.h5"

python3.5 mains/uncertainity.py --in_h5 ${INF_H5_0} ${INF_H5_1} --out_h5 ${UNC_OUT_H5}
