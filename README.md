# VidStabFormer: Full-frame Video Stabilization via Spatial-Temporal Transformers

Offical implementation of "VidStabFormer: Full-frame Video Stabilization via Spatial-Temporal Transformers"

Preparation: 

Clone https://github.com/sniklaus/pytorch-pwc inside model folder. We used a modified version of its backwarp function. 

Follow the CVPR2020 paper implementation of "Yu and Ramamoorthi, 2020" in https://github.com/alex04072000/FuSta/tree/main for creating warping field of any video. 

Put video and warping fields in data folder. You can use "1" for video frames and "1_wf" for warping fields. You can put multiple videos. Add video frame numbers inside test.txt. 

run test.sh for generating the stable video. 

Citation: 

SOON
