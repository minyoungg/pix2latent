# biggan examples
CUDA_VISIBLE_DEVICES=0 python ./invert_biggan_adam.py --make_video
CUDA_VISIBLE_DEVICES=0 python ./invert_biggan_cma.py --make_video
CUDA_VISIBLE_DEVICES=0 python ./invert_biggan_basincma.py --make_video
CUDA_VISIBLE_DEVICES=0 python ./invert_biggan_ng.py --make_video
CUDA_VISIBLE_DEVICES=0 python ./invert_biggan_hybrid_ng.py --make_video

# stylegan 2 examples
CUDA_VISIBLE_DEVICES=0 python ./invert_stylegan2_cars_adam.py --make_video
CUDA_VISIBLE_DEVICES=0 python ./invert_stylegan2_cars_cma.py --make_video
CUDA_VISIBLE_DEVICES=0 python ./invert_stylegan2_cars_basincma.py --make_video
CUDA_VISIBLE_DEVICES=0 python ./invert_stylegan2_cars_ng.py --make_video
CUDA_VISIBLE_DEVICES=0 python ./invert_stylegan2_cars_hybrid_ng.py --make_video
