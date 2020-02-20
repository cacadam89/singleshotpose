FROM bramtoula/ros:melodic-trt-py3-tx2-jp430
WORKDIR /root

# Change default shell for RUN per https://stackoverflow.com/questions/20635472/using-the-run-instruction-in-a-dockerfile-with-source-does-not-work/39777387#39777387
SHELL ["/bin/bash", "-c"]


# [IF NOT MOUNTING CODE!!] Clone and build singleshotpose
# RUN source /ros_python3_entrypoint.sh && \
#     mkdir ssp_ws && cd ssp_ws && \
#     mkdir src && \
#     cd  src && \
#     git clone https://github.com/cacadam89/singleshotpose.git singleshotpose && \
#     cd .. && \
#     catkin build

# [ELSE!!] Make directories
# RUN mkdir -p /root/ssp_ws/src/singleshotpose && \
#     cd /root/ssp_ws && \
#     catkin_make
# Scipy
RUN apt-get update && \
    apt-get install -y libatlas-base-dev gfortran && \
    pip3 install wheel pybind11 && \
    pip3 install numpy==1.16 && \
    pip3 install scipy && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y python3-h5py python3-llvmlite python3-pandas python3-numba && \
    pip3 install Cython==0.29.4 colorama==0.3.9 requests==2.21.0 fire==0.1.3 tqdm==4.29.1 tensorboardX==1.6  matplotlib && \
    rm -rf /var/lib/apt/lists/*

# Pytorch and torchvision
RUN apt-get update && \
    apt-get install -y python3-pip libatlas-base-dev git libjpeg-dev zlib1g-dev && \
    rm -rf /var/lib/apt/lists/* 

RUN wget https://nvidia.box.com/shared/static/phqe92v26cbhqjohwtvxorrwnmrnfx1o.whl -O torch-1.3.0-cp36-cp36m-linux_aarch64.whl && \
    pip3 install torch-1.3.0-cp36-cp36m-linux_aarch64.whl && \
    pip3 install pillow==6.2.2 && \
    git clone --branch v0.4.2 https://github.com/pytorch/vision torchvision && \
    cd torchvision && \
    python3 setup.py install && \
    cd .. && \
    rm torch-1.3.0-cp36-cp36m-linux_aarch64.whl && \
    rm -rf torchvision


# # Modify avahi-config
# RUN perl -p -i -e 's|#allow-interfaces=eth0|allow-interfaces=wlp2s0,wlp2s1|g' /etc/avahi/avahi-daemon.conf

# ---------------------------- Reset argument ------------------------------ #
ARG rebuild=0
# -------------------------------------------------------------------------- #

# ---------------------------- DEBUG -start ------------------------------ #
# Install: ping, locate, nano, x11-apps (xclock)
RUN apt-get update && apt-get install -y \
        avahi-utils \
        iputils-ping \
        mlocate \
        nano \
        x11-apps \
    && rm -rf /var/lib/apt/lists/*
# ---------------------------- DEBUG - end ------------------------------ #

# Add aliases to bashrc in docker containers
RUN echo 'alias test_train_go="cd /root/ssp_ws/src/singleshotpose/ && python3 train.py --datacfg cfg/ape.data --modelcfg cfg/yolo-pose.cfg --initweightfile cfg/darknet19_448.conv.23 --pretrain_num_epochs 15"' >> ~/.bashrc && \
    echo 'alias test_valid_go="cd /root/ssp_ws/src/singleshotpose/ && python3 valid.py --datacfg cfg/ape.data --modelcfg cfg/yolo-pose.cfg --weightfile backup/ape/model_backup.weights"' >> ~/.bashrc && \
    echo 'alias test_multi_go="cd /root/ssp_ws/src/singleshotpose/multi_obj_pose_estimation/ && python3 valid_multi.py --modelcfg cfg/yolo-pose-multi.cfg --initweightfile backup_multi/model_backup.weights"' >> ~/.bashrc && \
    echo 'alias train_go="cd /root/ssp_ws/src/singleshotpose/ && python3 -m pdb train.py --datacfg cfg/mslquad.data --modelcfg cfg/yolo-pose.cfg --initweightfile cfg/darknet19_448.conv.23 --pretrain_num_epochs 15"' >> ~/.bashrc && \
    echo 'alias validate_go="cd /root/ssp_ws/src/singleshotpose/ && python3 valid.py --datacfg cfg/mslquad.data --modelcfg cfg/yolo-pose.cfg --weightfile backup/mslquad/model.weights"' >> ~/.bashrc && \
    echo 'alias ssp_go="cd /root/ssp_ws/src/singleshotpose/ && roslaunch singleshotpose ssp_rosbag.launch"' >> ~/.bashrc 
    

# RUN source /ros_python3_entrypoint.sh && \
#     cd ssp_ws/src/singleshotpose && \
#     catkin build

COPY ./singleshot_entrypoint.sh /
ENTRYPOINT ["/singleshot_entrypoint.sh"]
CMD ["bash"]
