ssp setup
1. make ros ws dir and src folder
2. git clone code into src folder [DO NOT INIT ROS WS]
3. make sure sourcing in entry point is commented out (source "/root/ssp_ws/devel/setup.bash")
4. build and run docker (change paths to match your local computer):
# IF YOU DONT ALREADY HAVE THIS ADD IT:
    export ROS_HOSTNAME=$HOSTNAME
    export ROS_MASTER_URI=http://$HOSTNAME.local:11311
# THEN ADD THE FOLLOWING:
    export SINGLESHOT_PATH="/home/awc11/Documents/ssp_ws/src/singleshotpose"
    single_shot_env_go() { 
        echo 'ROS_HOSTNAME='${ROS_HOSTNAME} > ${SINGLESHOT_PATH}/docker/.env;
        echo 'ROS_MASTER_URI='${ROS_MASTER_URI} >> ${SINGLESHOT_PATH}/docker/.env;
        echo 'DISPLAY'=${DISPLAY} >> ${SINGLESHOT_PATH}/docker/.env; 
        echo 'QT_X11_NO_MITSHM'=1 >> ${SINGLESHOT_PATH}/docker/.env; }
    alias build_singleshot_go='docker build $SINGLESHOT_PATH/docker/x86/ -t singleshotpose'
    alias run_singleshot_go='$(single_shot_env_go) && docker run -it --gpus all --ipc=host --privileged \
                            --volume="/home/awc11/Documents/ssp_ws/:/root/ssp_ws/:rw" \
                            --network host --env-file $SINGLESHOT_PATH/docker/.env --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" -env="XAUTHORITY=$XAUTH" --volume="$XAUTH:$XAUTH" singleshotpose:latest bash'
5. inside docker type source /ros_python3_entrypoint.sh
6. inside docker go to /root/ssp_ws/ and run catkin build
7. repeat previous line while praying to the ros gods
8. exit docker and uncomment source line from entry point
9. try rebuilding / reruning
10. manually add LINEMODE, MSLQUAD, VOCdevkit data sets to the main directly. Also add rosbag(s) into the /rosbag/ directory

