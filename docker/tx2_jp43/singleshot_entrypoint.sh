#!/bin/bash
set -e

# start avahi daemon
/etc/init.d/dbus start &>/dev/null
service avahi-daemon start &>/dev/null

cd /root/ssp_ws/src/singleshotpose

# ls /root/ssp_ws
# ls /root/ssp_ws/devel
# ls /root/ssp_ws/devel/setup.bash
# cat /root/ssp_ws/devel/setup.bash

# setup ros environment
# source "/ros_python3_entrypoint.sh"
source "/root/ssp_ws/devel/setup.bash"
export PYTHONPATH="$PYTHONPATH:/root/python3_ws/install/lib/python3/dist-packages"

exec "$@"
