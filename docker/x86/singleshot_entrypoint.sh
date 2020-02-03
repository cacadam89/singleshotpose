#!/bin/bash
set -e

# start avahi daemon
/etc/init.d/dbus start &>/dev/null
service avahi-daemon start &>/dev/null

cd /root/ssp_ws/src/singleshotpose

# setup ros environment
source "/root/ssp_ws/devel/setup.bash"
export PYTHONPATH="$PYTHONPATH:/root/python3_ws/install/lib/python3/dist-packages"

exec "$@"
