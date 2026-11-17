#!/usr/bin/env bash
# shut the VM down 30 min after the given training PID exits
while kill -0 "$1" 2>/dev/null; do sleep 60; done
echo "training exited, shutdown in 30 min" >> ~/train_bag.log
sleep 1800; sudo shutdown -h now
