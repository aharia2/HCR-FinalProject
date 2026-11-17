#!/usr/bin/env bash
L=$(tail -c 60000 ~/train_bag.log | sed 's/\x1b\[[0-9;]*m//g')
bad=$(grep -E 'Traceback|out of memory|Killed|training exited' ~/train_bag.log | tail -1)
kill -0 $(cat ~/train_bag.pid) 2>/dev/null && st=running || st=EXITED
g(){ echo "$L" | grep -E "$1" | tail -1 | awk -F: '{gsub(/ /,"",$NF); print $NF}'; }
it=$(echo "$L" | grep -oE 'Learning iteration [0-9]+/[0-9]+' | tail -1 | awk '{print $3}')
echo "STATUS $st it=$it ep_len=$(g 'Mean episode length') err_body_pos=$(g 'Metrics/motion/error_body_pos') err_anchor_pos=$(g 'Metrics/motion/error_anchor_pos') term_timeout=$(g 'Episode_Termination/time_out') term_fall_z=$(g 'Episode_Termination/anchor_pos') term_fall_ori=$(g 'Episode_Termination/anchor_ori') ETA=$(echo "$L" | grep -E ' ETA:' | tail -1 | awk '{print $2}') ${bad:+ALERT: $bad}"
