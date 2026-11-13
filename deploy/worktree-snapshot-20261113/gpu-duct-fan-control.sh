#!/bin/sh
set -eu
INTERVAL=2
CURRENT=-1
DOWN_COUNT=0
set_fan() { target=$1; [ "$target" -eq "$CURRENT" ] && return; fan-percent "$target" >/dev/null; CURRENT=$target; }
# Calibrated for the P14 Pro on fan2 and both MI50 junction sensors.
set_fan 100
while sleep "$INTERVAL"; do
 max_junction=0; sensors=0
 for bdf in 0000:05:00.0 0000:08:00.0 0000:0b:00.0 0000:0e:00.0; do
  for hwmon in /sys/bus/pci/devices/$bdf/hwmon/hwmon*; do
   [ -r "$hwmon/temp2_input" ] || continue
   junction=$(( $(cat "$hwmon/temp2_input") / 1000 ))
   [ "$junction" -gt "$max_junction" ] && max_junction=$junction
   sensors=$((sensors + 1))
  done
 done
 # Missing sensors are unsafe: keep maximum airflow.
 if [ "$sensors" -ne 4 ]; then set_fan 100; DOWN_COUNT=0; continue; fi
 if [ "$max_junction" -ge 82 ]; then target=100
 elif [ "$max_junction" -ge 75 ]; then target=90
 elif [ "$max_junction" -ge 65 ]; then target=80
 elif [ "$max_junction" -ge 55 ]; then target=65
 elif [ "$max_junction" -ge 45 ]; then target=50
 else target=35
 fi
 if [ "$target" -gt "$CURRENT" ]; then set_fan "$target"; DOWN_COUNT=0
 elif [ "$target" -lt "$CURRENT" ]; then
  DOWN_COUNT=$((DOWN_COUNT + 1))
  # Hold higher speed for 30 seconds to avoid oscillation and heat soak.
  if [ "$DOWN_COUNT" -ge 15 ]; then set_fan "$target"; DOWN_COUNT=0; fi
 else DOWN_COUNT=0
 fi
done
