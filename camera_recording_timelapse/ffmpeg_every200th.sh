ffmpeg -an -i "$1" -vf framestep=200,setpts=N/30/TB -r 30 "$1".timelapse.mp4
