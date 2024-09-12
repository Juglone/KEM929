ffmpeg -an -i "$1" -vf framestep=20,setpts=N/30/TB -r 30 "$1".presentation.mp4
