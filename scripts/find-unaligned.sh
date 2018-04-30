find /home/csbotos/video/stable_synced/ -name "*.jpg"  -not -path "*/thumbnail/*" -not -path "*/aligned/*" | sort > all_img.txt
find /home/csbotos/video/stable_synced/ -name "*.jpg" -path "*/aligned/*" | sort | sed 's/\/aligned\//\//' > aligned.txt
comm -23 all_img.txt aligned.txt --check-order > unaligned.txt
find /home/csbotos/video/stable_synced/ -name "*.jpg" -path "*/aligned/*" | sort | sed 's/\/aligned\//\//' > aligned.txt
