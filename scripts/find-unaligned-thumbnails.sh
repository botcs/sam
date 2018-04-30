find /home/csbotos/video/stable_synced/ -name "*.jpg"  -path "*/thumbnail/*" -not -path "*/aligned/*" | sort > all-thumbnail-img.txt
find /home/csbotos/video/stable_synced/ -name "*.jpg" -path "*/aligned/*" | sort | sed 's/\/aligned\//\//' > aligned.txt
comm -23 all-thumbnail-img.txt aligned.txt --check-order > unaligned-thumbnail.txt
