#Written with some assistance from Deepseek AI - particularly in the label_order part.

import sys
import os
import re

LABEL_ORDER = ["alis", "hydra", "mal", "maxus", "mouse"]

folder = sys.argv[1] if len(sys.argv) > 1 else None
path = os.path.abspath(folder)
filenames = os.listdir(path)

#print(os.getcwd()) # Checks the current working directory before cd-ing to another path
os.chdir(path)
#print(os.getcwd()) # checks the current working directory to confirm it did change.

for filename in filenames:
    # splits the path by the . as that is what filename has rather than the full path and takes the first item in the array
    # so maxus_hydra.jpg becomes ["maxus_hydra", ".jpg"] and the first item at the 0th array would be maxus_hydra.
    base_name = os.path.splitext(filename)[0]

    #
    clean_name = re.sub(r'\d+', "", base_name).replace("_", " ").lower()

    # Label is all 0's for however many items is in LABEL_ORDER
    label = [0] * len(LABEL_ORDER)

    # looks through the list in the LABEL_ORDER and takes the name and checks it against the index. 
    # the i is the position. the name is the actual name. for example, maxus is in the 0th position, hydra is in the 1st position..
    # if the name in that index is there, flip the corresponding index in label to 1.
    for i, name in enumerate(LABEL_ORDER):
        if name in clean_name:
            label[i] = 1
    # and if there are no 1's in the label, ie if there are no characters present, then make it all 0's (for stuff like background only images.)
    if 1 not in label:
        label = [0] * len(LABEL_ORDER)
    
    # this takes the label, whcih was an array, and converts each item in the array into a string and adds a space between them. resulting in something like 0 1 0 1 1
    clean_label = ' '.join(map(str, label))

    # new file variable = the base name (maxus_hydra2) + .txt so for example maxus_hydra2.txt
    new_file = f"{base_name}.txt"
    
    #create a file with the new_file name
    with open(new_file, 'w') as f:
        f.write(clean_label)

# When I ran this on the testfilename folder, It produced the correct results.
# Now I just need to make sure when i set the array in the model's training and validation data that it's the same as the label order in this script.