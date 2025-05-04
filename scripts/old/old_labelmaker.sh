#!/bin/bash

# for each file in the path
# Take the filename
# remove.png < keep a variable with just that name.
# add .txt after that variable
# from that variable we need to remove the number
# we need an if else statement...

# filename should be something like maxus_hydra2.jpg for images that contain Maxus and Hydra

# Order is: [Maxus Hydra Mouse Alis Mal]

# For Maxus and Hydra, the script should add this to the filename.txt: 1 1 0 0 0

# $1 = the first argument since $0 is the script itself I think
# $1 should be the directory.


### CONSIDER CODING THIS IN PYTHON! IT MIGHT BE EASIER! (Why am I writing this in Bash anyways..)

echo $(realpath $1)

for file in $(find "$1" -type f); do

    # this is the path without the file name and without the last /
    path=${file%/*}

    # This is something like maxus_hydra we'll need this for the label name so we can get maxus_hydra.txt
    filename=$(basename "$file")
    filename="$filename.txt"
    
    # This gets the individual character names
    label=$(echo $filename  | awk -F '[0-9]+' '{print $1}')
    label=$(echo $label | tr _ " ")
    # echo $label

    for i in $label; do
        labellist+=($i)
        # if i = maxus, flip the first 0 to 1. etc

        # for the background logic, if any of the character ones are flipped to 1, then the 6th number should be a 0.

        # else, it would be 0 0 0 0 0 0. (for bg only)

        echo $i
        # echo $label >> $filename

    done
done
exit
