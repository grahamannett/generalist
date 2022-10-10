#!/bin/bash

# DOWNLOAD COCO PANOPTIC DATASET
get_panoptic_data() {
    CONFIG_FILE="config/$HOSTNAME.yaml"
    DATA_DIR=$(yq eval '.coco_panoptic' $CONFIG_FILE)
    ZIP_FILE=$DATA_DIR/panoptic_annotations_trainval2017.zip
    DATA_URL=$(yq eval '.coco.panoptic.url' config/datasets.yaml)

    mkdir -p $DATA_DIR

    wget -N -c $DATA_URL -O $ZIP_FILE
    unzip -o $ZIP_FILE -d $DATA_DIR

    # FOLDER STRUCTURE IS WHAT IVE SEEN IN OTHER REPOS
    mv $DATA_DIR/annotations/panoptic_train2017.zip $DATA_DIR/panoptic_train2017.zip
    mv $DATA_DIR/annotations/panoptic_val2017.zip $DATA_DIR/panoptic_val2017.zip

    unzip -o $DATA_DIR/panoptic_train2017.zip -d $DATA_DIR
    unzip -o $DATA_DIR/panoptic_val2017.zip -d $DATA_DIR

    # CLEANUP
    rm -rf $DATA_DIR/__MACOSX
    mkdir $DATA_DIR/zip_files
    mv $DATA_DIR/*.zip $DATA_DIR/zip_files

    echo 'done...'
}

get_panoptic_data
