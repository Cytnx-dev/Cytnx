conda config --set anaconda_upload no

OUTPUT_FN=$(conda build conda_gen/ --output)
conda build conda_gen/

anaconda -t $CONDA_UPLOAD_TOKEN upload -u kaihsinwu $OUTPUT_FN --force
