# Set up Mask2Former
CWD=$pwd
cd Mask2Former/mask2former/modeling/pixel_decoder/ops
pwd
sh make.sh
pip install .
cd ${CWD}
pwd