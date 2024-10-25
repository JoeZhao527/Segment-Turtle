
import sys, os, distutils.core

dist = distutils.core.run_setup("./detectron2/setup.py")

detectron2_install = ' '.join([f"'{x}'" for x in dist.install_requires])

cmd = f"python -m pip install {detectron2_install}"
print(cmd)
os.system(cmd)

cmd = "python -m pip install -e detectron2"
print(cmd)
os.system(cmd)