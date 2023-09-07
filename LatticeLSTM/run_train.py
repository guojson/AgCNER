
import os

os.system("python main.py --status train --train ./data/Resume/train.char.bmes --dev ./data/Resume/dev.char.bmes "
          "--test ./data/Resume/test.char.bmes --savemodel ./ckpt/saved_model ")
