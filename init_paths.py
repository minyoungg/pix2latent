import sys
import os.path as osp


sys.path.append('./utils')
sys.path.append('./nets')

if not osp.exists('./PerceptualSimilarity'):
    print('Could not find LPIPS. Clone the repository from https://github.com/richzhang/PerceptualSimilarity')

sys.path.append('./PerceptualSimilarity')
