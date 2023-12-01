import glob
import io
from IPython.display import HTML
from IPython import display as ipythondisplay
import base64
def show_video():
  mp4list = glob.glob('video/*.mp4')
  if len(mp4list) > 0:
    mp4 = mp4list[0]
    video = io.open(mp4, 'r+b').read()
    encoded = base64.b64encode(video)
    ipythondisplay.display(HTML(data=''''''.format(encoded.decode('ascii'))))
  else:
    print("Could not find video")

def return_replay(reward_replay,discount_factor):
  cul_reward = 0
  return_list=[]
  for a in range(len(reward_replay)):
    return_list.append(cul_reward*discount_factor + reward_replay[len(reward_replay)-1-a])
    cul_reward = return_list[a]
  return_list.reverse()
  return return_list