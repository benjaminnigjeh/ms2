from src.CONSTANTS import *

def seq_int(my_seq):
  a = []
  for i in [x for x in my_seq]:
    a.append(PROSIT_ALHABET.get(i))
  a = a + [0]*(30-len([x for x in my_seq]))
  return(a)
