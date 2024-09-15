from src.CONSTANTS import *
from pyteomics import mass

def seq_int(my_seq: str):
  """This is a function that receive string of peptide sequnce and returns a list of aminoacids
  """
  a = []
  for i in [x for x in my_seq]:
    a.append(PROSIT_ALHABET.get(i))
  a = a + [0]*(130-len([x for x in my_seq]))
  return(a)


def seq_vec(sequence: str):
  """This is a function that receive string of peptides and generate theoretical b y ions
  """
  if len(sequence)>30: sequence=sequence[0:30]
  length = len(sequence)
  a = []
  for i in range(1, length):
    y = sequence[length-i:length]
    a.append(mass.calculate_mass(sequence=y, ion_type='y', charge=1))
    a.append(mass.calculate_mass(sequence=y, ion_type='y', charge=2))
    a.append(mass.calculate_mass(sequence=y, ion_type='y', charge=3))

    b = sequence[0:i]
    a.append(mass.calculate_mass(sequence=b, ion_type='b', charge=1))
    a.append(mass.calculate_mass(sequence=b, ion_type='b', charge=2))
    a.append(mass.calculate_mass(sequence=b, ion_type='b', charge=3))
  a = a + [-1]*(174-len(a))
  return(a)
