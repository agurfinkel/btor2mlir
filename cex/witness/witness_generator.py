import sys
import os

class CexWitnessGenerator(object):

  def __init__(self, name='CexWitnessGenerator', help='Generate btor witness from SeaHorn cex'):
    self.name = name
    self.help = help
    self.states = list()
    self.inputs = dict()

  def get_value(self, value, width):
    return format(value, '0{}b'.format(width))

  def mk_arg_parser(self, ap):
    ap.add_argument('-o',
                    dest='out_file',
                    metavar='FILE',
                    help='Output file name',
                    default=None)
    ap.add_argument('in_file',
                    metavar='FILE',
                    help='Input file')

    return ap

  def run(self, args=None):
    # default output destination is file `cex.txt`
    if args.out_file is None:
        args.out_file = 'cex.txt'
    print('Creating', args.out_file, '...')
    # keeps track of seen states and inputs
    seenStates = dict()
    seenInputs = dict()
    # keeps track of states and inputs per frame
    states = list()
    inputs = list()
    # register violated assertion
    violatedProperty = 0
    # read each line
    with open(args.in_file, errors='replace') as input:
      for line in input:
        if line.__contains__('[sea]'):
          if not line.__contains__('__VERIFIER_assert'):
            continue
          splitSeaLine = line.split(':')
          violatedProperty = int(splitSeaLine[1].split(',')[0].split()[0])
          continue
        splitLine = line.split(',')
        name = splitLine[0].split()[0]
        btorId = int(splitLine[1].split()[0])
        btorValue = int(splitLine[2].split()[0])
        bvWidth = int(splitLine[3].split()[0])
        # print(f'{name}, {btorId}, {btorValue}, {bvWidth}, {self.get_value(btorValue, bvWidth)}')
        if name == 'state':
          if btorId in seenStates:
            # print('repeat state: ', btorId)
            states.append(seenStates)
            inputs.append(seenInputs)
            seenStates = dict(); seenInputs = dict()
            seenStates[btorId] = self.get_value(btorValue, bvWidth)
            # print(inputs)
            # print(states)
          else:
            # print('got state: ', btorId)
            seenStates[btorId] = self.get_value(btorValue, bvWidth)
            # print(inputs)
            # print(states)
        else:
          if btorId in seenInputs:
            # print('repeat input: ', btorId)
            states.append(seenStates)
            inputs.append(seenInputs)
            seenStates = dict(); seenInputs = dict()
            seenInputs[btorId] = self.get_value(btorValue, bvWidth)
            # print(inputs)
            # print(states)
          else:
            # print('got input: ', btorId)
            seenInputs[btorId] = self.get_value(btorValue, bvWidth)
            # print(inputs)
            # print(states)

    # write to output file
    f = open(args.out_file, 'w')
    frame = 0
    # print header
    f.write(f'sat\n')
    f.write(f'b{violatedProperty}\n') # which property is violated? (b0, b1, j0,...)

    if seenStates or seenInputs:
      states.append(seenStates)
      inputs.append(seenInputs)
    # print(inputs)
    # print(states)
      for (s, i) in zip(states, inputs):
        # print(s, i)
        if s:
          f.write(f'#{frame}\n')
          for k, v in s.items():
            f.write(f'{k} {v}\n')
        if i:
          f.write(f'@{frame}\n')
          for k, v in i.items():
            f.write(f'{k} {v}\n')
        frame += 1
    else:
      for i in range(21):
        f.write(f'#{i}\n')
        f.write(f'@{i}\n')

    f.write('.\n')
    f.close()
    return 0

  def main(self, argv):
    import argparse

    ap = argparse.ArgumentParser(prog=self.name, description=self.help)
    ap = self.mk_arg_parser(ap)

    args = ap.parse_args(argv)
    return self.run(args)

def main():
  cmd = CexWitnessGenerator()
  return cmd.main(sys.argv[1:])


if __name__ == '__main__':
  sys.exit(main())