import sys
from ECG_READER import EcgReader


if __name__ == '__main__':
    outfile = open(sys.argv[3], 'w')
    outfile.close()
    ecgReader = EcgReader(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
    ecgReader.run()
