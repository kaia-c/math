import collections
import matplotlib as plt


def longestCommonPrefix(s1, s2):
    i=0
    while i < len(s1) and i < len(s2) and s1[i]==s2[i]:
        i+=1
    return s1[:i]

def reverseComplement(s):
    #arg=top to bottom side of dna
    #return bottom to top complement on other side
    complement={'A':'T', 'C':'G','G':'C','T':'A'}
    t=''
    for base in s:
        t=complement[base]+t
    return t


def readGenome(fn):
    genome=''
    with open(fn, 'r') as f:
        for line in f:
            if not line[0]=='>':
                genome+=line.rstrip()
    return genome

def readFastq(fn):
    sequences=[]
    qualities=[]
    with open(fn) as f:
        while True:
            f.readline()
            seq=f.readline().rstrip()
            f.readline()
            qual=f.readline().rstrip()
            if len(seq)==0:
                break
            sequences.append(seq)
            qualities.append(qual)
    return sequences, qualities

def phred33ToQ(qual):
    return ord(qual)-33

def createHist(qualList):
    hist=[0]*50
    for qual in qualList:
        for phred in qual:
            hist[phred33ToQ(phred)]+=1
    return hist

def findGCByPos(reads):
    gc=[0]*100
    totals=[0]*100

    for read in reads:
        for i in range(len(read)):
            if read[i]=='C' or read[i]=='G':
                gc[i]+=i
            totals[i]+=1
    for i in range(len(gc)):
        if totals[i]!=0:
            gc[i]/=float(totals[i])
    return gc
"""
print(reverseComplement('ACATTAC'))

print(longestCommonPrefix('ACATTAC', 'ACAGTAGTA'))

genome=readGenome('lambda_virus.fa')
print(genome[:100])
print(len(genome))

counts={'A':0, 'C':0, 'G':0,'T':0}
for base in genome:
    counts[base]+=1
print(counts)
print(collections.Counter(genome))
"""

seqs, quals=readFastq('SRR835775_1.first1000.fastq')
"""
h= createHist(quals)
print(h)
plt.bar(range(len(h)), h)
plt.show()
"""
gc=findGCByPos(seqs)
plt.plot(range(len(gc)), gc)
plt.show()
