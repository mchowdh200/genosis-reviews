import argparse
import pysam
from collections import namedtuple
import sys
import random

Segment = namedtuple('Segment', ['chr', 'start', 'end', 'id'])

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m',
                        '--map',
                        required=True, help="Path to the segment map bed file.")
    parser.add_argument('-v',
                        '--vcf',
                        required=True, help="Path to the VCF file.")
    parser.add_argument('-r',
                        '--region',
                        required=True,
                        help="Region to extract from the VCF file.")
    parser.add_argument('-s',
                        '--segment',
                        required=True,
                        help="Segment ID to extract from the segment map file.")
    parser.add_argument('-n',
                        '--number',
                        required=True,
                        type=int,
                        help="Number of samples to generate.")
    parser.add_argument('-d',
                        '--density',
                        type=float,
                        required=True,
                        help="Variant density")
    parser.add_argument('-b',
                        '--binary',
                        type=str,
                        required=True,
                        help="Binary encoding output file name.")
    parser.add_argument('-p',
                        '--position',
                        type=str,
                        required=True,
                        help="Positional encoding output file name.")

    return parser.parse_args()

def get_segment(segment_file, region, segment_id):
    tbx = pysam.TabixFile(segment_file)
    for row in tbx.fetch(region):
        row = row.strip().split('\t')
        segment = Segment(chr=row[0],
                          start=int(row[1]),
                          end=int(row[2]),
                          id=row[3])
        if segment.id == segment_id:
            tbx.close()
            return segment

def main():
    args = get_args()

    segment = get_segment(args.map, args.region, args.segment)
    vcf_in = pysam.VariantFile(args.vcf)

    rel_poss = []

    for rec in vcf_in.fetch(segment.chr, segment.start, segment.end):
        pos = rec.pos
        rel_pos = float(args.segment) \
                + (pos-segment.start)/(segment.end-segment.start)
        rel_poss.append(rel_pos)
    vcf_in.close()

    bf = open(args.binary, 'w')
    pf = open(args.position, 'w')

    for i in range(args.number):
        numbers = sorted(random.sample(range(len(rel_poss)), 10))
        print(numbers)
        curr_gts = bytearray(len(rel_poss))
        curr_pos = ''
        for j in numbers:
            curr_gts[j] = 1
            curr_pos += ' ' + str(rel_poss[j])

        gt_string = ' '.join(str(b) for b in curr_gts)


        bf.write(str(i) + ' ' + gt_string + '\n')
        pf.write(str(i) + ' ' + curr_pos + '\n')

    bf.close()
    pf.close()

#    print(rel_poss)
#
#    with open(args.binary, 'w') as out:
#        for sample in samples:
#            out.write(segment_gt[sample + '_0'] + '\n')
#            out.write(segment_gt[sample + '_1'] + '\n')
#
#    with open(args.position, 'w') as out:
#        for sample in samples:
#            out.write(segment_pos[sample + '_0'] + '\n')
#            out.write(segment_pos[sample + '_1'] + '\n')
#
if __name__ == '__main__':
    main()
