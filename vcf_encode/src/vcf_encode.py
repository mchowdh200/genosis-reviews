import argparse
import pysam
from collections import namedtuple
import sys

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
    parser.add_argument('-b',
                        '--binary',
                        required=True,
                        help="Binary encoding output file name.")
    parser.add_argument('-p',
                        '--position',
                        required=True,
                        help="Positional encoding output file name.")
    parser.add_argument('-d',
                        '--density',
                        required=True,
                        help="File of density values.")
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
    samples = vcf_in.header.samples

    segment_gt = {}
    segment_pos = {}
    segment_len = {}
    for sample in samples:
        segment_gt[sample + '_0'] = sample + '_0'
        segment_gt[sample + '_1'] = sample + '_1'
        segment_pos[sample + '_0'] = sample + '_0'
        segment_pos[sample + '_1'] = sample + '_1'
        segment_len[sample + '_0'] = 0
        segment_len[sample + '_1'] = 0

    num_pos = 0
    for rec in vcf_in.fetch(segment.chr, segment.start, segment.end):
        pos = rec.pos
        rel_pos = float(args.segment) \
                + (pos-segment.start)/(segment.end-segment.start)
        num_pos += 1
        for sample in samples:
            gt = rec.samples[sample]['GT']
            segment_gt[sample + '_0'] += ' ' + str(gt[0])
            segment_gt[sample + '_1'] += ' ' + str(gt[1])

            if gt[0] == 1:
                segment_pos[sample + '_0'] += ' ' + str(rel_pos)
                segment_len[sample + '_0'] = segment_len[sample + '_0'] + 1

            if gt[1] == 1:
                segment_pos[sample + '_1'] += ' ' + str(rel_pos)
                segment_len[sample + '_1'] = segment_len[sample + '_1'] + 1

    with open(args.binary, 'w') as out:
        for sample in samples:
            out.write(segment_gt[sample + '_0'] + '\n')
            out.write(segment_gt[sample + '_1'] + '\n')

    with open(args.position, 'w') as out:
        for sample in samples:
            out.write(segment_pos[sample + '_0'] + '\n')
            out.write(segment_pos[sample + '_1'] + '\n')

    with open(args.density, 'w') as out:
        for sample in samples:
            out.write(sample + '_0' + ' ' \
                      + str(segment_len[sample + '_0']) + ' ' \
                      + str(segment_len[sample + '_0']/num_pos ) + '\n')
            out.write(sample + '_1' + ' ' \
                      + str(segment_len[sample + '_1']) + ' ' \
                      + str(segment_len[sample + '_1']/num_pos) + '\n')

if __name__ == '__main__':
    main()
