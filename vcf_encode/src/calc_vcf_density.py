import argparse
import pysam
import random

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vcf_in',
                        required=True,
                        help="Path to the input VCF file.")

    return parser.parse_args()

def main():
    args = get_args()

    vcf_in = pysam.VariantFile(args.vcf_in)

    sample_names = list(vcf_in.header.samples)
    samples = {}
    for sample in sample_names:
        samples[sample] = [0,0]

    recs = 0
    for rec in vcf_in:
        for sample in rec.samples:
            gt = rec.samples[sample]['GT']
            samples[sample][0] = samples[sample][0] + gt[0]
            samples[sample][1] = samples[sample][1] + gt[1]
        recs += 1

    for sample in samples:
        print(samples[sample][0] / recs)
        print(samples[sample][1] / recs)
    
if __name__ == '__main__':
    main()
