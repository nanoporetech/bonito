# Bonito Draft SAM specification

Bonito v0.5.0 added support for outputing aligned and unaligned SAM, BAM or CRAM.
Output type is triggered by the extension of the specified output file.
All outputs are unsorted.

```
$ bonito basecaller $model $data > unaligned.sam
$ bonito basecaller $model $data > unaligned.bam
$ bonito basecaller $model $data > unaligned.cram

$ bonito basecaller $model $data --reference ref.mmi   > aligned.sam
$ bonito basecaller $model $data --reference ref.mmi   > aligned.bam
$ bonito basecaller $model $data --reference ref.fasta > aligned.cram
```

#### Header

```
@HD     VN:1.5  SO:unknown      ob:0.0.2
@PG     ID:basecaller   PN:bonito       VN:0.4.0        CL:bonito basecaller dna_r9.4.1_e8_fast@v3.3 reads
@PG     ID:aligner      PN:minimap2     VN:2.20 DS:mappy
```

#### Read Group Header

|    |    |                                                       |
| -- | -- | ----------------------------------------------------- |
| RG | ID | `<runid>_<basecalling_model>`  	                  |
|    | PU | `<flow_cell_id>`                                      |
|    | PM | `<device_id>`                                         |
|    | DT | `<exp_start_time>`                                    |
|    | PL | `ONT`                                                 |
|    | DS | `basecall_model=<basecall_model_name> runid=<run_id>` |
|    | LB | `<sample_id>`                                         |
|    | SM | `<sample_id>`                                         |

#### Read Tags

|        |                                                            |
| ------ | -----------------------------------------------------------|
| RG:Z:  | `<runid>_<basecalling_model>`                              |
| qs:i:  | mean basecall qscore rounded to the nearest integer        |
| ns:i:  | the number of samples in the signal prior to trimming      |
| ts:i:  | the number of samples trimmed from the start of the signal |
| mx:i:	 | read mux                                                   |
| ch:i:  | read channel                                               |
| rn:i:	 | read number                                                |
| st:Z:	 | read start time (in UTC)                                   |
| du:f:	 | duration of the read (in seconds)                          |
| f5:Z:	 | fast5 file name                                            |
| sm:f:	 | scaling midpoint/mean/median (pA to ~0-mean/1-sd)          |
| sd:f:	 | scaling dispersion  (pA to ~0-mean/1-sd)                   |
| sv:Z:	 | scaling version                                            |
| mv:B:c | sequence to signal move table _(optional)_                 |

#### Modified Base Tags

When modified base output is requested (via the `--modified-bases` CLI argument), the modified base calls will be output directly in the output files via SAM tags.
The `MM` and `ML` tags are specified in the [SAM format specification documentation](https://samtools.github.io/hts-specs/SAMtags.pdf).
Breifly, these tags represent the relative positions and probability that particular canonical bases have the specified modified bases.

These tags in the SAM/BAM/CRAM formats can be parsed by either the `modbam2bed` or `pysam` software for downstream analysis.
For algined outputs, visualization of these tags is available in popular genome browsers, including IGV and JBrowse.
