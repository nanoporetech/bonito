# Bonito Draft SAM specification

Bonito v0.5.0 added support for outputing aligned and unaligned SAM, BAM or CRAM. 

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
@HD     VN:1.5  SO:unknown      ob:0.0.1
@PG     ID:basecaller   PN:bonito       VN:0.4.0        CL:bonito basecaller dna_r9.4.1_e8_fast@v3.3 reads
@PG     ID:aligner      PN:minimap2     VN:2.20 DS:mappy
```

#### Read Group Header

|    |    |                                                       |
| -- | -- | ----------------------------------------------------- |            
| RG | ID | `<runid>_<basecalling_model>`  	                      |
|    | PU | `<flow_cell_id>`                                      |	    
|    | PM | `<device_id>`                                         |
|    | DT | `<exp_start_time>`                                    |
|    | PL | `ONT`                                                 |
|    | DS | `basecall_model=<basecall_model_name> runid=<run_id>` |
|    | LB | `<sample_id>`                                         |    
|    | SM | `<sample_id>`                                         |

#### Read Tags

|       |                                                     |
| ----- | --------------------------------------------------- |
| RG:Z: | `<runid>_<basecalling_model>`                        |
| qs:i: | mean basecall qscore rounded to the nearest integer |
| mx:i:	| read mux                                            | 
| ch:i: | read channel                                        |
| rn:i:	| read number                                         |
| st:Z:	| read start time                                     |
| f5:Z:	| fast5 file name                                     |
