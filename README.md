# sds_visualgenome
# Situation Description Systems applied to Visual Genome data

## Dependencies

pgmax: need to install under Python 3.8

## Setup

In the directory with the source code, make a file `settings.txt` with the following content: (Settings below show use of Visual Genome 1.2 throughout, but should also work with 1.3)
```
[VisualGenome]
VGPATH = <path to visual genome in your system>
VGOBJECTS = <relative path to objects zip, v1_2/objects_v1_2.json.zip>
VGATTRIB = <relative path to attrib. zip, v1_2/attributes.json.zip>
VGREL = <relative path to rel. zip, v1_2/relationships_v1_2.json.zip> 
[Parameters]
VGFreqCutoffObj = <frequency cutoff for objects in VG, suggestion 50>
VGFreqCutoffAtt= <frequency cutoff for attributes in VG, suggestion 50>
VGFreqCutoffRel = <frequency cutoff for relations in VG, suggestion 50>
Testpercentage = <fraction of images to use for testing, suggestion 0.1>
```

## Preprocessing visual genome data

1. Determine sufficiently frequent objects, attributes, relations in the Visual Genome

```python3 vgcounts.py [--vgdata <data_dir default data/>]```

2. Split Visual Genome images into training and test portion

```python3 train_test_split.py [--vgdata <data_dir default data/>]```

3. Make input for LDA topic modeling 

```python3 make_topicmodeling_input.py [--vgdata <data_dir default data/>]```

4. Do LDA topic modeling: Create scenarios as LDA topic models over VG images

```python3 gensim_topic_modeling.py [--vgdata <data_dir default data/>] [--numtopics <num default 20>]```


## Running Situation Description Systems inference

* Creating input to SDS that isn't specific to any evaluation task: This currently uses sentences from the test portion.

```python3 make_sds_input_noeval.py [--output <out_dir default sds_in/vanilla] [--vgdata <data_dir default data/>] [--numsent <num default 100]```

* Running SDS on a pre-stored collection of test sentences, running MAP inference, writing results to output directory

```python3 sds.py <indir> <outdir>```

* Running SDS interactively, inspect sentences and individual literals, run MAP and marginal inference, imagine additional objects and attributes

```python3 sds_interactive.py <indir> [--cloze to look for cloze-generated word labels in indir]```


## Evaluation tasks

* Cloze word sense disambiguation for objects (basically homonymy):
  1. ```python3 make_sds_input_cloze.py	[--output <outdir default sds_in/cloze>] --vgdata <data_dir default data/>]```

  2. ```python3 sds.py sds_in/cloze sds_out/cloze```

  3. ```python3 eval_cloze [--sdsinput <dir default sds_in/cloze>] [--sdsoutput <dir default sds_out/cloze>] [--outdir <dir for qualitative analysis output default inspect_output/cloze>] [--vgdata <data_dir default data/>]```

* Imagining additional objects based on scenarios:

  1. ```python3 make_sds_input_imagine_scen.py [--output <outdir default sds_in/imagine_scen>] --vgdata <data_dir default data/>]```

  2. ```python3 sds.py sds_in/imagine_scen sds_out/imagine_scen```

  3. ```python3 eval_imagine_scen.py [--sdsinput <dir default sds_in/imagine_scen>] [--sdsoutput <dir default sds_out/imagine_scen>] [--outdir <dir for qualitative analysis output default inspect_output/imagine_scen>] [--vgdata <data_dir default data/>]```

* Imagining additional attributes based on object embeddings:
  1. ```python3 eval_imagine_attr.py [--outdir <dir for qualitative analysis output default inspect_output/imagine_attr>] [--vgdata <data_dir default data/>]```

## Transforming Visual Genome data to make vectors for concepts

Keep only images in the training portion, reduce objects/attributes/relations to frequent ones according to `vgcounts.py`

```python3 change_vgdata_for_vecs.py <outputdir> [--vgdata <data_dir default data/>]```
