# sds_visualgenome
# Situation Description Systems applied to Visual Genome data

## Dependencies

pgmax: need to install under Python 3.8

## Setup

In the directory with the source code, make a file `settings.txt` with the following content: (Settings below show use of Visual Genome 1.2 throughout, but should also work with 1.3)
```
[VisualGenome]
VGPATH = <path to visual genome>
VGOBJECTS = <relative path to objects zip, v1_2/objects_v1_2.json.zip>
VGATTRIB = <relative path to attrib. zip, v1_2/attributes.json.zip>
VGREL = <relative path to rel. zip, v1_2/relationships_v1_2.json.zip> 
VECPATH = <path to VG-based skipgram vectors>
[Parameters]
VGFreqCutoffObj = <frequency cutoff for objects in VG, suggestion 50>
VGFreqCutoffAtt= <frequency cutoff for attributes in VG, suggestion 50>
VGFreqCutoffRel = <frequency cutoff for relations in VG, suggestion 50>
Testpercentage = <fraction of images to use for testing, suggestion 0.1>
[Scenarios]
InSDS = <heuristic for integrating scenarios: tiled or unary, suggestion unary>
TopScenarios = <num scenarios to keep for tiled heuristic, suggestion 5>
Tilesize = <size of scenario nodes jointly constrained ("tiles") for tiled heuristic, suggestion 6>
Tileoverlap = <overlap of tiles for tiled heuristic, suggestion 2>
NumSamples = <unary heuristic runs Gibbs sampler. Number of samples for Gibbs samples, suggestion 500>
Discard = <unary heuristic, samples to discard during Gibbs sampler burn-in, suggestion 50> 
Restarts = <unary heuristic, number of restarts of Gibbs sampler, suggestion 10>
[Selpref]
Method=<method for computing selectional preferences, suggested relfreq or plsr>
CentroidParam = <for centroid selectional preferences, use top arguments or all arguments? suggested top>

```
Selpref options:
* relfreq: relative frequency
* centroid: centroid of vectors of seen arguments
* plsr: regression to predict #occurrences of an argument with the predicate, basis: vector of argument.
          PLSR to predict #occurrences for an argument for all predicates at once
* 1cclassif: one-class classifier (SVM), hypersphere around vectors of seen arguments, 
          separate classifier for each predicate/role pair
* linreg: regression to predict #occurrences of an argument with the predicate, basis: vector of argument.
          separate model for each predicate/role pair

## Preprocessing visual genome data

1. Determine sufficiently frequent objects, attributes, relations in the Visual Genome

```python3 vgcounts.py [--vgdata <default data/>]```

2. Split Visual Genome images into training and test portion

```python3 train_test_split.py [--vgdata <default data/>]```

3. Make input for LDA topic modeling 

```python3 make_topicmodeling_input.py --filter [--vgdata <default data/>]```

4. Do LDA topic modeling: Create scenarios as LDA topic models over VG images

```python3 gensim_topic_modeling.py [--vgdata <default data/>] [--numtopics <default 20>] [--alpha <default 0.05>]```


## Running Situation Description Systems inference

* Creating input to SDS that isn't specific to any evaluation task: This currently uses sentences from the test portion.

```python3 make_sds_input_noeval.py [--output <default sds_in/vanilla] [--vgdata <default data/>] [--numsent <default 100]```

* Running SDS on a pre-stored collection of test sentences, running MAP inference, writing results to output directory

```python3 sds.py <indir> <outdir>```

* Running SDS interactively, inspect sentences and individual literals, run MAP and marginal inference, imagine additional objects and attributes

```python3 sds_interactive.py <indir> [--cloze]```

* Running SDS on a pre-stored collection of multi-sentence passages with possible coreference, running MAP inference, writing results to output directory

```python3 sdsd.py <indir> <outdir>```

## Evaluation tasks: One component at a time

### Selectional preferences

```python3 eval_selpref.py [--test] [--numpts <default 3000] [--simlevel <default -1>] [--vgdata <default data/>]```

Set selectional preference method in the settings file. Set sim level to choose vector based similarity bin of confounder to observed argument. -1 means no restriction. 0 approximates quasi synonymy, 1 approximates close polysemes, 2 approximates distant polysemes, 3 is homonymy. Choose vector space in settings file.

Set --test to run on test data rather than development data. 

### Scenarios: imagining additional objects based on scenarios

  1. ```python3 make_sds_input_imagine_scen.py [--test] [--output <default sds_in/imagine_scen>] [--vgdata <default data/>] [--numsent <default 1000>] [--maxobj <default 25>] [--testfrac <default 0.3>]```

Set number of topics when calling `gensim_topic_modeling.py`. Set scenario combination heuristic, tiled or unary, in the settings file. Set parameters of tiling or Gibbs sampler in the settings file. 
Set --test to run on test data rather than development data. 


  2. ```python3 sds.py sds_in/imagine_scen sds_out/imagine_scen```

  3. ```python3 eval_imagine_scen.py [--sdsinput <default sds_in/imagine_scen>] [--sdsoutput <default sds_out/imagine_scen>] [--outdir <default inspect_output/imagine_scen>] [--vgdata <default data/>]```
  
This writes performance to screen, and writes examples of inferred objects to the output directory, by default `inspect_output/imagine_scen`

### Property prediction: predicting relative frequency of attributes for objects

```python3 eval_imagine_attr.py [--outdir <default inspect_output/imagine_attr>] [--vgdata <default data/>] [--trainperc <default 0.8>] [--num_att <default 1000>] [--plsr_components <default 100>] [--num_inspect <default 20>] [--test]```

Set number of PLSR components as a parameter to the call. Choose vector space in settings file. Set --test to run on test data rather than development data. 

  
## Evaluation tasks: Multiple components interacting

### Cloze word sense disambiguation:
  1. ```python3 make_sds_input_veccloze.py	 [--output <default sds_in/veccloze>] [--vgdata <default data/>] [--numsent <default 2000>] [--maxlen <default 25>] [--simlevel <default -1>] [--singleword] [--polyfraction <default 1.0>] [--test] [--freqmatch]```

Set sim level to choose vector based similarity bin of confounder to observed argument. -1 means no restriction. 0 approximates quasi synonymy, 1 approximates close polysemes, 2 approximates distant polysemes, 3 is homonymy. Choose --singleword to only have a single ambiguous word per sentence, otherwise it's a random number between 1 and polyfraction * sentencelength.  Set --test to run on test data rather than development data. Set --freqmatch to choose cloze words in the same frequency bin as the target word.

  
  2. ```python3 sds.py sds_in/veccloze sds_out/veccloze```

  3. ```python3 eval_veccloze [--sdsinput <default sds_in/veccloze>] [--sdsoutput <default sds_out/veccloze>] [--outdir <default inspect_output/veccloze>] [--vgdata <default data/>]```

This writes performance to screen, and writes examples of inferred objects to the output directory, by default `inspect_output/veccloze`

### Coreference resolution

To be added. 

Running coreference resolution on a small battery of hand-created sentences, once without synthetic polysemy and once with:

```python3 sdsd_testcases.py```
  
## Utilities

### Turning SDS output to human-readable text

```python3 sds_output_text.py <indir> <outdir> [--cloze] [--allunary]```

Output is written to screen.
Indir, outdir: input and output directory given to the call to `sds.py`. Set --cloze if the data contains cloze words. Set --allunary to see all unary literals, not just unary literals for objects. 

### Looking up word IDs from words

```python3 vglookup.py```

### Transforming Visual Genome data to make vectors for concepts

Keep only images in the training portion, reduce objects/attributes/relations to frequent ones according to `vgcounts.py`

```python3 change_vgdata_for_vecs.py <outputdir> [--vgdata <data_dir default data/>]```
