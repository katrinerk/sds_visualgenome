# sds_visualgenome
##Situation Description Systems applied to Visual Genome data**

**Dependencies:**

pgmax: need to install under Python 3.8

**Setup:**

In the directory with the source code, make a file settings.txt with the following content:
```
[VisualGenome]
VGPATH = <path to visual genome in your system>
VGOBJECTS = v1_2/objects_v1_2.json.zip
VGATTRIB = v1_2/attributes.json.zip
VGREL = v1_2/relationships_v1_2.json.zip
[Parameters]
VGFreqCutoffObj = <frequency cutoff for objects in VG, suggestion 50>
VGFreqCutoffAtt= <frequency cutoff for attributes in VG, suggestion 50>
VGFreqCutoffRel = <frequency cutoff for relations in VG, suggestion 50>
Testpercentage = <fraction of images to use for testing, suggestion 0.1>
```

**General preprocessing steps, in order:**

* vgcounts: Determine sufficiently frequent objects, attributes, relations in the Visual Genome
* train_test_split: Split Visual Genome images into training and test portion
* make_topicmodeling_input
* gensim_topic_modeling: Create scenarios as LDA topic models over VG images
* make_sds_input_noeval: Create input to SDS, not specific to any evaluation task
* change_vgdata_for_vecs: make version of the Visual Genome that only contains images in the training part of the split, retaining only sufficiently frequent objects, attributes, relations


**Situation Description Systems:**

* sds: create factor graphs, run MAP inference on given collection of sentences
* sds_interactive: interactively inspect sentences, make factor graphs, run MAP inference, marginal inference and what-if analyses

**Evaluation tasks:**

* Cloze word sense disambiguation, objects (basically homonymy):
  1. make_sds_input_cloze.py	
  2. sds.py sds_in/cloze sds_out/cloze
  3. eval_cloze
* Object imagination based on scenarios:
  1. make_sds_input_imagine_scen.py
  2. sds.py sds_in/imagine_scen sds_out/imagine_scen
  3. eval_imagine_scen
* Attribute imagination based on object embeddings:
  1. eval_imagine_attr.py
